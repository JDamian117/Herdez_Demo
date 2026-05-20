# supply_chain_hf.py
# Aplicación para cadena de suministro con Hugging Face Inference API (gratuita)
# Despliegue en Streamlit Cloud

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
import streamlit as st

# LangChain + Hugging Face (solo lo necesario)
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
APP_TITLE = "Supply Chain A2A Assistant (Hugging Face)"
DB_PATH = "long_term_memory.db"
DUCKDB_PATH = "data/herdez.duckdb"
XGB_MODEL_PATH = "modelo_xgboost_local.pkl"
ENCODERS_PATH = "encoders.pkl"
EXCEL_PATH = "data/Data_Prueba_Tecnica_Herdez_IA.xlsx"

# Modelo gratuito de Hugging Face
HF_REPO_ID = "microsoft/Phi-3.5-mini-instruct"
HF_TEMPERATURE = 0.1
HF_MAX_TOKENS = 200

# Obtener API key desde secrets de Streamlit (seguro)
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.error("❌ Falta la clave de API de Hugging Face. Configúrala en los secrets de Streamlit.")
    st.stop()

# =============================================================================
# ESQUEMAS PYDANTIC
# =============================================================================
class DemandForecastOutput(BaseModel):
    sku_id: str
    cedi: str
    fecha: str
    demanda_pronosticada_7d: float
    confianza: float = 0.9
    metodo: str

class InventoryOutput(BaseModel):
    cedi_destino: str
    cedi_origen_sugerido: str
    stock_disponible_origen: int
    costo_transferencia_unidad: float
    costo_transferencia_total: float

class CostOutput(BaseModel):
    costo_quiebre_total: float
    costo_transferencia_total: float
    ahorro_estimado: float

class DecisionOutput(BaseModel):
    decision: str
    razonamiento: str
    costo_asociado: float
    cedi_origen_recomendado: Optional[str] = None
    unidades_a_transferir: Optional[int] = None
    disclaimer: str = "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."

# =============================================================================
# UTILIDADES
# =============================================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x)) if x is not None else default
    except Exception:
        return default

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    candidate = candidate.replace("```json", "").replace("```", "")
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(candidate)
    except Exception:
        return None

# =============================================================================
# MEMORIA SQLITE (DECISIONES)
# =============================================================================
def init_long_term_memory():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT, cedi TEXT, fecha TEXT, demanda_pronosticada REAL,
            stock_actual REAL, decision TEXT, costo_real REAL, razonamiento TEXT, timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def guardar_decision(sku_id, cedi, fecha, demanda, stock, decision, costo, razonamiento):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO decisiones (sku_id, cedi, fecha, demanda_pronosticada, stock_actual, decision, costo_real, razonamiento, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sku_id, cedi, fecha, demanda, stock, decision, costo, razonamiento, now_iso()))
    conn.commit()
    conn.close()

def recuperar_historico(limit=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM decisiones ORDER BY timestamp DESC LIMIT ?", conn, params=[limit])
    conn.close()
    return df.to_dict(orient="records")

# =============================================================================
# MODELO XGBOOST Y DUCKDB
# =============================================================================
class ModelBundle:
    def __init__(self):
        self._model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
        self._encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None
    @property
    def model(self):
        return self._model
    @property
    def encoders(self):
        return self._encoders

MODELS = ModelBundle()

def get_connection():
    return duckdb.connect(DUCKDB_PATH)

def ensure_inventory_table():
    if not os.path.exists(DUCKDB_PATH) and os.path.exists(EXCEL_PATH):
        conn = duckdb.connect(DUCKDB_PATH)
        df = pd.read_excel(EXCEL_PATH)
        conn.execute("CREATE TABLE inventario_raw AS SELECT * FROM df")
        conn.close()
        st.info("📁 Base de datos inicializada desde el Excel.")

# =============================================================================
# AGENTES DETERMINISTAS
# =============================================================================
def demand_forecast_tool(sku_id, cedi, fecha, clima) -> Dict[str, Any]:
    try:
        conn = get_connection()
        query = """
            SELECT Fecha, Ventas_Unidades, Stock_Actual, Lead_Time_Dias, Promocion_Activa,
                   Precio_Combustible_MXN, Clima, Costo_Quiebre_Stock_Diario, Costo_Transferencia_Unidad
            FROM inventario_raw
            WHERE SKU_ID = ? AND CEDI = ? AND Fecha <= ?
            ORDER BY Fecha DESC LIMIT 8
        """
        df_hist = conn.execute(query, [sku_id, cedi, fecha]).df()
        conn.close()
        if df_hist.empty:
            return {"demanda_pronosticada_7d": 700.0, "confianza": 0.10, "metodo": "fallback_empty"}
        df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)
        if len(df_hist) < 7:
            baseline = df_hist["Ventas_Unidades"].tail(3).mean()
            return {"demanda_pronosticada_7d": baseline * 7, "confianza": 0.35, "metodo": "fallback_short"}
        model = MODELS.model
        encoders = MODELS.encoders
        if model and encoders:
            df_hist["ventas_lag_1"] = df_hist["Ventas_Unidades"].shift(1)
            df_hist["ventas_lag_7"] = df_hist["Ventas_Unidades"].shift(7)
            df_hist["ventas_media_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).mean()
            df_hist["ventas_std_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).std()
            df_hist["dias_cobertura"] = df_hist["Stock_Actual"] / (df_hist["Ventas_Unidades"] + 1)
            df_hist["promo_x_ventas"] = df_hist["Promocion_Activa"] * df_hist["Ventas_Unidades"]
            df_hist["leadtime_x_ventas"] = df_hist["Lead_Time_Dias"] * df_hist["Ventas_Unidades"]
            sku_enc = encoders["sku"].transform([sku_id])[0]
            cedi_enc = encoders["cedi"].transform([cedi])[0]
            clima_enc = encoders["clima"].transform([clima])[0]
            df_hist["SKU_encoded"] = sku_enc
            df_hist["CEDI_encoded"] = cedi_enc
            df_hist["Clima_encoded"] = clima_enc
            features = [
                "Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
                "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
                "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
                "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
                "SKU_encoded", "CEDI_encoded", "Clima_encoded"
            ]
            X = df_hist.iloc[-1:][features].fillna(0)
            pred = model.predict(X)[0]
            return {"demanda_pronosticada_7d": round(float(pred), 2), "confianza": 0.85, "metodo": "xgboost_local"}
        else:
            demand = df_hist["Ventas_Unidades"].tail(7).mean() * 7
            return {"demanda_pronosticada_7d": round(demand, 2), "confianza": 0.65, "metodo": "moving_average"}
    except Exception as e:
        return {"demanda_pronosticada_7d": 700.0, "confianza": 0.0, "metodo": "error", "error": str(e)}

def inventory_tool(sku_id, cedi_destino, unidades_necesarias) -> Dict[str, Any]:
    try:
        conn = get_connection()
        query = """
            SELECT CEDI, Stock_Actual, Costo_Transferencia_Unidad
            FROM inventario_raw
            WHERE SKU_ID = ? AND CEDI != ?
            ORDER BY Stock_Actual DESC, Costo_Transferencia_Unidad ASC
        """
        df = conn.execute(query, [sku_id, cedi_destino]).df()
        conn.close()
        if df.empty:
            return {"cedi_origen_sugerido": "NO_DISPONIBLE", "stock_disponible_origen": 0,
                    "costo_transferencia_unidad": 0.0, "costo_transferencia_total": 0.0}
        row = df.iloc[0]
        costo_total = row["Costo_Transferencia_Unidad"] * unidades_necesarias
        return {
            "cedi_origen_sugerido": row["CEDI"],
            "stock_disponible_origen": int(row["Stock_Actual"]),
            "costo_transferencia_unidad": row["Costo_Transferencia_Unidad"],
            "costo_transferencia_total": round(costo_total, 2)
        }
    except Exception as e:
        return {"cedi_origen_sugerido": "ERROR", "stock_disponible_origen": 0,
                "costo_transferencia_unidad": 0.0, "costo_transferencia_total": 0.0, "error": str(e)}

def cost_tool(stock_actual, demanda, costo_quiebre_diario, costo_transferencia_unidad, unidades_necesarias) -> Dict[str, Any]:
    deficit = max(0, demanda - stock_actual)
    costo_quiebre = deficit * costo_quiebre_diario
    costo_transferencia = unidades_necesarias * costo_transferencia_unidad
    return {
        "costo_quiebre_total": costo_quiebre,
        "costo_transferencia_total": costo_transferencia,
        "ahorro_estimado": costo_quiebre - costo_transferencia
    }

# =============================================================================
# DECISIONAGENT (LANGCHAIN + HUGGING FACE)
# =============================================================================
class DecisionAgent:
    def __init__(self, repo_id=HF_REPO_ID, temperature=HF_TEMPERATURE):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=HF_MAX_TOKENS,
            temperature=temperature,
        )
        self.parser = JsonOutputParser(pydantic_object=DecisionOutput)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
# IDENTITY
Eres Claudia Mendoza, directora de logística con 15 años de experiencia en cadena de suministro.

# MISSION
Decide si TRANSFERIR inventario desde otro CEDI o ESPERAR el reabastecimiento, basándote en análisis costo-beneficio.

# METHODOLOGY
1. Recibe los datos: SKU, CEDI, stock_actual, demanda_pronosticada_7d, costo_transferencia_total, costo_quiebre_total.
2. Calcula déficit = max(0, demanda - stock).
3. Si déficit == 0 → decisión ESPERAR.
4. Si déficit > 0:
   - Si costo_transferencia_total < costo_quiebre_total y hay un CEDI origen válido → TRANSFERIR.
   - En caso contrario → ESPERAR.
5. Devuelve un JSON con: decision, razonamiento, costo_asociado, cedi_origen_recomendado (si TRANSFERIR), unidades_a_transferir, disclaimer.

# LIMITS
- Nunca inventes datos. Usa solo los números proporcionados.
- No recomiendes transferir si no hay CEDI origen válido.
- Siempre incluye el disclaimer: "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."

# EXAMPLES
Ejemplo con déficit y transferencia rentable:
{"decision": "TRANSFERIR", "razonamiento": "Transferir cuesta 2,528 MXN mientras que el quiebre costaría 2,250,000 MXN. Se ahorran 2,247,472 MXN.", "costo_asociado": 2528.0, "cedi_origen_recomendado": "CEDI_Sur", "unidades_a_transferir": 200, "disclaimer": "..."}

Ejemplo sin déficit:
{"decision": "ESPERAR", "razonamiento": "El stock actual es suficiente para cubrir la demanda, no hay riesgo de quiebre.", "costo_asociado": 0, "disclaimer": "..."}

{format_instructions}
"""),
            ("user", "{input}")
        ])
        self.chain = self.prompt.partial(format_instructions=self.parser.get_format_instructions()) | self.llm | self.parser

    def decide(self, alerta_data: Dict[str, Any]) -> DecisionOutput:
        try:
            result = self.chain.invoke({"input": json.dumps(alerta_data, ensure_ascii=False)})
            return DecisionOutput(**result)
        except Exception as e:
            # Fallback determinista
            if alerta_data.get("costo_transferencia_total", 0) < alerta_data.get("costo_quiebre_total", 0) and alerta_data.get("deficit", 0) > 0:
                return DecisionOutput(
                    decision="TRANSFERIR",
                    razonamiento=f"Transferencia cuesta {alerta_data['costo_transferencia_total']:.2f}, quiebre costaría {alerta_data['costo_quiebre_total']:.2f}. Transferir es más económico.",
                    costo_asociado=alerta_data["costo_transferencia_total"],
                    cedi_origen_recomendado=alerta_data.get("best_origin_cedi"),
                    unidades_a_transferir=alerta_data.get("units_needed"),
                    disclaimer="Recomendación basada en análisis económico."
                )
            else:
                return DecisionOutput(
                    decision="ESPERAR",
                    razonamiento="No hay beneficio económico en transferir o no hay déficit significativo.",
                    costo_asociado=0.0,
                    disclaimer="Recomendación basada en análisis económico."
                )

# =============================================================================
# CHATAGENT (SIN LANGCHAIN MEMORY, USA ST.SESSION_STATE)
# =============================================================================
class ChatAgent:
    def __init__(self, repo_id=HF_REPO_ID, temperature=0.35):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=300,
            temperature=temperature,
        )
    def answer(self, question: str, last_context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        # Convertir historial a string (últimos 6 mensajes)
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        context_str = json.dumps({"question": question, "last_context": last_context}, ensure_ascii=False)
        prompt = f"""Historial reciente de la conversación (role: user o assistant):
{history_str}

Contexto actual de la alerta (última decisión):
{context_str}

Responde en español, de forma clara y útil, usando el contexto si es relevante.
"""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error en el chat: {e}"

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
@dataclass
class AlertInput:
    sku_id: str
    cedi: str
    fecha: str
    stock_actual: float
    costo_quiebre_stock_diario: float
    costo_transferencia_unidad: float
    clima: str = "Despejado"

def default_alert() -> AlertInput:
    return AlertInput(
        sku_id="HZ-Salsa-Verde-200g",
        cedi="CEDI_Norte",
        fecha="2024-03-15",
        stock_actual=50,
        costo_quiebre_stock_diario=15000,
        costo_transferencia_unidad=12.64,
        clima="Despejado"
    )

def compute_units_needed(demand_7d: float, stock_actual: float) -> int:
    return max(0, math.ceil(demand_7d - stock_actual))

def build_alert_data(alerta: AlertInput, forecast: Dict, inventory: Dict, costs: Dict) -> Dict:
    units = compute_units_needed(forecast["demanda_pronosticada_7d"], alerta.stock_actual)
    deficit = max(0, forecast["demanda_pronosticada_7d"] - alerta.stock_actual)
    return {
        "sku": alerta.sku_id,
        "cedi": alerta.cedi,
        "stock_actual": alerta.stock_actual,
        "demanda_7d": forecast["demanda_pronosticada_7d"],
        "costo_transferencia_total": costs["costo_transferencia_total"],
        "costo_quiebre_total": costs["costo_quiebre_total"],
        "deficit": deficit,
        "units_needed": units,
        "best_origin_cedi": inventory["cedi_origen_sugerido"]
    }

def procesar_alerta(alerta: AlertInput) -> Tuple[DecisionOutput, Dict[str, Any], Dict[str, float]]:
    t0 = time.perf_counter()
    # 1. Forecast
    t1 = time.perf_counter()
    forecast = demand_forecast_tool(alerta.sku_id, alerta.cedi, alerta.fecha, alerta.clima)
    t2 = time.perf_counter()
    # 2. Inventory
    units = compute_units_needed(forecast["demanda_pronosticada_7d"], alerta.stock_actual)
    t3 = time.perf_counter()
    inventory = inventory_tool(alerta.sku_id, alerta.cedi, units)
    t4 = time.perf_counter()
    # 3. Costs
    t5 = time.perf_counter()
    costs = cost_tool(alerta.stock_actual, forecast["demanda_pronosticada_7d"],
                      alerta.costo_quiebre_stock_diario,
                      inventory.get("costo_transferencia_unidad", alerta.costo_transferencia_unidad),
                      units)
    t6 = time.perf_counter()
    # 4. Decision
    decision_agent = DecisionAgent()
    alert_data = build_alert_data(alerta, forecast, inventory, costs)
    t7 = time.perf_counter()
    decision = decision_agent.decide(alert_data)
    t8 = time.perf_counter()
    # Guardar
    guardar_decision(
        alerta.sku_id, alerta.cedi, alerta.fecha,
        forecast["demanda_pronosticada_7d"], alerta.stock_actual,
        decision.decision, decision.costo_asociado, decision.razonamiento
    )
    timings = {
        "forecast_s": round(t2 - t1, 4),
        "inventory_s": round(t4 - t3, 4),
        "cost_s": round(t6 - t5, 4),
        "decision_s": round(t8 - t7, 4),
        "total_s": round(t8 - t0, 4)
    }
    context = {
        "alerta": alerta.__dict__,
        "forecast": forecast,
        "inventory": inventory,
        "costs": costs,
        "decision": decision.model_dump()
    }
    return decision, context, timings

# =============================================================================
# GRÁFICAS Y MÉTRICAS
# =============================================================================
def generar_graficas(decision: DecisionOutput, context: Dict) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    alerta = context["alerta"]
    forecast = context["forecast"]
    costs = context["costs"]
    # Gráfico 1: Stock vs Demanda
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    demanda = forecast["demanda_pronosticada_7d"]
    stock = alerta["stock_actual"]
    ax1.bar(["Stock actual", "Demanda 7d"], [stock, demanda],
            color=["blue", "red"] if demanda > stock else ["blue", "green"])
    ax1.set_ylabel("Unidades")
    ax1.set_title("Stock vs Demanda pronosticada")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Gráfico 2: Costos
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    transfer = costs["costo_transferencia_total"]
    quiebre = costs["costo_quiebre_total"]
    ax2.bar(["Costo transferencia", "Costo quiebre"], [transfer, quiebre],
            color=["green", "red"] if transfer < quiebre else ["red", "green"])
    ax2.set_ylabel("MXN")
    ax2.set_title("Comparación de costos")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Gráfico 3: ROI estimado
    ahorro = max(0, quiebre - transfer)
    alertas_anuales = 120
    ahorro_anual = ahorro * alertas_anuales
    inversion = 50000
    roi = ((ahorro_anual - inversion) / inversion) * 100 if inversion > 0 else 0
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(["Ahorro esta alerta", "Ahorro anual\n(120 alertas)"], [ahorro, ahorro_anual], color=["gold", "orange"])
    ax3.set_ylabel("MXN")
    ax3.set_title(f"ROI estimado: {roi:.0f}%")
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    return fig1, fig2, fig3

# =============================================================================
# STREAMLIT UI
# =============================================================================
def run_streamlit():
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    st.title("📦 Cadena de Suministro Inteligente")
    st.markdown("### Agente A2A con Hugging Face (gratuito)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_context" not in st.session_state:
        st.session_state.last_context = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with st.sidebar:
        st.header("⚙️ Configuración de alerta")
        sku = st.text_input("SKU", default_alert().sku_id)
        cedi = st.text_input("CEDI", default_alert().cedi)
        fecha = st.date_input("Fecha", datetime.strptime(default_alert().fecha, "%Y-%m-%d"))
        stock = st.number_input("Stock actual", min_value=0.0, value=float(default_alert().stock_actual), step=1.0)
        costo_q = st.number_input("Costo quiebre diario (MXN)", min_value=0.0, value=float(default_alert().costo_quiebre_stock_diario), step=100.0)
        costo_t = st.number_input("Costo transferencia por unidad (MXN)", min_value=0.0, value=float(default_alert().costo_transferencia_unidad), step=0.1)
        clima = st.text_input("Clima", default_alert().clima)

        if st.button("🚀 Ejecutar alerta", use_container_width=True):
            alerta = AlertInput(
                sku_id=sku.strip(),
                cedi=cedi.strip(),
                fecha=fecha.strftime("%Y-%m-%d"),
                stock_actual=stock,
                costo_quiebre_stock_diario=costo_q,
                costo_transferencia_unidad=costo_t,
                clima=clima.strip() or "Despejado"
            )
            with st.spinner("Analizando (usando Hugging Face)..."):
                decision, context, timings = procesar_alerta(alerta)
                st.session_state.last_context = context
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Decisión:** {decision.decision}\n\n{decision.razonamiento}\n\n**Costo asociado:** MXN {decision.costo_asociado:,.2f}\n\n⏱️ Tiempos: {timings}"
                })
                st.rerun()

        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_context = {}
            st.rerun()

    col1, col2 = st.columns([2, 1.2])
    with col1:
        st.subheader("💬 Conversación")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_input = st.chat_input("Pregunta algo al agente...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            if st.session_state.last_context:
                chat_agent = ChatAgent()
                # Pasamos el historial de mensajes (sin el último, que es el usuario)
                history = st.session_state.messages[:-1]
                reply = chat_agent.answer(user_input, st.session_state.last_context, history)
            else:
                reply = "Primero ejecuta una alerta para tener contexto."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    with col2:
        st.subheader("📊 Análisis gráfico")
        if st.session_state.last_context:
            decision = DecisionOutput(**st.session_state.last_context["decision"])
            fig1, fig2, fig3 = generar_graficas(decision, st.session_state.last_context)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)
            costs = st.session_state.last_context["costs"]
            ahorro = max(0, costs["costo_quiebre_total"] - costs["costo_transferencia_total"])
            st.metric("Ahorro en esta alerta", f"MXN {ahorro:,.2f}")
        else:
            st.info("Ejecuta una alerta para ver las gráficas.")

    with st.expander("📜 Historial de decisiones"):
        hist = recuperar_historico(10)
        if hist:
            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist[["timestamp", "sku_id", "cedi", "decision", "costo_real"]], use_container_width=True)
        else:
            st.write("Aún no hay decisiones guardadas.")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    init_long_term_memory()
    ensure_inventory_table()
    run_streamlit()
