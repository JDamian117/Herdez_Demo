# SupplyChainFinal.py
# Sistema A2A híbrido con LangChain, tolerante a fallos y prompts con 5 patrones.
# Compatible con Streamlit y CLI.
# Modelo recomendado: qwen2.5:0.5b-instruct

from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# LangChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory

# Streamlit (opcional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
APP_TITLE = "Supply Chain A2A Assistant (Fault-Tolerant)"
DB_PATH = os.environ.get("SC_DB_PATH", "long_term_memory.db")
DUCKDB_PATH = os.environ.get("SC_DUCKDB_PATH", "data/herdez.duckdb")
XGB_MODEL_PATH = os.environ.get("SC_XGB_MODEL_PATH", "modelo_xgboost_local.pkl")
ENCODERS_PATH = os.environ.get("SC_ENCODERS_PATH", "encoders.pkl")

DEFAULT_OLLAMA_MODEL = os.environ.get("SC_OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
OLLAMA_URL = os.environ.get("SC_OLLAMA_URL", "http://localhost:11434")

MAX_CHAT_HISTORY = 12
MAX_CONTEXT_CHARS = 5000

# Simulación de fallos (global, se puede cambiar en tiempo de ejecución)
SIMULATE_FAILURE = False

# =============================================================================
# ESQUEMAS PYDANTIC
# =============================================================================
class DemandForecastOutput(BaseModel):
    sku_id: str
    cedi: str
    fecha: str
    demanda_pronosticada_7d: float = Field(ge=0)
    confianza: float = Field(ge=0, le=1)
    metodo: str

class InventoryOutput(BaseModel):
    cedi_destino: str
    cedi_origen_sugerido: str
    stock_disponible_origen: int = Field(ge=0)
    costo_transferencia_unidad: float = Field(ge=0)
    costo_transferencia_total: float = Field(ge=0)

class CostOutput(BaseModel):
    costo_quiebre_total: float = Field(ge=0)
    costo_transferencia_total: float = Field(ge=0)
    ahorro_estimado: float

class DecisionOutput(BaseModel):
    decision: str
    razonamiento: str
    costo_asociado: float = Field(ge=0)
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

def trim_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + "\n...[truncated]..."

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
    candidate = match.group(0).replace("```json", "").replace("```", "")
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(candidate)
    except Exception:
        return None

# =============================================================================
# CLIENTE OLLAMA (FALLBACK)
# =============================================================================
class OllamaClient:
    def __init__(self, model: str, url: str = f"{OLLAMA_URL}/api/chat"):
        self.model = model
        self.url = url

    def chat(self, system: str, user: str, temperature: float = 0.2, num_ctx: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]

# =============================================================================
# MEMORIA SQLITE (LARGO PLAZO)
# =============================================================================
def init_long_term_memory() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT,
            cedi TEXT,
            fecha TEXT,
            demanda_pronosticada REAL,
            stock_actual REAL,
            decision TEXT,
            costo_real REAL,
            razonamiento TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def upgrade_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("PRAGMA table_info(decisiones)")
    columns = [col[1] for col in cursor.fetchall()]
    if "razonamiento" not in columns:
        conn.execute("ALTER TABLE decisiones ADD COLUMN razonamiento TEXT")
        print("✅ Columna 'razonamiento' agregada.")
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

def save_chat_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                 (session_id, role, content, now_iso()))
    conn.commit()
    conn.close()

def get_chat_history(session_id, limit=MAX_CHAT_HISTORY):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                           conn, params=[session_id, limit])
    conn.close()
    rows = df.to_dict(orient="records")
    return list(reversed(rows))

# =============================================================================
# CARGA DE MODELOS DETERMINISTAS (XGBoost)
# =============================================================================
class ModelBundle:
    def __init__(self, model_path=XGB_MODEL_PATH, encoders_path=ENCODERS_PATH):
        self.model_path = model_path
        self.encoders_path = encoders_path
        self._model = None
        self._encoders = None

    @property
    def model(self):
        if self._model is None and os.path.exists(self.model_path):
            self._model = joblib.load(self.model_path)
        return self._model

    @property
    def encoders(self):
        if self._encoders is None and os.path.exists(self.encoders_path):
            self._encoders = joblib.load(self.encoders_path)
        return self._encoders

MODELS = ModelBundle()

def get_connection():
    return duckdb.connect(DUCKDB_PATH)

def ensure_inventory_table():
    conn = get_connection()
    try:
        conn.execute("SELECT 1 FROM inventario_raw LIMIT 1")
    except Exception:
        excel_path = "data/Data_Prueba_Tecnica_Herdez_IA.xlsx"
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            conn.execute("CREATE TABLE inventario_raw AS SELECT * FROM df")
            print("Tabla inventario_raw creada desde Excel.")
        else:
            print("Advertencia: no se encontró el Excel. Las herramientas deterministas usarán fallbacks.")
    finally:
        conn.close()

# =============================================================================
# AGENTES DETERMINISTAS (SIN LLM) - CON FALLBACKS
# =============================================================================
def demand_forecast_tool(sku_id: str, cedi: str, fecha: str, clima: str) -> Dict[str, Any]:
    try:
        conn = get_connection()
        query = """
            SELECT Fecha, Ventas_Unidades, Stock_Actual, Lead_Time_Dias, Promocion_Activa,
                   Precio_Combustible_MXN, Clima, Costo_Quiebre_Stock_Diario, Costo_Transferencia_Unidad
            FROM inventario_raw
            WHERE SKU_ID = ? AND CEDI = ? AND Fecha <= ?
            ORDER BY Fecha DESC
            LIMIT 8
        """
        df_hist = conn.execute(query, [sku_id, cedi, fecha]).df()
        conn.close()

        if df_hist.empty:
            return {"sku_id": sku_id, "cedi": cedi, "fecha": fecha,
                    "demanda_pronosticada_7d": 0.0, "confianza": 0.10, "metodo": "fallback_empty_history"}

        df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)

        if len(df_hist) < 7:
            baseline = float(df_hist["Ventas_Unidades"].tail(min(3, len(df_hist))).mean())
            return {"sku_id": sku_id, "cedi": cedi, "fecha": fecha,
                    "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                    "confianza": 0.35, "metodo": "fallback_short_history"}

        # Feature engineering
        df_hist["ventas_lag_1"] = df_hist["Ventas_Unidades"].shift(1)
        df_hist["ventas_lag_7"] = df_hist["Ventas_Unidades"].shift(7)
        df_hist["ventas_media_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).mean()
        df_hist["ventas_std_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).std()
        df_hist["dias_cobertura"] = df_hist["Stock_Actual"] / (df_hist["Ventas_Unidades"] + 1)
        df_hist["promo_x_ventas"] = df_hist["Promocion_Activa"] * df_hist["Ventas_Unidades"]
        df_hist["leadtime_x_ventas"] = df_hist["Lead_Time_Dias"] * df_hist["Ventas_Unidades"]

        model = MODELS.model
        encoders = MODELS.encoders

        if model is None or encoders is None:
            baseline = float(df_hist["Ventas_Unidades"].tail(7).mean())
            return {"sku_id": sku_id, "cedi": cedi, "fecha": fecha,
                    "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                    "confianza": 0.45, "metodo": "fallback_no_artifacts"}

        def encode_safe(key: str, value: str) -> int:
            try:
                return int(encoders[key].transform([value])[0])
            except Exception:
                return 0

        df_hist["SKU_encoded"] = encode_safe("sku", sku_id)
        df_hist["CEDI_encoded"] = encode_safe("cedi", cedi)
        df_hist["Clima_encoded"] = encode_safe("clima", clima)

        feature_cols = [
            "Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
            "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
            "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
            "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
            "SKU_encoded", "CEDI_encoded", "Clima_encoded",
        ]

        X = df_hist.iloc[-1:][feature_cols].fillna(0)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype("float32")

        pred = float(model.predict(X)[0])
        confidence = 0.85
        std = float(df_hist["ventas_std_7d"].iloc[-1] or 0.0)
        mean = float(df_hist["ventas_media_7d"].iloc[-1] or 1.0)
        ratio = std / max(mean, 1.0)
        confidence = max(0.25, min(0.95, confidence - min(ratio * 0.15, 0.25)))

        return {"sku_id": sku_id, "cedi": cedi, "fecha": fecha,
                "demanda_pronosticada_7d": round(max(0.0, pred), 2),
                "confianza": round(confidence, 2), "metodo": "xgboost_local"}

    except Exception as e:
        return {"sku_id": sku_id, "cedi": cedi, "fecha": fecha,
                "demanda_pronosticada_7d": 0.0, "confianza": 0.0, "metodo": "error", "error": str(e)}

def inventory_tool(sku_id: str, cedi_destino: str, unidades_necesarias: int) -> Dict[str, Any]:
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
            return {"cedi_destino": cedi_destino, "cedi_origen_sugerido": "NO_DISPONIBLE",
                    "stock_disponible_origen": 0, "costo_transferencia_unidad": 0.0, "costo_transferencia_total": 0.0}
        row = df.iloc[0]
        stock = safe_int(row["Stock_Actual"])
        costo_unit = safe_float(row["Costo_Transferencia_Unidad"])
        costo_total = round(costo_unit * max(0, int(unidades_necesarias)), 2)
        return {"cedi_destino": cedi_destino, "cedi_origen_sugerido": str(row["CEDI"]),
                "stock_disponible_origen": stock, "costo_transferencia_unidad": round(costo_unit, 4),
                "costo_transferencia_total": costo_total}
    except Exception as e:
        return {"cedi_destino": cedi_destino, "cedi_origen_sugerido": "ERROR",
                "stock_disponible_origen": 0, "costo_transferencia_unidad": 0.0, "costo_transferencia_total": 0.0,
                "error": str(e)}

def cost_tool(stock_actual: float, demanda: float, costo_quiebre_diario: float,
              costo_transferencia_unidad: float, unidades_necesarias: int) -> Dict[str, Any]:
    dias_quiebre = max(0.0, float(demanda) - float(stock_actual))
    costo_quiebre = max(0.0, dias_quiebre * float(costo_quiebre_diario))
    costo_transferencia = max(0.0, float(unidades_necesarias) * float(costo_transferencia_unidad))
    ahorro = costo_quiebre - costo_transferencia
    return {"costo_quiebre_total": round(costo_quiebre, 2),
            "costo_transferencia_total": round(costo_transferencia, 2),
            "ahorro_estimado": round(ahorro, 2)}

# =============================================================================
# AGENTES CON LANGCHAIN (5 PATRONES + MEMORIA)
# =============================================================================
class DecisionAgent:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, temperature: float = 0.1):
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_URL)
        self.parser = JsonOutputParser(pydantic_object=DecisionOutput)
        # Prompt con los 5 patrones (en inglés)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
# IDENTITY
You are Claudia Mendoza, a logistics director with 15 years of experience in supply chain optimization for food companies.

# MISSION
Your mission is to decide whether to TRANSFER inventory from another CEDI or WAIT for replenishment, based strictly on cost-benefit analysis.

# METHODOLOGY
1. Receive the alert context (SKU, CEDI, stock, demand forecast, costs).
2. Compute deficit = max(0, forecast_demand - current_stock).
3. If deficit == 0 → decision = WAIT.
4. If deficit > 0:
   - Compare transfer_cost_total vs stockout_cost_total.
   - If transfer_cost_total < stockout_cost_total and there is a valid origin CEDI → decision = TRANSFER.
   - Otherwise → decision = WAIT.
5. Output a JSON with decision, concise reasoning, associated cost, recommended origin CEDI (if transfer), units to transfer, and disclaimer.

# LIMITS
- Never invent data. Use only the numbers provided.
- Do not recommend transfer if no valid origin CEDI exists.
- Always include the disclaimer: "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."

# EXAMPLES
Example 1 (transfer): {"decision": "TRANSFERIR", "razonamiento": "Transfer cost MXN 2,528 is much lower than stockout cost MXN 2,250,000.", "costo_asociado": 2528, "cedi_origen_recomendado": "CEDI_Sur", "unidades_a_transferir": 200, "disclaimer": "..."}
Example 2 (wait): {"decision": "ESPERAR", "razonamiento": "Current stock covers demand, no action needed.", "costo_asociado": 0, "cedi_origen_recomendado": null, "unidades_a_transferir": null, "disclaimer": "..."}
"""),
            ("user", "{input}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def decide(self, alerta: 'AlertInput', forecast: Dict[str, Any], inventory: Dict[str, Any], costs: Dict[str, Any]) -> DecisionOutput:
        units = compute_units_needed(forecast.get("demanda_pronosticada_7d", 0.0), alerta.stock_actual)
        deficit = max(0.0, forecast.get("demanda_pronosticada_7d", 0.0) - alerta.stock_actual)
        input_data = {
            "alerta": {
                "sku": alerta.sku_id,
                "cedi": alerta.cedi,
                "stock": alerta.stock_actual,
                "daily_stockout_cost": alerta.costo_quiebre_stock_diario,
                "transfer_cost_per_unit": alerta.costo_transferencia_unidad,
            },
            "demand_7d": forecast.get("demanda_pronosticada_7d", 0.0),
            "best_origin_cedi": inventory.get("cedi_origen_sugerido"),
            "transfer_cost_total": costs.get("costo_transferencia_total", 0.0),
            "stockout_cost_total": costs.get("costo_quiebre_total", 0.0),
            "deficit": deficit,
            "units_needed": units,
        }
        try:
            result = self.chain.invoke({"input": json.dumps(input_data, ensure_ascii=False)})
            decision = DecisionOutput(**result)
        except Exception as e:
            # Fallback a llamada HTTP directa
            fallback_llm = OllamaClient(model=DEFAULT_OLLAMA_MODEL)
            raw = fallback_llm.chat(
                system=self.prompt.messages[0].prompt.template,
                user=json.dumps(input_data, ensure_ascii=False),
                temperature=0.1,
            )
            data = extract_json(raw) or {}
            decision = DecisionOutput(
                decision=str(data.get("decision", "ERROR")).upper(),
                razonamiento=str(data.get("razonamiento", raw[:800])),
                costo_asociado=safe_float(data.get("costo_asociado", 0.0)),
                cedi_origen_recomendado=data.get("cedi_origen_recomendado"),
                unidades_a_transferir=safe_int(data.get("unidades_a_transferir"), 0) if data.get("unidades_a_transferir") is not None else None,
                disclaimer=data.get("disclaimer") or "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística.",
            )
        # Guardrail final
        if decision.decision == "TRANSFERIR" and inventory.get("cedi_origen_sugerido") in (None, "", "NO_DISPONIBLE", "ERROR"):
            decision.decision = "ESPERAR"
            decision.razonamiento += " Guardrail override: no valid origin CEDI was available."
            decision.cedi_origen_recomendado = None
            decision.unidades_a_transferir = None
            decision.costo_asociado = 0.0
        return decision

class ChatAgent:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, temperature: float = 0.35):
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_URL)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        # Prompt para chat con los 5 patrones (en inglés)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
# IDENTITY
You are a supply-chain assistant for Grupo Herdez, specialized in explaining stockout risks, transfer decisions, and cost analysis.

# MISSION
Answer user questions in Spanish, using only the provided context (last alert, decision, costs). Be clear, concise, and helpful.

# METHODOLOGY
1. Read the user's question and the last context (if any).
2. If the question refers to a previous alert, use that data.
3. If data is missing, state it clearly.
4. Provide a short, actionable response in Spanish.
5. Always end with the disclaimer: "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."

# LIMITS
- Do not invent numbers or facts.
- Do not give medical, legal, or financial advice outside supply chain.
- Do not reveal internal chain-of-thought.

# EXAMPLE
User: "¿Por qué se recomendó transferir?"
Assistant: "Se recomendó transferir porque el costo de transferencia era MXN 2,528 mientras que el costo de quiebre era MXN 2,250,000, generando un ahorro de MXN 2,247,472. ... (disclaimer)"
"""),
            ("placeholder", "{history}"),
            ("user", "{input}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, question: str, last_context: Dict[str, Any], session_id: str) -> str:
        # Cargar historial de la base de datos
        db_history = get_chat_history(session_id, MAX_CHAT_HISTORY)
        self.memory.clear()
        for msg in db_history:
            if msg["role"] == "user":
                self.memory.chat_memory.add_user_message(msg["content"])
            else:
                self.memory.chat_memory.add_ai_message(msg["content"])
        context_str = json.dumps({"question": question, "last_context": last_context}, ensure_ascii=False)
        try:
            response = self.chain.invoke({
                "input": context_str,
                "history": self.memory.load_memory_variables({})["history"]
            })
            # Guardar la nueva interacción
            save_chat_message(session_id, "user", question)
            save_chat_message(session_id, "assistant", response)
            return response
        except Exception as e:
            fallback_llm = OllamaClient(model=DEFAULT_OLLAMA_MODEL)
            reply = fallback_llm.chat(
                system="You are a supply-chain assistant. Answer in Spanish, briefly.",
                user=context_str,
                temperature=0.35,
            )
            save_chat_message(session_id, "user", question)
            save_chat_message(session_id, "assistant", reply)
            return reply

# =============================================================================
# ORQUESTADOR (procesar_alerta con tolerancia a fallos)
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
        clima="Despejado",
    )

def compute_units_needed(demand_7d: float, stock_actual: float) -> int:
    return max(0, math.ceil(float(demand_7d) - float(stock_actual)))

def build_last_context(alerta: AlertInput, forecast: Dict[str, Any], inventory: Dict[str, Any], costs: Dict[str, Any], decision: Optional[DecisionOutput] = None) -> Dict[str, Any]:
    return {
        "alerta": {
            "sku_id": alerta.sku_id,
            "cedi": alerta.cedi,
            "fecha": alerta.fecha,
            "stock_actual": alerta.stock_actual,
            "costo_quiebre_stock_diario": alerta.costo_quiebre_stock_diario,
            "costo_transferencia_unidad": alerta.costo_transferencia_unidad,
            "clima": alerta.clima,
        },
        "forecast": forecast,
        "inventory": inventory,
        "costs": costs,
        "decision": decision.model_dump() if decision else None,
    }

def procesar_alerta(alerta: AlertInput, simulate_failure: bool = False) -> Tuple[DecisionOutput, Dict[str, Any], Dict[str, float]]:
    t0 = time.perf_counter()

    # 1. Forecast (con tolerancia)
    try:
        if simulate_failure and random.random() < 0.3:
            raise Exception("Simulated failure: demand_forecast_tool")
        t1 = time.perf_counter()
        forecast = demand_forecast_tool(alerta.sku_id, alerta.cedi, alerta.fecha, alerta.clima)
        t2 = time.perf_counter()
    except Exception as e:
        print(f"⚠️ Forecast agent failed: {e}")
        forecast = {"demanda_pronosticada_7d": 0.0, "confianza": 0.0, "metodo": "fallback_orchestrator", "error": str(e)}
        t2 = t1 = time.perf_counter()

    # 2. Inventory
    try:
        units = compute_units_needed(forecast.get("demanda_pronosticada_7d", 0.0), alerta.stock_actual)
        if simulate_failure and random.random() < 0.3:
            raise Exception("Simulated failure: inventory_tool")
        t3 = time.perf_counter()
        inventory = inventory_tool(alerta.sku_id, alerta.cedi, units)
        t4 = time.perf_counter()
    except Exception as e:
        print(f"⚠️ Inventory agent failed: {e}")
        inventory = {"cedi_origen_sugerido": "ERROR", "costo_transferencia_unidad": alerta.costo_transferencia_unidad, "costo_transferencia_total": 0.0, "error": str(e)}
        t4 = t3 = time.perf_counter()

    # 3. Cost tool (determinista, poco probable que falle)
    try:
        if simulate_failure and random.random() < 0.3:
            raise Exception("Simulated failure: cost_tool")
        t5 = time.perf_counter()
        costs = cost_tool(
            stock_actual=alerta.stock_actual,
            demanda=forecast.get("demanda_pronosticada_7d", 0.0),
            costo_quiebre_diario=alerta.costo_quiebre_stock_diario,
            costo_transferencia_unidad=inventory.get("costo_transferencia_unidad", alerta.costo_transferencia_unidad),
            unidades_necesarias=units,
        )
        t6 = time.perf_counter()
    except Exception as e:
        print(f"⚠️ Cost agent failed: {e}")
        costs = {"costo_quiebre_total": 0.0, "costo_transferencia_total": 0.0, "ahorro_estimado": 0.0, "error": str(e)}
        t6 = t5 = time.perf_counter()

    # 4. Decision agent (LangChain)
    decision_agent = DecisionAgent()
    try:
        if simulate_failure and random.random() < 0.3:
            raise Exception("Simulated failure: DecisionAgent")
        t7 = time.perf_counter()
        decision = decision_agent.decide(alerta, forecast, inventory, costs)
        t8 = time.perf_counter()
    except Exception as e:
        print(f"⚠️ Decision agent failed: {e}")
        decision = DecisionOutput(
            decision="ESPERAR",
            razonamiento=f"El agente decisor falló: {str(e)}. Se recomienda esperar por precaución.",
            costo_asociado=0.0,
            cedi_origen_recomendado=None,
            unidades_a_transferir=None,
            disclaimer="Decisión de respaldo por fallo del sistema."
        )
        t8 = t7 = time.perf_counter()

    # Guardar en memoria
    error_summary = f" | Fallos: forecast={forecast.get('error')}, inventory={inventory.get('error')}, costs={costs.get('error')}"
    guardar_decision(
        sku_id=alerta.sku_id,
        cedi=alerta.cedi,
        fecha=alerta.fecha,
        demanda=forecast.get("demanda_pronosticada_7d", 0.0),
        stock=alerta.stock_actual,
        decision=decision.decision,
        costo=decision.costo_asociado,
        razonamiento=decision.razonamiento + error_summary,
    )

    context = build_last_context(alerta, forecast, inventory, costs, decision)
    timings = {
        "forecast_s": round(t2 - t1, 4),
        "inventory_s": round(t4 - t3, 4),
        "cost_s": round(t6 - t5, 4),
        "decision_s": round(t8 - t7, 4),
        "total_s": round(time.perf_counter() - t0, 4),
    }
    return decision, context, timings

# =============================================================================
# STREAMLIT UI (corregida para evitar página en blanco)
# =============================================================================
def run_streamlit_app():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    st.title("🧠 Supply Chain A2A Assistant (Fault-Tolerant)")
    st.markdown("### Modelo: `qwen2.5:0.5b-instruct` | Tolerancia a fallos | 5 patrones de prompt")

    # Inicializar estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_context" not in st.session_state:
        st.session_state.last_context = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración de Alerta")
        sku = st.text_input("SKU ID", default_alert().sku_id)
        cedi = st.text_input("CEDI", default_alert().cedi)
        fecha = st.date_input("Fecha", datetime.strptime(default_alert().fecha, "%Y-%m-%d"))
        stock = st.number_input("Stock actual (unidades)", min_value=0.0, value=float(default_alert().stock_actual), step=1.0)
        costo_quiebre = st.number_input("Costo de quiebre diario (MXN)", min_value=0.0, value=float(default_alert().costo_quiebre_stock_diario), step=100.0)
        costo_transfer = st.number_input("Costo de transferencia por unidad (MXN)", min_value=0.0, value=float(default_alert().costo_transferencia_unidad), step=0.1)
        clima = st.text_input("Clima", default_alert().clima)
        simulate = st.checkbox("Simular fallos aleatorios", value=False)

        if st.button("🚀 Ejecutar Alerta", use_container_width=True):
            alerta = AlertInput(
                sku_id=sku.strip(),
                cedi=cedi.strip(),
                fecha=fecha.strftime("%Y-%m-%d"),
                stock_actual=stock,
                costo_quiebre_stock_diario=costo_quiebre,
                costo_transferencia_unidad=costo_transfer,
                clima=clima.strip() or "Despejado",
            )
            with st.spinner("Procesando alerta con agentes deterministas y LLM..."):
                decision, context, timings = procesar_alerta(alerta, simulate_failure=simulate)
                st.session_state.last_context = context
                st.success(f"**Decisión: {decision.decision}**")
                st.json(decision.model_dump())
                st.caption(f"Tiempos: {timings}")
                # Añadir al chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Decisión:** {decision.decision}\n\n{decision.razonamiento}\n\n**Costo asociado:** MXN {decision.costo_asociado:,.2f}"
                })
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Área de chat
    st.subheader("💬 Conversación con el Agente")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu pregunta aquí...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_agent = ChatAgent()
        reply = chat_agent.answer(user_input, st.session_state.last_context, st.session_state.session_id)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Mostrar historial de decisiones (opcional)
    with st.expander("📜 Historial de decisiones recientes"):
        hist = recuperar_historico(limit=10)
        if hist:
            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist[["timestamp", "sku_id", "cedi", "decision", "costo_real"]], use_container_width=True)
        else:
            st.write("No hay decisiones guardadas aún.")

# =============================================================================
# CLI (modo texto)
# =============================================================================
def run_cli():
    init_long_term_memory()
    ensure_inventory_table()
    print(f"\n{APP_TITLE}")
    print(f"Modelo: {DEFAULT_OLLAMA_MODEL}")
    print("Comandos: 'alerta' | 'historial' | 'salir'")
    print("También puedes hacer preguntas libres después de generar una alerta.\n")

    last_context = {}
    session_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    simulate = False  # Se puede activar con --simulate-failure, pero simplificamos

    while True:
        user = input("You: ").strip().lower()
        if user in ("salir", "exit"):
            break
        elif user == "alerta":
            alerta = default_alert()
            decision, context, timings = procesar_alerta(alerta, simulate_failure=simulate)
            last_context = context
            print(f"\nDecisión: {decision.decision}")
            print(f"Razón: {decision.razonamiento}")
            print(f"Costo asociado: MXN {decision.costo_asociado:,.2f}")
            print(f"Tiempos: {timings}\n")
        elif user == "historial":
            hist = recuperar_historico(10)
            if not hist:
                print("No hay registros.")
            else:
                for i, row in enumerate(hist, 1):
                    print(f"{i}. {row['timestamp']} | {row['sku_id']} | {row['cedi']} | {row['decision']} | MXN {row['costo_real']:,.2f}")
        else:
            # Pregunta libre
            chat_agent = ChatAgent()
            reply = chat_agent.answer(user, last_context, session_id)
            print(f"\nAssistant: {reply}\n")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    init_long_term_memory()
    ensure_inventory_table()
    # Si se ejecuta con 'streamlit run', forzamos la interfaz UI.
    # También permitimos forzar CLI con el flag '--cli'
    if STREAMLIT_AVAILABLE and not ("--cli" in sys.argv):
        run_streamlit_app()
    else:
        run_cli()