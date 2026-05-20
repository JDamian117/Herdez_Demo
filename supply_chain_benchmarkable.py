# supply_chain_langchain.py
# Sistema híbrido A2A con LangChain para el decisor y el chat.
# Comentarios en español. Prompts internos en inglés.
# Optimizado para hardware limitado (8 GB RAM, CPU antigua).

from __future__ import annotations

import argparse
import json
import math
import os
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

APP_TITLE = "Supply Chain A2A Assistant with LangChain"
DB_PATH = os.environ.get("SC_DB_PATH", "long_term_memory.db")
DUCKDB_PATH = os.environ.get("SC_DUCKDB_PATH", "data/herdez.duckdb")
XGB_MODEL_PATH = os.environ.get("SC_XGB_MODEL_PATH", "modelo_xgboost_local.pkl")
ENCODERS_PATH = os.environ.get("SC_ENCODERS_PATH", "encoders.pkl")

# Modelo base sin LoRA. Cambia este valor para comparar después con una versión afinada.
DEFAULT_OLLAMA_MODEL = os.environ.get("SC_OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
OLLAMA_URL = os.environ.get("SC_OLLAMA_URL", "http://localhost:11434")

MAX_CHAT_HISTORY = 12
MAX_CONTEXT_CHARS = 5000


# =============================================================================
# ESQUEMAS DE SALIDA
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
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def trim_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + "\n...[truncated]..."


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Intenta recuperar un JSON incluso si el modelo agrega texto extra."""
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
    try:
        return json.loads(candidate)
    except Exception:
        candidate = candidate.replace("```json", "").replace("```", "")
        candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            return json.loads(candidate)
        except Exception:
            return None


# =============================================================================
# CLIENTE DE OLLAMA (FALLBACK si LangChain no está disponible, aunque siempre lo está)
# =============================================================================

class OllamaClient:
    """Cliente HTTP directo a Ollama, como fallback (no usado si LangChain funciona)."""
    def __init__(self, model: str, url: str = f"{OLLAMA_URL}/api/chat"):
        self.model = model
        self.url = url

    def chat(self, system: str, user: str, temperature: float = 0.2, num_ctx: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return data["message"]["content"]


# =============================================================================
# MEMORIA (SQLite + LangChain Memory)
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


def guardar_decision(
    sku_id: str,
    cedi: str,
    fecha: str,
    demanda: float,
    stock: float,
    decision: str,
    costo: float,
    razonamiento: str,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO decisiones
        (sku_id, cedi, fecha, demanda_pronosticada, stock_actual, decision, costo_real, razonamiento, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (sku_id, cedi, fecha, demanda, stock, decision, costo, razonamiento, now_iso()))
    conn.commit()
    conn.close()


def recuperar_historico(limit: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT * FROM decisiones
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        conn,
        params=[limit],
    )
    conn.close()
    return df.to_dict(orient="records")


def save_chat_message(session_id: str, role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO chat_messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, role, content, now_iso()),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str, limit: int = MAX_CHAT_HISTORY) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT role, content, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=[session_id, limit],
    )
    conn.close()
    rows = df.to_dict(orient="records")
    return list(reversed(rows))


# =============================================================================
# CARGA DE MODELOS DETERMINISTAS
# =============================================================================

class ModelBundle:
    def __init__(self, model_path: str = XGB_MODEL_PATH, encoders_path: str = ENCODERS_PATH):
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


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DUCKDB_PATH)


def ensure_inventory_table() -> None:
    """Crea la tabla inventario_raw desde el Excel si no existe."""
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
# AGENTES DETERMINISTAS (SIN LLM)
# =============================================================================

def demand_forecast_tool(sku_id: str, cedi: str, fecha: str, clima: str) -> Dict[str, Any]:
    """Agente especializado en pronóstico. No usa LLM."""
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
            return {
                "sku_id": sku_id,
                "cedi": cedi,
                "fecha": fecha,
                "demanda_pronosticada_7d": 0.0,
                "confianza": 0.10,
                "metodo": "fallback_empty_history",
            }

        df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)

        if len(df_hist) < 7:
            baseline = float(df_hist["Ventas_Unidades"].tail(min(3, len(df_hist))).mean())
            return {
                "sku_id": sku_id,
                "cedi": cedi,
                "fecha": fecha,
                "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                "confianza": 0.35,
                "metodo": "fallback_short_history",
            }

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
            return {
                "sku_id": sku_id,
                "cedi": cedi,
                "fecha": fecha,
                "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                "confianza": 0.45,
                "metodo": "fallback_no_artifacts",
            }

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

        return {
            "sku_id": sku_id,
            "cedi": cedi,
            "fecha": fecha,
            "demanda_pronosticada_7d": round(max(0.0, pred), 2),
            "confianza": round(confidence, 2),
            "metodo": "xgboost_local",
        }
    except Exception as e:
        return {
            "sku_id": sku_id,
            "cedi": cedi,
            "fecha": fecha,
            "demanda_pronosticada_7d": 0.0,
            "confianza": 0.0,
            "metodo": "error",
            "error": str(e),
        }


def inventory_tool(sku_id: str, cedi_destino: str, unidades_necesarias: int) -> Dict[str, Any]:
    """Agente especializado en inventario. No usa LLM."""
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
            return {
                "cedi_destino": cedi_destino,
                "cedi_origen_sugerido": "NO_DISPONIBLE",
                "stock_disponible_origen": 0,
                "costo_transferencia_unidad": 0.0,
                "costo_transferencia_total": 0.0,
            }

        row = df.iloc[0]
        stock = safe_int(row["Stock_Actual"])
        costo_unit = safe_float(row["Costo_Transferencia_Unidad"])
        costo_total = round(costo_unit * max(0, int(unidades_necesarias)), 2)

        return {
            "cedi_destino": cedi_destino,
            "cedi_origen_sugerido": str(row["CEDI"]),
            "stock_disponible_origen": stock,
            "costo_transferencia_unidad": round(costo_unit, 4),
            "costo_transferencia_total": costo_total,
        }
    except Exception as e:
        return {
            "cedi_destino": cedi_destino,
            "cedi_origen_sugerido": "ERROR",
            "stock_disponible_origen": 0,
            "costo_transferencia_unidad": 0.0,
            "costo_transferencia_total": 0.0,
            "error": str(e),
        }


def cost_tool(stock_actual: float, demanda: float, costo_quiebre_diario: float,
              costo_transferencia_unidad: float, unidades_necesarias: int) -> Dict[str, Any]:
    """Agente especializado en costos. No usa LLM."""
    dias_quiebre = max(0.0, float(demanda) - float(stock_actual))
    costo_quiebre = max(0.0, dias_quiebre * float(costo_quiebre_diario))
    costo_transferencia = max(0.0, float(unidades_necesarias) * float(costo_transferencia_unidad))
    ahorro = costo_quiebre - costo_transferencia
    return {
        "costo_quiebre_total": round(costo_quiebre, 2),
        "costo_transferencia_total": round(costo_transferencia, 2),
        "ahorro_estimado": round(ahorro, 2),
    }


# =============================================================================
# AGENTES CON LANGCHAIN (DecisionAgent y ChatAgent)
# =============================================================================

class DecisionAgent:
    """Usa LangChain + Ollama para decidir transferir o esperar."""
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, temperature: float = 0.1):
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_URL)
        self.parser = JsonOutputParser(pydantic_object=DecisionOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a supply-chain decision agent.

Identity:
- You are Claudia Mendoza, a logistics director with 15 years of experience.

Mission:
- Decide whether to TRANSFER inventory or WAIT.

Methodology:
1. Read only the provided JSON context.
2. Compute deficit = max(0, forecast demand - current stock).
3. If deficit == 0, decision = WAIT.
4. If deficit > 0:
   - if there is valid origin stock and transfer cost < stockout cost, decision = TRANSFER
   - otherwise decision = WAIT
5. Return concise, numeric, grounded reasoning.
6. Always include the disclaimer.

Limits:
- Do not invent data.
- Do not use external knowledge.
- Return valid JSON only.
- Keep reasoning under 3 short sentences.
- Do not output markdown.

Example:
{{
  "decision": "TRANSFERIR",
  "razonamiento": "Transfer cost is MXN 2,528.00 and stockout cost is MXN 2,250,000.00, so transferring is economically better.",
  "costo_asociado": 2528.0,
  "cedi_origen_recomendado": "CEDI_Sur",
  "unidades_a_transferir": 200,
  "disclaimer": "Recommendation based on economic analysis. It does not replace logistics team judgment."
}}
"""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def decide(self, alerta: 'AlertInput', forecast: Dict[str, Any], inventory: Dict[str, Any], costs: Dict[str, Any]) -> DecisionOutput:
        units = compute_units_needed(forecast["demanda_pronosticada_7d"], alerta.stock_actual)
        deficit = max(0.0, forecast["demanda_pronosticada_7d"] - alerta.stock_actual)

        # Contexto compacto para evitar tokens largos
        input_data = {
            "alerta": {
                "sku": alerta.sku_id,
                "cedi": alerta.cedi,
                "stock": alerta.stock_actual,
                "daily_stockout_cost": alerta.costo_quiebre_stock_diario,
                "transfer_cost_per_unit": alerta.costo_transferencia_unidad,
            },
            "demand_7d": forecast["demanda_pronosticada_7d"],
            "best_origin_cedi": inventory.get("cedi_origen_sugerido"),
            "transfer_cost_total": costs["costo_transferencia_total"],
            "stockout_cost_total": costs["costo_quiebre_total"],
            "deficit": deficit,
            "units_needed": units,
        }

        try:
            result = self.chain.invoke({"input": json.dumps(input_data, ensure_ascii=False)})
            decision = DecisionOutput(**result)
        except Exception as e:
            # Fallback a HTTP directo por si LangChain falla
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
                disclaimer=data.get("disclaimer") or "Recommendation based on economic analysis. It does not replace logistics team judgment.",
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
    """Usa LangChain + Ollama con memoria de conversación."""
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, temperature: float = 0.35):
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_URL)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a supply-chain assistant for a logistics team.

Identity:
- You help explain the latest decision and its numbers.

Mission:
- Answer user questions about stockout risk, transfer decisions, costs, and history.

Methodology:
1. Use only the provided context.
2. If data is missing, say it clearly.
3. Keep the answer concise and useful.
4. Prefer short paragraphs and bullets only if necessary.

Limits:
- Do not invent numbers.
- Do not mention internal chain-of-thought.
- Return plain text, not JSON.

Example:
"According to the latest alert, the transfer was recommended because the transfer cost was lower than the stockout cost."
"""),
            ("placeholder", "{history}"),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, question: str, last_context: Dict[str, Any], session_id: str) -> str:
        # Cargar historial reciente desde SQLite y convertirlo a mensajes de LangChain
        db_history = get_chat_history(session_id, MAX_CHAT_HISTORY)
        history_messages = []
        for msg in db_history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))

        # Actualizar memoria de LangChain
        self.memory.clear()
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                self.memory.chat_memory.add_user_message(msg.content)
            else:
                self.memory.chat_memory.add_ai_message(msg.content)

        context_str = json.dumps({
            "question": question,
            "last_context": last_context,
        }, ensure_ascii=False)

        try:
            response = self.chain.invoke({
                "input": context_str,
                "history": self.memory.load_memory_variables({})["history"]
            })
            return response
        except Exception as e:
            # Fallback a HTTP directo
            fallback_llm = OllamaClient(model=DEFAULT_OLLAMA_MODEL)
            return fallback_llm.chat(
                system="You are a supply-chain assistant. Answer in Spanish, briefly.",
                user=context_str,
                temperature=0.35,
            )


# =============================================================================
# LÓGICA PRINCIPAL Y ORQUESTACIÓN
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


def procesar_alerta(alerta: AlertInput) -> Tuple[DecisionOutput, Dict[str, Any], Dict[str, float]]:
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    forecast = demand_forecast_tool(alerta.sku_id, alerta.cedi, alerta.fecha, alerta.clima)
    t2 = time.perf_counter()

    units = compute_units_needed(forecast["demanda_pronosticada_7d"], alerta.stock_actual)
    inventory = inventory_tool(alerta.sku_id, alerta.cedi, units)
    t3 = time.perf_counter()

    costs = cost_tool(
        stock_actual=alerta.stock_actual,
        demanda=forecast["demanda_pronosticada_7d"],
        costo_quiebre_diario=alerta.costo_quiebre_stock_diario,
        costo_transferencia_unidad=inventory.get("costo_transferencia_unidad", alerta.costo_transferencia_unidad),
        unidades_necesarias=units,
    )
    t4 = time.perf_counter()

    decision_agent = DecisionAgent()
    decision = decision_agent.decide(alerta, forecast, inventory, costs)
    t5 = time.perf_counter()

    guardar_decision(
        sku_id=alerta.sku_id,
        cedi=alerta.cedi,
        fecha=alerta.fecha,
        demanda=forecast["demanda_pronosticada_7d"],
        stock=alerta.stock_actual,
        decision=decision.decision,
        costo=decision.costo_asociado,
        razonamiento=decision.razonamiento,
    )

    context = build_last_context(alerta, forecast, inventory, costs, decision)
    timings = {
        "forecast_s": round(t2 - t1, 4),
        "inventory_s": round(t3 - t2, 4),
        "cost_s": round(t4 - t3, 4),
        "decision_s": round(t5 - t4, 4),
        "total_s": round(t5 - t0, 4),
    }
    return decision, context, timings


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


def load_sample_alerts(n: int = 5) -> List[AlertInput]:
    alerts: List[AlertInput] = []
    if not os.path.exists(DUCKDB_PATH):
        return [default_alert() for _ in range(n)]

    try:
        conn = get_connection()
        df = conn.execute(
            """
            SELECT SKU_ID, CEDI, Fecha, Stock_Actual, Costo_Quiebre_Stock_Diario, Costo_Transferencia_Unidad, Clima
            FROM inventario_raw
            WHERE SKU_ID IS NOT NULL AND CEDI IS NOT NULL
            LIMIT ?
            """,
            [n],
        ).df()
        conn.close()

        if df.empty:
            return [default_alert() for _ in range(n)]

        for _, row in df.iterrows():
            alerts.append(
                AlertInput(
                    sku_id=str(row["SKU_ID"]),
                    cedi=str(row["CEDI"]),
                    fecha=str(row["Fecha"]),
                    stock_actual=safe_float(row["Stock_Actual"]),
                    costo_quiebre_stock_diario=safe_float(row["Costo_Quiebre_Stock_Diario"], 15000.0),
                    costo_transferencia_unidad=safe_float(row["Costo_Transferencia_Unidad"], 12.64),
                    clima=str(row["Clima"]) if pd.notna(row["Clima"]) else "Despejado",
                )
            )
        return alerts if alerts else [default_alert() for _ in range(n)]
    except Exception:
        return [default_alert() for _ in range(n)]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_model(model_name: str, runs_per_alert: int = 3, n_alerts: int = 5, report_path: str = "benchmark_report.csv") -> pd.DataFrame:
    init_long_term_memory()
    # Forzamos el modelo en los agentes (ya usan DEFAULT_OLLAMA_MODEL, pero podemos cambiarlo temporalmente)
    global DEFAULT_OLLAMA_MODEL
    original_model = DEFAULT_OLLAMA_MODEL
    DEFAULT_OLLAMA_MODEL = model_name
    try:
        alerts = load_sample_alerts(n_alerts)
        rows: List[Dict[str, Any]] = []

        for i, alert in enumerate(alerts, 1):
            for r in range(runs_per_alert):
                decision, context, timings = procesar_alerta(alert)
                rows.append({
                    "model": model_name,
                    "alert_index": i,
                    "run_index": r + 1,
                    "sku_id": alert.sku_id,
                    "cedi": alert.cedi,
                    "decision": decision.decision,
                    "costo_asociado": decision.costo_asociado,
                    "forecast_s": timings["forecast_s"],
                    "inventory_s": timings["inventory_s"],
                    "cost_s": timings["cost_s"],
                    "decision_s": timings["decision_s"],
                    "total_s": timings["total_s"],
                    "forecast": context["forecast"]["demanda_pronosticada_7d"],
                    "confidence": context["forecast"].get("confianza", 0.0),
                    "method": context["forecast"].get("metodo", ""),
                    "origin_cedi": context["inventory"].get("cedi_origen_sugerido", ""),
                })
    finally:
        DEFAULT_OLLAMA_MODEL = original_model

    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)
    return df


def summarize_benchmark(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    def q(series: pd.Series, p: float) -> float:
        return float(series.quantile(p))
    return {
        "runs": int(len(df)),
        "avg_total_s": round(float(df["total_s"].mean()), 4),
        "median_total_s": round(float(df["total_s"].median()), 4),
        "p90_total_s": round(q(df["total_s"], 0.90), 4),
        "avg_forecast_s": round(float(df["forecast_s"].mean()), 4),
        "avg_inventory_s": round(float(df["inventory_s"].mean()), 4),
        "avg_cost_s": round(float(df["cost_s"].mean()), 4),
        "avg_decision_s": round(float(df["decision_s"].mean()), 4),
        "transfer_rate": round(float((df["decision"] == "TRANSFERIR").mean()), 4),
    }


# =============================================================================
# UI EN STREAMLIT
# =============================================================================

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = default_alert()
    if "last_context" not in st.session_state:
        st.session_state.last_context = {}
    if "last_decision" not in st.session_state:
        st.session_state.last_decision = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    save_chat_message(st.session_state.session_id, role, content)


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def ui_sidebar() -> AlertInput:
    st.sidebar.header("Alert configuration")
    sku_id = st.sidebar.text_input("SKU ID", st.session_state.last_alert.sku_id)
    cedi = st.sidebar.text_input("CEDI", st.session_state.last_alert.cedi)
    fecha = st.sidebar.text_input("Date (YYYY-MM-DD)", st.session_state.last_alert.fecha)
    stock_actual = st.sidebar.number_input("Current stock", min_value=0.0, value=float(st.session_state.last_alert.stock_actual), step=1.0)
    costo_quiebre = st.sidebar.number_input("Daily stockout cost (MXN)", min_value=0.0, value=float(st.session_state.last_alert.costo_quiebre_stock_diario), step=100.0)
    costo_transfer = st.sidebar.number_input("Transfer cost per unit (MXN)", min_value=0.0, value=float(st.session_state.last_alert.costo_transferencia_unidad), step=0.1)
    clima = st.sidebar.text_input("Weather", st.session_state.last_alert.clima)

    st.sidebar.divider()
    st.sidebar.caption(f"Model: `{DEFAULT_OLLAMA_MODEL}`")
    st.sidebar.caption("Internal prompts: English | UI: Spanish")
    st.sidebar.caption("LangChain used for decision agent + chat memory")

    return AlertInput(
        sku_id=sku_id.strip(),
        cedi=cedi.strip(),
        fecha=fecha.strip(),
        stock_actual=stock_actual,
        costo_quiebre_stock_diario=costo_quiebre,
        costo_transferencia_unidad=costo_transfer,
        clima=clima.strip() or "Despejado",
    )


def run_streamlit_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    init_long_term_memory()
    ensure_inventory_table()
    init_session_state()

    st.title("🧠 Supply Chain A2A Assistant with LangChain")
    st.caption("Deterministic forecast/inventory + LangChain-powered decision & chat (with memory)")

    with st.sidebar:
        current_alert = ui_sidebar()
        run_btn = st.button("Run decision", use_container_width=True)
        sample_btn = st.button("Load sample alert", use_container_width=True)
        benchmark_btn = st.button("Run micro-benchmark", use_container_width=True)
        history_btn = st.button("Show history", use_container_width=True)
        clear_btn = st.button("Clear chat", use_container_width=True)

    if sample_btn:
        st.session_state.last_alert = default_alert()
        st.rerun()

    if clear_btn:
        st.session_state.messages = []
        st.session_state.last_context = {}
        st.session_state.last_decision = None
        st.rerun()

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.subheader("Chat")
        render_chat_history()

        chat_text = st.chat_input("Ask about the last alert, the decision, costs, or history...")
        if chat_text:
            add_message("user", chat_text)
            chat_agent = ChatAgent()
            reply = chat_agent.answer(chat_text, st.session_state.last_context, st.session_state.session_id)
            add_message("assistant", reply)
            st.rerun()

        if run_btn:
            st.session_state.last_alert = current_alert
            with st.spinner("Running A2A pipeline..."):
                decision, context, timings = procesar_alerta(current_alert)
                st.session_state.last_decision = decision
                st.session_state.last_context = context

                add_message(
                    "assistant",
                    (
                        f"Decision: **{decision.decision}**\n\n"
                        f"{decision.razonamiento}\n\n"
                        f"Associated cost: **MXN {decision.costo_asociado:,.2f}**\n\n"
                        f"Timings: `{timings}`"
                    ),
                )
            st.rerun()

        if benchmark_btn:
            with st.spinner("Running quick benchmark..."):
                dfb = benchmark_model(DEFAULT_OLLAMA_MODEL, runs_per_alert=2, n_alerts=3, report_path="benchmark_report.csv")
                summary = summarize_benchmark(dfb)
            st.success("Benchmark completed")
            st.json(summary)
            st.dataframe(dfb, use_container_width=True)

    with col2:
        st.subheader("Latest analysis")
        if st.session_state.last_context:
            fc = st.session_state.last_context.get("forecast", {})
            inv = st.session_state.last_context.get("inventory", {})
            costs = st.session_state.last_context.get("costs", {})
            dec = st.session_state.last_context.get("decision", {})

            st.metric("Forecast (7d)", f"{safe_float(fc.get('demanda_pronosticada_7d', 0.0)):,.2f}")
            st.metric("Confidence", f"{safe_float(fc.get('confianza', 0.0)):.2f}")
            st.metric("Transfer cost", f"MXN {safe_float(costs.get('costo_transferencia_total', 0.0)):,.2f}")
            st.metric("Stockout cost", f"MXN {safe_float(costs.get('costo_quiebre_total', 0.0)):,.2f}")

            st.write("**Inventory**")
            st.json(inv)

            st.write("**Costs**")
            st.json(costs)

            st.write("**Decision**")
            st.json(dec)
        else:
            st.info("Run a decision to see the structured A2A context here.")

        if history_btn:
            st.divider()
            st.write("**Recent decisions**")
            hist = recuperar_historico(limit=10)
            if not hist:
                st.write("No records yet.")
            else:
                dfh = pd.DataFrame(hist)
                st.dataframe(
                    dfh[["timestamp", "sku_id", "cedi", "decision", "demanda_pronosticada", "stock_actual", "costo_real"]],
                    use_container_width=True,
                )

    st.divider()
    st.caption(
        "Baseline model: qwen2.5:0.5b-instruct. LangChain provides chat memory and structured JSON output. "
        "This version is ready to compare against a later LoRA-tuned model."
    )


# =============================================================================
# CLI
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    summary = summarize_benchmark(df)
    print("\n=== BENCHMARK SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nDetailed rows saved in benchmark_report.csv")


def run_cli(model_name: str, benchmark: bool, runs: int, alerts: int) -> None:
    init_long_term_memory()
    ensure_inventory_table()

    if benchmark:
        df = benchmark_model(model_name, runs_per_alert=runs, n_alerts=alerts, report_path="benchmark_report.csv")
        print_summary(df)
        return

    print(f"\n{APP_TITLE}")
    print(f"Model: {model_name}")
    print("Commands: alerta | historial | salir")
    print("You can also ask questions after generating an alert.\n")

    last_context: Dict[str, Any] = {}
    session_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        low = user_input.lower()
        if low in {"salir", "exit", "quit"}:
            print("Bye.")
            break

        if low == "alerta":
            alerta = default_alert()
            decision, last_context, timings = procesar_alerta(alerta)
            print("\nDecision:", decision.decision)
            print("Reason:", decision.razonamiento)
            print(f"Cost: MXN {decision.costo_asociado:,.2f}")
            print("Timings:", timings)
            continue

        if low == "historial":
            hist = recuperar_historico(limit=10)
            if not hist:
                print("No records yet.")
            else:
                for i, row in enumerate(hist, 1):
                    print(f"{i}. {row['timestamp']} | {row['sku_id']} | {row['cedi']} | {row['decision']} | MXN {row['costo_real']:,.2f}")
            continue

        # Pregunta al chat
        chat_agent = ChatAgent()
        reply = chat_agent.answer(user_input, last_context, session_id)
        print("\nAssistant:", reply)
        save_chat_message(session_id, "user", user_input)
        save_chat_message(session_id, "assistant", reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply chain A2A assistant with LangChain")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of interactive CLI.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per alert for benchmark.")
    parser.add_argument("--alerts", type=int, default=5, help="Number of alerts for benchmark.")
    args = parser.parse_args()

    if STREAMLIT_AVAILABLE and os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit_app()
    else:
        run_cli(args.model, args.benchmark, args.runs, args.alerts)


if __name__ == "__main__":
    main()