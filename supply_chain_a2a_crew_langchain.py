# supply_chain_a2a_crew_langchain.py
# Hybrid A2A supply-chain assistant for Grupo Herdez.
#
# Key design decisions:
# 1) Keep one local Ollama model for all LLM tasks to reduce RAM usage.
# 2) Use CrewAI for orchestration and task-to-task context passing (A2A style).
# 3) Use LangChain only as a prompt/Chat wrapper for the final chat interface.
# 4) Keep analytical steps deterministic: XGBoost/DuckDB/SQLite.
# 5) Encode the 5 instruction patterns in every agent prompt:
#    Identity, Mission, Methodology, Limits, Examples.
# 6) Prompts are bilingual: English first, Spanish second.
# 7) No DummyLLM is used. The script depends on Ollama being available locally.
#
# Recommended Ollama model for an 8 GB PC:
#   qwen2.5:0.5b-instruct
# If you need slightly better reasoning and still fit comfortably, try:
#   qwen2.5:1.5b-instruct

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool

# Optional LangChain layer for the chat interface.
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

APP_TITLE = "Grupo Herdez - A2A Supply Chain Decision Assistant"

DB_PATH = os.environ.get("SC_DB_PATH", "long_term_memory.db")
DUCKDB_PATH = os.environ.get("SC_DUCKDB_PATH", "data/herdez.duckdb")
XGB_MODEL_PATH = os.environ.get("SC_XGB_MODEL_PATH", "modelo_xgboost_local.pkl")
ENCODERS_PATH = os.environ.get("SC_ENCODERS_PATH", "encoders.pkl")

OLLAMA_MODEL = os.environ.get("SC_OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
OLLAMA_BASE_URL = os.environ.get("SC_OLLAMA_BASE_URL", "http://localhost:11434")

MAX_CHAT_HISTORY = 16
MAX_CONTEXT_CHARS = 6000


# =============================================================================
# Pydantic schemas
# =============================================================================

class DemandForecastOutput(BaseModel):
    sku_id: str
    cedi: str
    fecha: str
    demanda_pronosticada_7d: float = Field(ge=0)
    confianza: float = Field(ge=0, le=1)
    method: Optional[str] = None


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
# General helpers
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
    try:
        return json.loads(candidate)
    except Exception:
        cleaned = candidate.replace("```json", "").replace("```", "")
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def trim_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + "\n...[truncated]..."


# =============================================================================
# Ollama / LangChain / CrewAI model setup
# =============================================================================

def make_crewai_llm(temperature: float) -> LLM:
    # Decision: use the same local Ollama model for every CrewAI agent.
    # That keeps the architecture consistent and avoids loading multiple models.
    return LLM(model=f"ollama/{OLLAMA_MODEL}", temperature=temperature)


def make_langchain_chat_model(temperature: float):
    if not LANGCHAIN_AVAILABLE:
        return None
    # Decision: use LangChain only for the chat UI wrapper and prompt composition.
    return ChatOllama(model=OLLAMA_MODEL, base_url=f"{OLLAMA_BASE_URL}/api", temperature=temperature)


def ollama_chat_http(system: str, user: str, temperature: float = 0.2) -> str:
    # Fallback when LangChain is not installed.
    import urllib.request

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 2048,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed["message"]["content"]


def ollama_is_available() -> bool:
    # Decision: fail fast with a clear error if Ollama is not running.
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False


# =============================================================================
# Memory
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
    """, (
        sku_id, cedi, fecha, demanda, stock, decision, costo, razonamiento, now_iso()
    ))
    conn.commit()
    conn.close()


def recuperar_historico(sku_id: Optional[str] = None, cedi: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM decisiones"
    params: List[Any] = []
    conditions = []
    if sku_id:
        conditions.append("sku_id = ?")
        params.append(sku_id)
    if cedi:
        conditions.append("cedi = ?")
        params.append(cedi)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df.to_dict(orient="records")


def save_chat_message(session_id: str, role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO chat_messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, now_iso()))
    conn.commit()
    conn.close()


def get_chat_history(session_id: str, limit: int = MAX_CHAT_HISTORY) -> List[Dict[str, str]]:
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
# Data/model loading
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


# =============================================================================
# Bilingual prompt builders (5 instruction patterns)
# =============================================================================

def bilingual_block(title_en: str, title_es: str, en: str, es: str) -> str:
    # Decision: keep the instruction format stable across agents.
    # English comes first because small models usually follow English instructions more reliably.
    return (
        f"# {title_en} / {title_es}\n"
        f"EN:\n{en.strip()}\n\n"
        f"ES:\n{es.strip()}\n"
    )


def build_agent_backstory(
    identity_en: str, identity_es: str,
    mission_en: str, mission_es: str,
    methodology_en: str, methodology_es: str,
    limits_en: str, limits_es: str,
    examples_en: str, examples_es: str,
) -> str:
    # Decision: explicitly encode the 5 prompt characteristics requested in the interview.
    return "\n\n".join([
        bilingual_block("IDENTITY", "IDENTIDAD", identity_en, identity_es),
        bilingual_block("MISSION", "MISIÓN", mission_en, mission_es),
        bilingual_block("METHODOLOGY", "METODOLOGÍA", methodology_en, methodology_es),
        bilingual_block("LIMITS", "LÍMITES", limits_en, limits_es),
        bilingual_block("EXAMPLES", "EJEMPLOS", examples_en, examples_es),
    ])


# =============================================================================
# Deterministic tools (A2A specialized agents)
# =============================================================================

class DemandForecastTool(BaseTool):
    name: str = "demand_forecast_tool"
    description: str = "Predict 7-day demand using the local XGBoost model and historical data from DuckDB."

    def _run(self, sku_id: str, cedi: str, fecha: str, clima: str) -> str:
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
                return json.dumps({
                    "sku_id": sku_id,
                    "cedi": cedi,
                    "fecha": fecha,
                    "demanda_pronosticada_7d": 0.0,
                    "confianza": 0.10,
                    "method": "fallback_empty_history",
                }, ensure_ascii=False)

            df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)

            if len(df_hist) < 7:
                baseline = float(df_hist["Ventas_Unidades"].tail(min(3, len(df_hist))).mean())
                return json.dumps({
                    "sku_id": sku_id,
                    "cedi": cedi,
                    "fecha": fecha,
                    "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                    "confianza": 0.35,
                    "method": "fallback_short_history",
                }, ensure_ascii=False)

            # Decision: lightweight feature engineering only; no large embeddings or vector stores.
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
                return json.dumps({
                    "sku_id": sku_id,
                    "cedi": cedi,
                    "fecha": fecha,
                    "demanda_pronosticada_7d": round(max(0.0, baseline * 7.0), 2),
                    "confianza": 0.45,
                    "method": "fallback_no_artifacts",
                }, ensure_ascii=False)

            def enc_value(key: str, value: str) -> int:
                try:
                    return int(encoders[key].transform([value])[0])
                except Exception:
                    return 0

            df_hist["SKU_encoded"] = enc_value("sku", sku_id)
            df_hist["CEDI_encoded"] = enc_value("cedi", cedi)
            df_hist["Clima_encoded"] = enc_value("clima", clima)

            feature_cols = [
                "Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
                "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
                "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
                "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
                "SKU_encoded", "CEDI_encoded", "Clima_encoded"
            ]

            X = df_hist.iloc[-1:][feature_cols].fillna(0)
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].astype("float32")

            pred = float(model.predict(X)[0])

            confidence = 0.85
            std = safe_float(df_hist["ventas_std_7d"].iloc[-1], 0.0)
            mean = safe_float(df_hist["ventas_media_7d"].iloc[-1], 1.0)
            ratio = std / max(mean, 1.0)
            confidence = max(0.25, min(0.95, confidence - min(ratio * 0.15, 0.25)))

            return json.dumps({
                "sku_id": sku_id,
                "cedi": cedi,
                "fecha": fecha,
                "demanda_pronosticada_7d": round(max(0.0, pred), 2),
                "confianza": round(confidence, 2),
                "method": "xgboost_local",
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "sku_id": sku_id,
                "cedi": cedi,
                "fecha": fecha,
                "demanda_pronosticada_7d": 0.0,
                "confianza": 0.0,
                "method": "error",
                "error": str(e),
            }, ensure_ascii=False)


class InventoryTool(BaseTool):
    name: str = "inventory_tool"
    description: str = "Find the best origin CEDI for the requested SKU and destination CEDI."

    def _run(self, sku_id: str, cedi_destino: str, unidades_necesarias: int) -> str:
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
                return json.dumps({
                    "cedi_destino": cedi_destino,
                    "cedi_origen_sugerido": "NO_DISPONIBLE",
                    "stock_disponible_origen": 0,
                    "costo_transferencia_unidad": 0.0,
                    "costo_transferencia_total": 0.0,
                }, ensure_ascii=False)

            row = df.iloc[0]
            stock = safe_int(row["Stock_Actual"])
            costo_unit = safe_float(row["Costo_Transferencia_Unidad"])
            costo_total = round(costo_unit * max(0, int(unidades_necesarias)), 2)

            return json.dumps({
                "cedi_destino": cedi_destino,
                "cedi_origen_sugerido": str(row["CEDI"]),
                "stock_disponible_origen": stock,
                "costo_transferencia_unidad": round(costo_unit, 4),
                "costo_transferencia_total": costo_total,
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "cedi_destino": cedi_destino,
                "cedi_origen_sugerido": "ERROR",
                "stock_disponible_origen": 0,
                "costo_transferencia_unidad": 0.0,
                "costo_transferencia_total": 0.0,
                "error": str(e),
            }, ensure_ascii=False)


class CostTool(BaseTool):
    name: str = "cost_tool"
    description: str = "Calculate stockout and transfer costs."

    def _run(self, stock_actual: float, demanda: float, costo_quiebre_diario: float,
             costo_transferencia_unidad: float, unidades_necesarias: int) -> str:
        dias_quiebre = max(0.0, float(demanda) - float(stock_actual))
        costo_quiebre = max(0.0, dias_quiebre * float(costo_quiebre_diario))
        costo_transferencia = max(0.0, float(unidades_necesarias) * float(costo_transferencia_unidad))
        ahorro = costo_quiebre - costo_transferencia
        return json.dumps({
            "costo_quiebre_total": round(costo_quiebre, 2),
            "costo_transferencia_total": round(costo_transferencia, 2),
            "ahorro_estimado": round(ahorro, 2),
        }, ensure_ascii=False)


# =============================================================================
# CrewAI agents
# =============================================================================

def build_agents() -> Tuple[Agent, Agent, Agent, Agent, LLM]:
    # Decision: one shared model name, different temperatures by role.
    # This preserves a consistent local runtime and reduces operational complexity.
    llm_forecast = make_crewai_llm(0.1)
    llm_inventory = make_crewai_llm(0.1)
    llm_cost = make_crewai_llm(0.1)
    llm_decision = make_crewai_llm(0.05)

    demand_agent = Agent(
        role="Demand Forecaster",
        goal="Predict 7-day demand with reliable local forecasting.",
        backstory=build_agent_backstory(
            identity_en="You are Ana Lopez, a senior data scientist specialized in retail and supply-chain forecasting.",
            identity_es="Eres Ana López, científica de datos senior especializada en pronóstico para retail y cadena de suministro.",
            mission_en="Produce a 7-day demand forecast using the local XGBoost tool and return strict JSON.",
            mission_es="Producir un pronóstico de demanda a 7 días usando la herramienta local XGBoost y devolver JSON estricto.",
            methodology_en=(
                "1. Read the SKU, CEDI, date, and weather.\n"
                "2. Use the demand_forecast_tool.\n"
                "3. Do not invent numbers.\n"
                "4. Return a compact JSON object."
            ),
            methodology_es=(
                "1. Leer el SKU, CEDI, fecha y clima.\n"
                "2. Usar la herramienta demand_forecast_tool.\n"
                "3. No inventar números.\n"
                "4. Regresar un JSON compacto."
            ),
            limits_en=(
                "Never fabricate data. If history is insufficient, accept the tool fallback. "
                "Always keep confidence between 0 and 1."
            ),
            limits_es=(
                "Nunca fabricar datos. Si el histórico es insuficiente, aceptar el fallback de la herramienta. "
                "Mantener la confianza entre 0 y 1."
            ),
            examples_en='Example output: {"sku_id":"HZ-Salsa-Verde-200g","cedi":"CEDI_Norte","fecha":"2024-03-15","demanda_pronosticada_7d":1450.0,"confianza":0.92,"method":"xgboost_local"}',
            examples_es='Salida de ejemplo: {"sku_id":"HZ-Salsa-Verde-200g","cedi":"CEDI_Norte","fecha":"2024-03-15","demanda_pronosticada_7d":1450.0,"confianza":0.92,"method":"xgboost_local"}',
        ),
        verbose=False,
        llm=llm_forecast,
        tools=[DemandForecastTool()],
        allow_delegation=False,
    )

    inventory_agent = Agent(
        role="Inventory Analyst",
        goal="Identify the best origin CEDI for transfer.",
        backstory=build_agent_backstory(
            identity_en="You are Carlos Mendez, an inventory analyst with deep operational experience.",
            identity_es="Eres Carlos Méndez, analista de inventarios con amplia experiencia operativa.",
            mission_en="Select the best origin CEDI using the inventory tool and return strict JSON.",
            mission_es="Seleccionar el mejor CEDI origen usando la herramienta de inventario y devolver JSON estricto.",
            methodology_en=(
                "1. Read the forecast context and SKU.\n"
                "2. Query other CEDIs.\n"
                "3. Sort by highest stock and lowest transfer cost.\n"
                "4. Return the best origin candidate."
            ),
            methodology_es=(
                "1. Leer el contexto del pronóstico y el SKU.\n"
                "2. Consultar otros CEDIs.\n"
                "3. Ordenar por mayor stock y menor costo.\n"
                "4. Devolver el mejor candidato de origen."
            ),
            limits_en=(
                "Never select the destination CEDI as origin. If no valid origin exists, return NO_DISPONIBLE."
            ),
            limits_es=(
                "Nunca seleccionar el CEDI destino como origen. Si no existe un origen válido, devolver NO_DISPONIBLE."
            ),
            examples_en='Example output: {"cedi_destino":"CEDI_Norte","cedi_origen_sugerido":"CEDI_Sur","stock_disponible_origen":800,"costo_transferencia_unidad":12.64,"costo_transferencia_total":2528.0}',
            examples_es='Salida de ejemplo: {"cedi_destino":"CEDI_Norte","cedi_origen_sugerido":"CEDI_Sur","stock_disponible_origen":800,"costo_transferencia_unidad":12.64,"costo_transferencia_total":2528.0}',
        ),
        verbose=False,
        llm=llm_inventory,
        tools=[InventoryTool()],
        allow_delegation=False,
    )

    cost_agent = Agent(
        role="Cost Analyst",
        goal="Compute stockout and transfer costs accurately.",
        backstory=build_agent_backstory(
            identity_en="You are Laura Fernandez, a financial analyst focused on logistics economics.",
            identity_es="Eres Laura Fernández, analista financiera enfocada en la economía logística.",
            mission_en="Quantify stockout cost versus transfer cost and return strict JSON.",
            mission_es="Cuantificar el costo de quiebre versus el costo de transferencia y devolver JSON estricto.",
            methodology_en=(
                "1. Use demand, stock, and per-unit costs.\n"
                "2. Compute stockout cost and transfer cost.\n"
                "3. Keep all values non-negative.\n"
                "4. Return a valid JSON result."
            ),
            methodology_es=(
                "1. Usar demanda, stock y costos unitarios.\n"
                "2. Calcular el costo de quiebre y el costo de transferencia.\n"
                "3. Mantener todos los valores no negativos.\n"
                "4. Regresar un resultado JSON válido."
            ),
            limits_en=(
                "Never return negative costs. Round to two decimals."
            ),
            limits_es=(
                "Nunca devolver costos negativos. Redondear a dos decimales."
            ),
            examples_en='Example output: {"costo_quiebre_total":2250000.0,"costo_transferencia_total":2528.0,"ahorro_estimado":2247472.0}',
            examples_es='Salida de ejemplo: {"costo_quiebre_total":2250000.0,"costo_transferencia_total":2528.0,"ahorro_estimado":2247472.0}',
        ),
        verbose=False,
        llm=llm_cost,
        tools=[CostTool()],
        allow_delegation=False,
    )

    decision_agent = Agent(
        role="Supply Chain Decision Maker",
        goal="Decide transfer vs wait with concise reasoning.",
        backstory=build_agent_backstory(
            identity_en="You are Claudia Mendoza, a logistics director with strong decision-making experience.",
            identity_es="Eres Claudia Mendoza, directora de logística con gran experiencia en toma de decisiones.",
            mission_en="Use the previous task outputs to decide TRANSFER or WAIT and explain the numbers.",
            mission_es="Usar los resultados previos para decidir TRANSFERIR o ESPERAR y explicar los números.",
            methodology_en=(
                "1. Read the forecast, inventory, and cost JSON.\n"
                "2. Compute the deficit.\n"
                "3. If deficit is zero, choose WAIT.\n"
                "4. If transfer cost is lower than stockout cost and origin stock exists, choose TRANSFER.\n"
                "5. Return a final JSON with rationale."
            ),
            methodology_es=(
                "1. Leer el JSON de pronóstico, inventario y costos.\n"
                "2. Calcular el déficit.\n"
                "3. Si el déficit es cero, elegir ESPERAR.\n"
                "4. Si el costo de transferir es menor que el costo de quiebre y existe stock de origen, elegir TRANSFERIR.\n"
                "5. Regresar un JSON final con razonamiento."
            ),
            limits_en=(
                "Never invent data, never recommend a transfer without a valid origin CEDI, and always include the disclaimer."
            ),
            limits_es=(
                "Nunca inventar datos, nunca recomendar una transferencia sin un CEDI origen válido y siempre incluir el disclaimer."
            ),
            examples_en='Example output: {"decision":"TRANSFERIR","razonamiento":"Transfer cost MXN 2,528 is far below stockout cost MXN 2,250,000, so transfer is recommended.","costo_asociado":2528.0,"cedi_origen_recomendado":"CEDI_Sur","unidades_a_transferir":200,"disclaimer":"Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."}',
            examples_es='Salida de ejemplo: {"decision":"TRANSFERIR","razonamiento":"El costo de transferir MXN 2,528 es mucho menor que el costo de quiebre MXN 2,250,000, por lo que se recomienda transferir.","costo_asociado":2528.0,"cedi_origen_recomendado":"CEDI_Sur","unidades_a_transferir":200,"disclaimer":"Recomendación basada en análisis económico. No sustituye juicio del equipo de logística."}',
        ),
        verbose=False,
        llm=llm_decision,
        tools=[],
        allow_delegation=False,
    )

    return demand_agent, inventory_agent, cost_agent, decision_agent, llm_decision


# =============================================================================
# Alert input and core pipeline
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
    # Decision: transfer only the deficit. This is a conservative business rule.
    return max(0, math.ceil(float(demand_7d) - float(stock_actual)))


def parse_decision_json(text: str) -> DecisionOutput:
    data = extract_json(text) or {}
    if not data:
        return DecisionOutput(
            decision="ERROR",
            razonamiento=text[:700],
            costo_asociado=0.0,
        )

    normalized = {
        "decision": str(data.get("decision", "ERROR")).upper(),
        "razonamiento": str(data.get("razonamiento", "")),
        "costo_asociado": safe_float(data.get("costo_asociado", 0.0)),
        "cedi_origen_recomendado": data.get("cedi_origen_recomendado"),
        "unidades_a_transferir": data.get("unidades_a_transferir"),
        "disclaimer": data.get("disclaimer")
        or "Recomendación basada en análisis económico. No sustituye juicio del equipo de logística.",
    }
    if normalized["unidades_a_transferir"] is not None:
        normalized["unidades_a_transferir"] = safe_int(normalized["unidades_a_transferir"], 0)
    return DecisionOutput(**normalized)


def build_crew() -> Crew:
    demand_agent, inventory_agent, cost_agent, decision_agent, _ = build_agents()

    t1 = Task(
        description=(
            "Predict 7-day demand for the current alert and return JSON only.\n"
            "Context inputs: sku_id, cedi, fecha, clima.\n"
            "Use the demand_forecast_tool."
        ),
        expected_output='JSON with fields: sku_id, cedi, fecha, demanda_pronosticada_7d, confianza, method',
        agent=demand_agent,
        output_pydantic=DemandForecastOutput,
    )

    t2 = Task(
        description=(
            "Find the best origin CEDI for the same SKU and destination.\n"
            "Use the forecast output as context.\n"
            "Return JSON only."
        ),
        expected_output='JSON with fields: cedi_destino, cedi_origen_sugerido, stock_disponible_origen, costo_transferencia_unidad, costo_transferencia_total',
        agent=inventory_agent,
        output_pydantic=InventoryOutput,
        context=[t1],
    )

    t3 = Task(
        description=(
            "Compute the stockout and transfer costs.\n"
            "Use forecast and inventory context.\n"
            "Return JSON only."
        ),
        expected_output='JSON with fields: costo_quiebre_total, costo_transferencia_total, ahorro_estimado',
        agent=cost_agent,
        output_pydantic=CostOutput,
        context=[t1, t2],
    )

    t4 = Task(
        description=(
            "Make the final decision using the forecast, inventory, and cost outputs.\n"
            "Return JSON only with decision, razonamiento, costo_asociado, cedi_origen_recomendado, unidades_a_transferir, disclaimer."
        ),
        expected_output='JSON with fields: decision, razonamiento, costo_asociado, cedi_origen_recomendado, unidades_a_transferir, disclaimer',
        agent=decision_agent,
        output_pydantic=DecisionOutput,
        context=[t1, t2, t3],
    )

    crew = Crew(
        agents=[demand_agent, inventory_agent, cost_agent, decision_agent],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=False,
    )
    return crew


def run_a2a_pipeline(alerta: AlertInput) -> Tuple[DecisionOutput, Dict[str, Any]]:
    # Decision: keep the analytical steps explicit so the A2A flow is easy to explain in an interview.
    forecast_raw = DemandForecastTool()._run(alerta.sku_id, alerta.cedi, alerta.fecha, alerta.clima)
    forecast = extract_json(forecast_raw) or json.loads(forecast_raw)

    units = compute_units_needed(safe_float(forecast.get("demanda_pronosticada_7d", 0.0)), alerta.stock_actual)
    inventory_raw = InventoryTool()._run(alerta.sku_id, alerta.cedi, units)
    inventory = extract_json(inventory_raw) or json.loads(inventory_raw)

    cost_raw = CostTool()._run(
        stock_actual=alerta.stock_actual,
        demanda=safe_float(forecast.get("demanda_pronosticada_7d", 0.0)),
        costo_quiebre_diario=alerta.costo_quiebre_stock_diario,
        costo_transferencia_unidad=safe_float(inventory.get("costo_transferencia_unidad", alerta.costo_transferencia_unidad)),
        unidades_necesarias=units,
    )
    costs = extract_json(cost_raw) or json.loads(cost_raw)

    # Decision task handled by CrewAI so the final reasoning remains LLM-driven.
    crew = build_crew()
    result = crew.kickoff(
        inputs={
            "sku_id": alerta.sku_id,
            "cedi": alerta.cedi,
            "fecha": alerta.fecha,
            "clima": alerta.clima,
            "stock_actual": alerta.stock_actual,
            "forecast_json": forecast,
            "inventory_json": inventory,
            "cost_json": costs,
            "units_to_move": units,
        }
    )

    output_str = getattr(result, "raw", None) or str(result)
    decision = parse_decision_json(output_str)

    # Hard guardrail: if a transfer is recommended but no valid origin exists, force WAIT.
    if decision.decision == "TRANSFERIR" and inventory.get("cedi_origen_sugerido") in (None, "", "NO_DISPONIBLE", "ERROR"):
        decision.decision = "ESPERAR"
        decision.razonamiento += " | Guardrail override: no valid origin CEDI was available."
        decision.cedi_origen_recomendado = None
        decision.unidades_a_transferir = None

    guardar_decision(
        sku_id=alerta.sku_id,
        cedi=alerta.cedi,
        fecha=alerta.fecha,
        demanda=safe_float(forecast.get("demanda_pronosticada_7d", 0.0)),
        stock=alerta.stock_actual,
        decision=decision.decision,
        costo=decision.costo_asociado,
        razonamiento=decision.razonamiento,
    )

    context = {
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
        "decision": decision.model_dump(),
        "units_to_move": units,
    }
    return decision, context


# =============================================================================
# Chat agent (LangChain wrapper preferred, direct Ollama fallback)
# =============================================================================

class ChatAgent:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.lc_model = make_langchain_chat_model(0.35)

    def answer(self, question: str, last_context: Dict[str, Any], history_snippet: str = "") -> str:
        # Decision: keep the chat prompt bilingual in comments, but use English internally for clarity.
        system = (
            "You are a supply-chain assistant for Grupo Herdez.\n"
            "Answer in Spanish.\n"
            "Use only the provided JSON context.\n"
            "Do not invent numbers.\n"
            "If the data is missing, say it clearly.\n"
            "Always include this disclaimer verbatim at the end:\n"
            "'Recomendación basada en análisis económico. No sustituye juicio del equipo de logística.'"
        )

        user = json.dumps({
            "question": question,
            "last_context": last_context,
            "history_snippet": trim_text(history_snippet, 3000),
        }, ensure_ascii=False)

        try:
            if self.lc_model is not None:
                # Decision: use LangChain prompt composition when the dependency is installed.
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system),
                    ("user", "{user}"),
                ])
                chain = prompt | self.lc_model
                response = chain.invoke({"user": user})
                return getattr(response, "content", str(response))

            # Fallback: direct Ollama HTTP call, still using the same local model.
            return ollama_chat_http(system=system, user=user, temperature=0.35)
        except Exception as e:
            return f"No pude responder por un error del modelo local: {e}"


# =============================================================================
# Streamlit UI
# =============================================================================

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = default_alert()
    if "last_context" not in st.session_state:
        st.session_state.last_context = {}
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
    st.sidebar.caption(f"Model: `{OLLAMA_MODEL}`")
    st.sidebar.caption("Prompts are bilingual (English + Spanish) in the agent backstories.")
    st.sidebar.caption("CrewAI handles the A2A workflow; LangChain helps with the chat wrapper.")

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
    init_session_state()

    if not ollama_is_available():
        st.error(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}. Start Ollama and pull the model: {OLLAMA_MODEL}"
        )
        st.stop()

    st.title("🧠 Grupo Herdez - A2A Supply Chain Decision Assistant")
    st.caption(
        "Hybrid A2A architecture: deterministic forecast/inventory/cost steps + one local Ollama agent for the final decision and chat."
    )

    with st.sidebar:
        current_alert = ui_sidebar()
        run_btn = st.button("Run decision", use_container_width=True)
        sample_btn = st.button("Load sample alert", use_container_width=True)
        history_btn = st.button("Show history", use_container_width=True)
        clear_btn = st.button("Clear chat", use_container_width=True)

    if sample_btn:
        st.session_state.last_alert = default_alert()
        st.rerun()

    if clear_btn:
        st.session_state.messages = []
        st.session_state.last_context = {}
        st.rerun()

    chat_agent = ChatAgent()

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.subheader("Chat")
        render_chat_history()

        chat_text = st.chat_input("Ask about the last alert, the decision, costs, or history...")
        if chat_text:
            add_message("user", chat_text)
            history = "\n".join([f"{m['role']}: {m['content']}" for m in get_chat_history(st.session_state.session_id, MAX_CHAT_HISTORY)])
            reply = chat_agent.answer(chat_text, st.session_state.last_context, history)
            add_message("assistant", reply)
            st.rerun()

        if run_btn:
            st.session_state.last_alert = current_alert
            with st.spinner("Running A2A pipeline..."):
                decision, context = run_a2a_pipeline(current_alert)
                st.session_state.last_context = context

                # Decision: expose the structured outcome directly in the chat so the user can ask follow-up questions.
                add_message(
                    "assistant",
                    (
                        f"**Decision:** {decision.decision}\n\n"
                        f"{decision.razonamiento}\n\n"
                        f"**Associated cost:** MXN {decision.costo_asociado:,.2f}"
                    )
                )
            st.rerun()

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
                    dfh[[
                        "timestamp", "sku_id", "cedi", "decision",
                        "demanda_pronosticada", "stock_actual", "costo_real"
                    ]],
                    use_container_width=True,
                )

    st.divider()
    st.caption(
        "Recommended local model for your hardware: qwen2.5:0.5b-instruct. "
        "If you need a bit more quality and still remain lightweight, try qwen2.5:1.5b-instruct."
    )


# =============================================================================
# CLI fallback
# =============================================================================

def run_cli() -> None:
    init_long_term_memory()
    if not ollama_is_available():
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}. Start Ollama and pull the model: {OLLAMA_MODEL}"
        )

    print(f"\n{APP_TITLE}")
    print(f"Model: {OLLAMA_MODEL}")
    print("Commands: alerta | historial | salir")
    print("You can also ask questions after generating an alert.\n")

    last_context: Dict[str, Any] = {}
    session_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    chat_agent = ChatAgent()

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
            decision, last_context = run_a2a_pipeline(alerta)
            print("\nDecision:", decision.decision)
            print("Reason:", decision.razonamiento)
            print(f"Cost: MXN {decision.costo_asociado:,.2f}")
            if decision.cedi_origen_recomendado:
                print("Origin CEDI:", decision.cedi_origen_recomendado)
                print("Units:", decision.unidades_a_transferir)
            continue

        if low == "historial":
            hist = recuperar_historico(limit=10)
            if not hist:
                print("No records yet.")
            else:
                for i, row in enumerate(hist, 1):
                    print(f"{i}. {row['timestamp']} | {row['sku_id']} | {row['cedi']} | {row['decision']} | MXN {row['costo_real']:,.2f}")
            continue

        history = "\n".join([f"{m['role']}: {m['content']}" for m in get_chat_history(session_id, MAX_CHAT_HISTORY)])
        answer = chat_agent.answer(user_input, last_context, history_snippet=history)
        print("\nAssistant:", answer)
        save_chat_message(session_id, "user", user_input)
        save_chat_message(session_id, "assistant", answer)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        run_cli()


if __name__ == "__main__":
    main()
