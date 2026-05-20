# supply_chain_hf.py
# Aplicación para cadena de suministro con Hugging Face Inference API
# Compatible con Streamlit Cloud

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
import streamlit as st

from pydantic import BaseModel
from xgboost import XGBRegressor

# LangChain + Hugging Face
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# =============================================================================
# CONFIG
# =============================================================================

APP_TITLE = "Supply Chain A2A Assistant (HF)"

DB_PATH = "long_term_memory.db"
DUCKDB_PATH = "data/herdez.duckdb"

# CAMBIO IMPORTANTE:
# usar .json en lugar de .pkl para XGBoost
XGB_MODEL_PATH = "xgb_model.json"

ENCODERS_PATH = "encoders.pkl"

EXCEL_PATH = "data/Data_Prueba_Tecnica_Herdez_IA.xlsx"

HF_REPO_ID = "microsoft/Phi-3.5-mini-instruct"
HF_TEMPERATURE = 0.1
HF_MAX_TOKENS = 200

# =============================================================================
# HUGGING FACE TOKEN
# =============================================================================

if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets[
        "HUGGINGFACEHUB_API_TOKEN"
    ]
else:
    st.error("❌ Falta HUGGINGFACEHUB_API_TOKEN en secrets.")
    st.stop()

# =============================================================================
# PYDANTIC
# =============================================================================


class DecisionOutput(BaseModel):
    decision: str
    razonamiento: str
    costo_asociado: float
    cedi_origen_recomendado: Optional[str] = None
    unidades_a_transferir: Optional[int] = None
    disclaimer: str


# =============================================================================
# UTILS
# =============================================================================


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def extract_json(text: str):

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

    candidate = candidate.replace("```json", "")
    candidate = candidate.replace("```", "")

    try:
        return json.loads(candidate)
    except Exception:
        return None


# =============================================================================
# SQLITE MEMORY
# =============================================================================


def init_long_term_memory():

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
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
    """
    )

    conn.commit()
    conn.close()


def guardar_decision(
    sku_id,
    cedi,
    fecha,
    demanda,
    stock,
    decision,
    costo,
    razonamiento,
):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO decisiones (
            sku_id,
            cedi,
            fecha,
            demanda_pronosticada,
            stock_actual,
            decision,
            costo_real,
            razonamiento,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            sku_id,
            cedi,
            fecha,
            demanda,
            stock,
            decision,
            costo,
            razonamiento,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()


def recuperar_historico(limit=10):

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM decisiones ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=[limit],
    )

    conn.close()

    return df.to_dict(orient="records")


# =============================================================================
# MODELOS
# =============================================================================


class ModelBundle:

    def __init__(self):

        self._model = None
        self._encoders = None

        # ==========================================
        # CARGAR MODELO XGBOOST JSON
        # ==========================================

        if os.path.exists(XGB_MODEL_PATH):

            try:

                model = XGBRegressor()

                model.load_model(XGB_MODEL_PATH)

                self._model = model

                print("✅ Modelo XGBoost cargado")

            except Exception as e:

                print(f"⚠️ Error cargando modelo: {e}")

        else:
            print("⚠️ No existe xgb_model.json")

        # ==========================================
        # CARGAR ENCODERS
        # ==========================================

        if os.path.exists(ENCODERS_PATH):

            try:

                self._encoders = joblib.load(ENCODERS_PATH)

                print("✅ Encoders cargados")

            except Exception as e:

                print(f"⚠️ Error cargando encoders: {e}")

        else:
            print("⚠️ No existe encoders.pkl")

    @property
    def model(self):
        return self._model

    @property
    def encoders(self):
        return self._encoders


MODELS = ModelBundle()

# =============================================================================
# DUCKDB
# =============================================================================


def get_connection():
    return duckdb.connect(DUCKDB_PATH)


def ensure_inventory_table():

    if not os.path.exists(DUCKDB_PATH):

        if os.path.exists(EXCEL_PATH):

            conn = duckdb.connect(DUCKDB_PATH)

            df = pd.read_excel(EXCEL_PATH)

            conn.execute(
                "CREATE TABLE inventario_raw AS SELECT * FROM df"
            )

            conn.close()

            print("✅ DuckDB inicializado")


# =============================================================================
# DEMAND FORECAST
# =============================================================================


def demand_forecast_tool(
    sku_id,
    cedi,
    fecha,
    clima,
):

    try:

        conn = get_connection()

        query = """
            SELECT
                Fecha,
                Ventas_Unidades,
                Stock_Actual,
                Lead_Time_Dias,
                Promocion_Activa,
                Precio_Combustible_MXN,
                Clima,
                Costo_Quiebre_Stock_Diario,
                Costo_Transferencia_Unidad
            FROM inventario_raw
            WHERE SKU_ID = ?
            AND CEDI = ?
            AND Fecha <= ?
            ORDER BY Fecha DESC
            LIMIT 8
        """

        df_hist = conn.execute(
            query,
            [sku_id, cedi, fecha],
        ).df()

        conn.close()

        if df_hist.empty:

            return {
                "demanda_pronosticada_7d": 700.0,
                "confianza": 0.10,
                "metodo": "fallback_empty",
            }

        df_hist = (
            df_hist.sort_values("Fecha")
            .reset_index(drop=True)
        )

        # ==========================================
        # FALLBACK SIMPLE
        # ==========================================

        if MODELS.model is None or MODELS.encoders is None:

            demand = (
                df_hist["Ventas_Unidades"]
                .tail(7)
                .mean()
                * 7
            )

            return {
                "demanda_pronosticada_7d": round(float(demand), 2),
                "confianza": 0.65,
                "metodo": "moving_average",
            }

        # ==========================================
        # FEATURES
        # ==========================================

        df_hist["ventas_lag_1"] = (
            df_hist["Ventas_Unidades"].shift(1)
        )

        df_hist["ventas_lag_7"] = (
            df_hist["Ventas_Unidades"].shift(7)
        )

        df_hist["ventas_media_7d"] = (
            df_hist["Ventas_Unidades"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        df_hist["ventas_std_7d"] = (
            df_hist["Ventas_Unidades"]
            .shift(1)
            .rolling(7)
            .std()
        )

        df_hist["dias_cobertura"] = (
            df_hist["Stock_Actual"]
            / (df_hist["Ventas_Unidades"] + 1)
        )

        sku_enc = MODELS.encoders["sku"].transform([sku_id])[0]

        cedi_enc = MODELS.encoders["cedi"].transform([cedi])[0]

        clima_enc = MODELS.encoders["clima"].transform([clima])[0]

        df_hist["SKU_encoded"] = sku_enc
        df_hist["CEDI_encoded"] = cedi_enc
        df_hist["Clima_encoded"] = clima_enc

        features = [
            "Ventas_Unidades",
            "Stock_Actual",
            "Lead_Time_Dias",
            "Promocion_Activa",
            "Precio_Combustible_MXN",
            "Costo_Quiebre_Stock_Diario",
            "Costo_Transferencia_Unidad",
            "ventas_lag_1",
            "ventas_lag_7",
            "ventas_media_7d",
            "ventas_std_7d",
            "dias_cobertura",
            "SKU_encoded",
            "CEDI_encoded",
            "Clima_encoded",
        ]

        X = (
            df_hist.iloc[-1:][features]
            .fillna(0)
        )

        pred = MODELS.model.predict(X)[0]

        return {
            "demanda_pronosticada_7d": round(float(pred), 2),
            "confianza": 0.85,
            "metodo": "xgboost",
        }

    except Exception as e:

        return {
            "demanda_pronosticada_7d": 700.0,
            "confianza": 0.0,
            "metodo": "error",
            "error": str(e),
        }


# =============================================================================
# INVENTORY
# =============================================================================


def inventory_tool(
    sku_id,
    cedi_destino,
    unidades_necesarias,
):

    try:

        conn = get_connection()

        query = """
            SELECT
                CEDI,
                Stock_Actual,
                Costo_Transferencia_Unidad
            FROM inventario_raw
            WHERE SKU_ID = ?
            AND CEDI != ?
            ORDER BY Stock_Actual DESC
        """

        df = conn.execute(
            query,
            [sku_id, cedi_destino],
        ).df()

        conn.close()

        if df.empty:

            return {
                "cedi_origen_sugerido": "NO_DISPONIBLE",
                "stock_disponible_origen": 0,
                "costo_transferencia_unidad": 0,
                "costo_transferencia_total": 0,
            }

        row = df.iloc[0]

        costo_total = (
            row["Costo_Transferencia_Unidad"]
            * unidades_necesarias
        )

        return {
            "cedi_origen_sugerido": row["CEDI"],
            "stock_disponible_origen": int(
                row["Stock_Actual"]
            ),
            "costo_transferencia_unidad": float(
                row["Costo_Transferencia_Unidad"]
            ),
            "costo_transferencia_total": round(
                float(costo_total),
                2,
            ),
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =============================================================================
# COST TOOL
# =============================================================================


def cost_tool(
    stock_actual,
    demanda,
    costo_quiebre_diario,
    costo_transferencia_unidad,
    unidades_necesarias,
):

    deficit = max(0, demanda - stock_actual)

    costo_quiebre = deficit * costo_quiebre_diario

    costo_transferencia = (
        unidades_necesarias
        * costo_transferencia_unidad
    )

    return {
        "costo_quiebre_total": costo_quiebre,
        "costo_transferencia_total": costo_transferencia,
        "ahorro_estimado": (
            costo_quiebre
            - costo_transferencia
        ),
    }


# =============================================================================
# DECISION AGENT
# =============================================================================


class DecisionAgent:

    def __init__(self):

        self.llm = HuggingFaceEndpoint(
            repo_id=HF_REPO_ID,
            temperature=HF_TEMPERATURE,
            max_new_tokens=HF_MAX_TOKENS,
        )

        self.parser = JsonOutputParser(
            pydantic_object=DecisionOutput
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Eres una directora logística experta.

Debes decidir:
- TRANSFERIR
- ESPERAR

Reglas:
- Si no hay déficit -> ESPERAR
- Si transferir es más barato que quiebre -> TRANSFERIR

Devuelve SOLO JSON válido.
""",
                ),
                ("user", "{input}"),
            ]
        )

        self.chain = (
            self.prompt
            | self.llm
            | self.parser
        )

    def decide(self, alerta_data):

        try:

            result = self.chain.invoke(
                {
                    "input": json.dumps(
                        alerta_data,
                        ensure_ascii=False,
                    )
                }
            )

            return DecisionOutput(**result)

        except Exception:

            if (
                alerta_data["costo_transferencia_total"]
                < alerta_data["costo_quiebre_total"]
                and alerta_data["deficit"] > 0
            ):

                return DecisionOutput(
                    decision="TRANSFERIR",
                    razonamiento="La transferencia es más barata.",
                    costo_asociado=alerta_data[
                        "costo_transferencia_total"
                    ],
                    cedi_origen_recomendado=alerta_data[
                        "best_origin_cedi"
                    ],
                    unidades_a_transferir=alerta_data[
                        "units_needed"
                    ],
                    disclaimer="Recomendación automática.",
                )

            return DecisionOutput(
                decision="ESPERAR",
                razonamiento="No conviene transferir.",
                costo_asociado=0,
                disclaimer="Recomendación automática.",
            )


# =============================================================================
# ALERT INPUT
# =============================================================================


@dataclass
class AlertInput:
    sku_id: str
    cedi: str
    fecha: str
    stock_actual: float
    costo_quiebre_stock_diario: float
    costo_transferencia_unidad: float
    clima: str


# =============================================================================
# HELPERS
# =============================================================================


def compute_units_needed(
    demand_7d,
    stock_actual,
):

    return max(
        0,
        math.ceil(demand_7d - stock_actual),
    )


# =============================================================================
# MAIN PROCESS
# =============================================================================


def procesar_alerta(alerta: AlertInput):

    forecast = demand_forecast_tool(
        alerta.sku_id,
        alerta.cedi,
        alerta.fecha,
        alerta.clima,
    )

    units = compute_units_needed(
        forecast["demanda_pronosticada_7d"],
        alerta.stock_actual,
    )

    inventory = inventory_tool(
        alerta.sku_id,
        alerta.cedi,
        units,
    )

    costs = cost_tool(
        alerta.stock_actual,
        forecast["demanda_pronosticada_7d"],
        alerta.costo_quiebre_stock_diario,
        inventory.get(
            "costo_transferencia_unidad",
            alerta.costo_transferencia_unidad,
        ),
        units,
    )

    alert_data = {
        "sku": alerta.sku_id,
        "cedi": alerta.cedi,
        "stock_actual": alerta.stock_actual,
        "demanda_7d": forecast[
            "demanda_pronosticada_7d"
        ],
        "costo_transferencia_total": costs[
            "costo_transferencia_total"
        ],
        "costo_quiebre_total": costs[
            "costo_quiebre_total"
        ],
        "deficit": max(
            0,
            forecast["demanda_pronosticada_7d"]
            - alerta.stock_actual,
        ),
        "units_needed": units,
        "best_origin_cedi": inventory[
            "cedi_origen_sugerido"
        ],
    }

    decision = DecisionAgent().decide(
        alert_data
    )

    guardar_decision(
        alerta.sku_id,
        alerta.cedi,
        alerta.fecha,
        forecast["demanda_pronosticada_7d"],
        alerta.stock_actual,
        decision.decision,
        decision.costo_asociado,
        decision.razonamiento,
    )

    return decision, {
        "forecast": forecast,
        "inventory": inventory,
        "costs": costs,
    }


# =============================================================================
# STREAMLIT
# =============================================================================


def run_streamlit():

    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    st.title("📦 Supply Chain AI")

    with st.sidebar:

        st.header("Configuración")

        sku = st.text_input(
            "SKU",
            "HZ-Salsa-Verde-200g",
        )

        cedi = st.text_input(
            "CEDI",
            "CEDI_Norte",
        )

        fecha = st.date_input(
            "Fecha"
        )

        stock = st.number_input(
            "Stock actual",
            value=50.0,
        )

        costo_q = st.number_input(
            "Costo quiebre",
            value=15000.0,
        )

        costo_t = st.number_input(
            "Costo transferencia",
            value=12.64,
        )

        clima = st.text_input(
            "Clima",
            "Despejado",
        )

        ejecutar = st.button(
            "🚀 Ejecutar"
        )

    if ejecutar:

        alerta = AlertInput(
            sku_id=sku,
            cedi=cedi,
            fecha=fecha.strftime("%Y-%m-%d"),
            stock_actual=stock,
            costo_quiebre_stock_diario=costo_q,
            costo_transferencia_unidad=costo_t,
            clima=clima,
        )

        with st.spinner("Analizando..."):

            decision, context = procesar_alerta(
                alerta
            )

        st.success(
            f"Decisión: {decision.decision}"
        )

        st.markdown(
            f"""
### Razonamiento
{decision.razonamiento}

### Costo asociado
MXN {decision.costo_asociado:,.2f}
"""
        )

        st.json(context)

        # =========================
        # GRÁFICA
        # =========================

        fig, ax = plt.subplots()

        valores = [
            context["costs"][
                "costo_transferencia_total"
            ],
            context["costs"][
                "costo_quiebre_total"
            ],
        ]

        etiquetas = [
            "Transferencia",
            "Quiebre",
        ]

        ax.bar(etiquetas, valores)

        ax.set_ylabel("MXN")

        st.pyplot(fig)

    st.subheader("📜 Historial")

    hist = recuperar_historico()

    if hist:

        st.dataframe(
            pd.DataFrame(hist)
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    init_long_term_memory()

    ensure_inventory_table()

    run_streamlit()
