# supply_chain_final.py - Usando Ollama con modelo pequeño (llama3.2:1b)
import os
import json
import sqlite3
import re
import pandas as pd
import duckdb
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# ------------------------------------------------------------
# 0. CONFIGURACIÓN DE OLLAMA CON MODELO PEQUEÑO (soporta tools)
# ------------------------------------------------------------
# Asegúrate de haber descargado el modelo: ollama pull llama3.2:1b
# Si quieres usar phi3:mini, cambia la línea siguiente
MODELO_OLLAMA = "llama3.2:1b"   # o "phi3:mini"

llm_demand = LLM(model=f"ollama/{MODELO_OLLAMA}", temperature=0.2)
llm_inventory = LLM(model=f"ollama/{MODELO_OLLAMA}", temperature=0.2)
llm_cost = LLM(model=f"ollama/{MODELO_OLLAMA}", temperature=0.2)
llm_decision = LLM(model=f"ollama/{MODELO_OLLAMA}", temperature=0.1)

# ------------------------------------------------------------
# 1. ESQUEMAS DE SALIDA (Pydantic)
# ------------------------------------------------------------
class DemandForecastOutput(BaseModel):
    sku_id: str
    cedi: str
    fecha: str
    demanda_pronosticada_7d: float
    confianza: float = 0.9

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

# ------------------------------------------------------------
# 2. MEMORIA A LARGO PLAZO (SQLite)
# ------------------------------------------------------------
def init_long_term_memory():
    conn = sqlite3.connect("long_term_memory.db")
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
            timestamp TEXT
        )
    """)
    conn.close()

def guardar_decision(sku_id, cedi, fecha, demanda, stock, decision, costo):
    conn = sqlite3.connect("long_term_memory.db")
    conn.execute("""
        INSERT INTO decisiones (sku_id, cedi, fecha, demanda_pronosticada, stock_actual, decision, costo_real, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (sku_id, cedi, fecha, demanda, stock, decision, costo, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def recuperar_historico(sku_id=None, cedi=None, limit=5):
    conn = sqlite3.connect("long_term_memory.db")
    query = "SELECT * FROM decisiones"
    params = []
    if sku_id and cedi:
        query += " WHERE sku_id=? AND cedi=?"
        params = [sku_id, cedi]
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df.to_dict(orient="records")

# ------------------------------------------------------------
# 3. HERRAMIENTAS (usando modelo XGBoost real)
# ------------------------------------------------------------
class DemandForecastTool(BaseTool):
    name: str = "demand_forecast_tool"
    description: str = "Predice demanda total a 7 días usando modelo XGBoost local. Entrada: sku_id, cedi, fecha, clima. Retorna un número (string)."
    def _run(self, sku_id: str, cedi: str, fecha: str, clima: str) -> str:
        try:
            global model, encoders
            try:
                model
            except NameError:
                model = joblib.load("modelo_xgboost_local.pkl")
                encoders = joblib.load("encoders.pkl")
            conn = duckdb.connect("data/herdez.duckdb")
            query = f"""
                SELECT Fecha, Ventas_Unidades, Stock_Actual, Lead_Time_Dias, Promocion_Activa,
                       Precio_Combustible_MXN, Clima, Costo_Quiebre_Stock_Diario, Costo_Transferencia_Unidad
                FROM inventario_raw
                WHERE SKU_ID='{sku_id}' AND CEDI='{cedi}' AND Fecha <= '{fecha}'
                ORDER BY Fecha DESC LIMIT 8
            """
            df_hist = conn.execute(query).df()
            conn.close()
            if len(df_hist) < 7:
                return "0.0"
            df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)
            # Feature engineering
            df_hist["ventas_lag_1"] = df_hist["Ventas_Unidades"].shift(1)
            df_hist["ventas_lag_7"] = df_hist["Ventas_Unidades"].shift(7)
            df_hist["ventas_media_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).mean()
            df_hist["ventas_std_7d"] = df_hist["Ventas_Unidades"].shift(1).rolling(7).std()
            df_hist["dias_cobertura"] = df_hist["Stock_Actual"] / (df_hist["Ventas_Unidades"] + 1)
            df_hist["promo_x_ventas"] = df_hist["Promocion_Activa"] * df_hist["Ventas_Unidades"]
            df_hist["leadtime_x_ventas"] = df_hist["Lead_Time_Dias"] * df_hist["Ventas_Unidades"]
            df_hist["SKU_encoded"] = encoders["sku"].transform([sku_id])[0]
            df_hist["CEDI_encoded"] = encoders["cedi"].transform([cedi])[0]
            df_hist["Clima_encoded"] = encoders["clima"].transform([clima])[0]
            feature_cols = ["Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
                            "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
                            "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
                            "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
                            "SKU_encoded", "CEDI_encoded", "Clima_encoded"]
            X = df_hist.iloc[-1:][feature_cols].fillna(0)
            pred = model.predict(X)[0]
            return f"{pred:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

class InventoryTool(BaseTool):
    name: str = "inventory_tool"
    description: str = "Dado un SKU, un CEDI destino y unidades necesarias, retorna el mejor CEDI origen (formato: 'CEDI_XXX|stock|costo_unitario|costo_total')."
    def _run(self, sku_id: str, cedi_destino: str, unidades_necesarias: int) -> str:
        try:
            conn = duckdb.connect("data/herdez.duckdb")
            # Aseguramos que los CEDIs existan en la base (ej: CEDI_Norte, CEDI_Sur, CEDI_Occidente, CEDI_Bajio)
            query = f"""
                SELECT DISTINCT CEDI, Stock_Actual, Costo_Transferencia_Unidad
                FROM inventario_raw
                WHERE SKU_ID='{sku_id}' AND CEDI != '{cedi_destino}'
                ORDER BY Stock_Actual DESC, Costo_Transferencia_Unidad ASC
                LIMIT 1
            """
            df = conn.execute(query).df()
            conn.close()
            if df.empty:
                return "NO_DISPONIBLE"
            row = df.iloc[0]
            costo_total = row['Costo_Transferencia_Unidad'] * unidades_necesarias
            # Retornamos en formato pipe para que el LLM lo entienda
            return f"{row['CEDI']}|{int(row['Stock_Actual'])}|{row['Costo_Transferencia_Unidad']}|{costo_total:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

class CostTool(BaseTool):
    name: str = "cost_tool"
    description: str = "Calcula costos de quiebre y transferencia. Entrada: stock_actual, demanda, costo_quiebre_diario, costo_transferencia_unidad, unidades_necesarias. Retorna 'costo_quiebre|costo_transferencia|ahorro'."
    def _run(self, stock_actual: float, demanda: float, costo_quiebre_diario: float,
             costo_transferencia_unidad: float, unidades_necesarias: int) -> str:
        dias_quiebre = max(0, demanda - stock_actual)
        costo_quiebre = dias_quiebre * costo_quiebre_diario
        costo_transferencia = unidades_necesarias * costo_transferencia_unidad
        ahorro = costo_quiebre - costo_transferencia
        return f"{costo_quiebre:.2f}|{costo_transferencia:.2f}|{ahorro:.2f}"

# ------------------------------------------------------------
# 4. AGENTES (con instrucciones mejoradas)
# ------------------------------------------------------------
demand_agent = Agent(
    role="Demand Forecaster",
    goal="Pronosticar la demanda de los próximos 7 días usando XGBoost.",
    backstory="""Eres Ana López, científica de datos con 8 años de experiencia en pronósticos.
Debes usar la herramienta demand_forecast_tool para obtener la predicción.
Devuelve un JSON con los campos: sku_id, cedi, fecha, demanda_pronosticada_7d, confianza.
Ejemplo: {"demanda_pronosticada_7d": 1450.50, "confianza": 0.95}""",
    verbose=False,
    llm=llm_demand,
    tools=[DemandForecastTool()],
    allow_delegation=False
)

inventory_agent = Agent(
    role="Inventory Analyst",
    goal="Identificar el mejor CEDI origen para transferir.",
    backstory="""Eres Carlos Méndez, analista de inventarios.
Usa inventory_tool con los parámetros: sku_id, cedi_destino, unidades_necesarias.
La herramienta devuelve un string con formato "CEDI_XXX|stock|costo_unitario|costo_total".
Debes parsear ese resultado y devolver un JSON como:
{
  "cedi_destino": "CEDI_Norte",
  "cedi_origen_sugerido": "CEDI_Sur",
  "stock_disponible_origen": 800,
  "costo_transferencia_unidad": 12.64,
  "costo_transferencia_total": 2528.0
}
Si la herramienta devuelve "NO_DISPONIBLE", indica que no hay origen disponible.""",
    verbose=False,
    llm=llm_inventory,
    tools=[InventoryTool()],
    allow_delegation=False
)

cost_agent = Agent(
    role="Cost Analyst",
    goal="Calcular costos de quiebre y transferencia.",
    backstory="""Eres Laura Fernández, analista financiera.
Usa la herramienta cost_tool con los parámetros: stock_actual, demanda, costo_quiebre_diario, costo_transferencia_unidad, unidades_necesarias.
La herramienta devuelve "costo_quiebre|costo_transferencia|ahorro".
Debes devolver un JSON con los campos: costo_quiebre_total, costo_transferencia_total, ahorro_estimado.
Ejemplo: {"costo_quiebre_total": 2250000.00, "costo_transferencia_total": 2528.00, "ahorro_estimado": 2247472.00}""",
    verbose=False,
    llm=llm_cost,
    tools=[CostTool()],
    allow_delegation=False
)

decision_agent = Agent(
    role="Supply Chain Decision Maker",
    goal="Decidir entre transferir inventario o esperar reabastecimiento, con razonamiento paso a paso.",
    backstory="""Eres Claudia Mendoza, Directora de Logística con 15 años de experiencia.
Recibirás los resultados de los otros agentes (pronóstico, inventario, costos).
Debes seguir esta metodología:
1. Calcula déficit = max(0, demanda_pronosticada - stock_actual)
2. Si déficit == 0 → decisión ESPERAR.
3. Si déficit > 0:
   - Compara costo_transferencia_total vs costo_quiebre_total
   - Si costo_transferencia_total < costo_quiebre_total y hay un CEDI origen disponible → TRANSFERIR
   - En caso contrario → ESPERAR
4. Explica claramente los números y el ahorro (o pérdida) en el razonamiento.
5. Si decides transferir, indica el CEDI origen y las unidades a transferir.
6. Incluye siempre el disclaimer.

Devuelve un JSON con: decision, razonamiento, costo_asociado (el costo de la acción elegida), cedi_origen_recomendado (si aplica), unidades_a_transferir (si aplica), disclaimer.
Ejemplo: {"decision": "TRANSFERIR", "razonamiento": "El costo de transferir 200 unidades desde CEDI_Sur es $2,528, mientras que el quiebre costaría $2,250,000. Ahorro de $2,247,472.", "costo_asociado": 2528.0, "cedi_origen_recomendado": "CEDI_Sur", "unidades_a_transferir": 200, "disclaimer": "..."}""",
    verbose=True,
    llm=llm_decision,
    tools=[],
    allow_delegation=False
)

# ------------------------------------------------------------
# 5. ORQUESTADOR
# ------------------------------------------------------------
def procesar_alerta(alerta: Dict[str, Any]) -> DecisionOutput:
    # Si no viene demanda_predicha, la calculamos con la herramienta real
    if "demanda_predicha" not in alerta:
        try:
            demanda_str = DemandForecastTool()._run(
                alerta["SKU_ID"], alerta["CEDI"], alerta["Fecha"], alerta.get("Clima", "Despejado")
            )
            alerta["demanda_predicha"] = float(demanda_str)
        except Exception as e:
            alerta["demanda_predicha"] = 0.0

    unidades_necesarias = max(0, alerta["demanda_predicha"] - alerta["Stock_Actual"])

    t1 = Task(
        description=f"Predice demanda para SKU={alerta['SKU_ID']}, CEDI={alerta['CEDI']}, Fecha={alerta['Fecha']}, Clima={alerta.get('Clima','Despejado')}",
        expected_output="JSON con demanda_pronosticada_7d y confianza",
        agent=demand_agent,
        output_pydantic=DemandForecastOutput
    )
    t2 = Task(
        description=f"Encuentra mejor CEDI origen para SKU={alerta['SKU_ID']}, destino={alerta['CEDI']}, unidades_necesarias={unidades_necesarias}",
        expected_output="JSON con cedi_origen_sugerido, stock_disponible_origen, costo_transferencia_unidad, costo_transferencia_total",
        agent=inventory_agent,
        output_pydantic=InventoryOutput
    )
    t3 = Task(
        description=f"Calcula costos: stock_actual={alerta['Stock_Actual']}, demanda={alerta['demanda_predicha']}, costo_quiebre_diario={alerta['Costo_Quiebre_Stock_Diario']}, costo_transferencia_unidad={alerta['Costo_Transferencia_Unidad']}, unidades_necesarias={unidades_necesarias}",
        expected_output="JSON con costo_quiebre_total, costo_transferencia_total, ahorro_estimado",
        agent=cost_agent,
        output_pydantic=CostOutput
    )
    t4 = Task(
        description="Con los resultados de las tareas anteriores, decide TRANSFERIR o ESPERAR. Devuelve JSON con decision, razonamiento, costo_asociado, cedi_origen_recomendado (si aplica), unidades_a_transferir (si aplica), disclaimer.",
        expected_output="JSON con decision, razonamiento, costo_asociado, etc.",
        agent=decision_agent,
        output_pydantic=DecisionOutput
    )

    crew = Crew(
        agents=[demand_agent, inventory_agent, cost_agent, decision_agent],
        tasks=[t1, t2, t3, t4],
        verbose=True,
        process=Process.sequential
    )
    result = crew.kickoff()

    # Extraer el texto de la respuesta
    try:
        if hasattr(result, 'raw'):
            output_str = result.raw
        elif isinstance(result, str):
            output_str = result
        else:
            output_str = str(result)
        # Buscar JSON en la salida
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            decision_obj = DecisionOutput(**data)
        else:
            decision_obj = DecisionOutput(decision="ERROR", razonamiento=output_str[:500], costo_asociado=0)
    except Exception as e:
        decision_obj = DecisionOutput(decision="ERROR", razonamiento=f"Error al parsear: {str(e)}", costo_asociado=0)

    # Guardar decisión en memoria a largo plazo
    guardar_decision(
        sku_id=alerta["SKU_ID"],
        cedi=alerta["CEDI"],
        fecha=alerta["Fecha"],
        demanda=alerta["demanda_predicha"],
        stock=alerta["Stock_Actual"],
        decision=decision_obj.decision,
        costo=decision_obj.costo_asociado
    )
    return decision_obj

# ------------------------------------------------------------
# 6. EJECUCIÓN
# ------------------------------------------------------------
if __name__ == "__main__":
    init_long_term_memory()
    alerta_ejemplo = {
        "SKU_ID": "HZ-Salsa-Verde-200g",
        "CEDI": "CEDI_Norte",
        "Fecha": "2024-03-15",
        "Stock_Actual": 50,
        "Costo_Quiebre_Stock_Diario": 15000,
        "Costo_Transferencia_Unidad": 12.64,
        "Clima": "Despejado"
    }
    resultado = procesar_alerta(alerta_ejemplo)
    print("\n=== DECISIÓN FINAL ===")
    print(f"Decisión: {resultado.decision}")
    print(f"Razonamiento: {resultado.razonamiento}")
    print(f"Costo asociado: ${resultado.costo_asociado:,.2f}")
    if resultado.cedi_origen_recomendado:
        print(f"Transferir desde: {resultado.cedi_origen_recomendado}")
        print(f"Unidades: {resultado.unidades_a_transferir}")