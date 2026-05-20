# supply_chain_final.py
# Sistema multiagente para cadena de suministro con:
# - Modelo XGBoost local (demanda a 7 días)
# - Agentes con instrucciones de 5 patrones (identidad, misión, metodología, límites, ejemplos)
# - Salidas en JSON para ahorrar tokens
# - Memoria a corto plazo (LangChain) y largo plazo (SQLite)
# - Temperaturas bajas (0.1-0.2) y razonamiento paso a paso (thinking_config de Gemini)
# - Herramientas unificadas (estilo MCP, sin necesidad de servidor externo)

import os
import json
import sqlite3
import re
import pandas as pd
import duckdb
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import content as schema_content

# ------------------------------------------------------------
# 0. CONFIGURACIÓN DE GEMINI CON RAZONAMIENTO
# ------------------------------------------------------------
# Clave de API (obtener desde Google AI Studio)
os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_AQUI"   # Reemplazar

# Configuración de generación con temperaturas bajas y razonamiento
# Para Gemini 1.5 Pro/Flash se puede activar el pensamiento interno
# usando `thinking_config`. Lo hacemos mediante model_kwargs.
# Nota: el parámetro `thinking_budget` define el máximo de tokens para el razonamiento.
generation_config = {
    "temperature": 0.2,        # Valor por defecto, luego se sobreescribe por agente
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    # Configuración de razonamiento (solo para Gemini 1.5 Pro/Flash)
    "thinking_config": {
        "include_thoughts": True,      # Incluye el razonamiento en la salida (visible en verbose)
        "thinking_budget": 2048        # Tokens dedicados a pensar antes de responder
    }
}

def crear_llm(temperature=0.2, thinking=True):
    """Crea una instancia de ChatGoogleGenerativeAI con configuración personalizada."""
    model_kwargs = {}
    if thinking:
        model_kwargs["thinking_config"] = generation_config["thinking_config"]
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # o gemini-1.5-pro si prefieres más capacidad
        temperature=temperature,
        top_p=generation_config["top_p"],
        top_k=generation_config["top_k"],
        max_output_tokens=generation_config["max_output_tokens"],
        model_kwargs=model_kwargs if thinking else None,
        convert_system_message_to_human=True
    )

# LLMs específicos para cada agente (temperatura baja para los factuales)
llm_demand = crear_llm(temperature=0.2, thinking=False)      # Solo forecast, no necesita razonar
llm_inventory = crear_llm(temperature=0.2, thinking=False)
llm_cost = crear_llm(temperature=0.2, thinking=False)
llm_decision = crear_llm(temperature=0.1, thinking=True)     # Máxima factualidad + razonamiento explícito

# ------------------------------------------------------------
# 1. ESQUEMAS DE SALIDA ESTRUCTURADA (Pydantic)
# ------------------------------------------------------------
class DemandForecastOutput(BaseModel):
    sku_id: str = Field(description="SKU del producto")
    cedi: str = Field(description="CEDI destino")
    fecha: str = Field(description="Fecha de la predicción")
    demanda_pronosticada_7d: float = Field(description="Demanda total pronosticada para los próximos 7 días")
    confianza: float = Field(default=0.9, description="Nivel de confianza (0-1) basado en error histórico")

class InventoryOutput(BaseModel):
    cedi_destino: str = Field(description="CEDI que sufre el déficit")
    cedi_origen_sugerido: str = Field(description="Mejor CEDI para transferir")
    stock_disponible_origen: int = Field(description="Unidades disponibles en el CEDI origen")
    costo_transferencia_unidad: float = Field(description="Costo por unidad desde el origen")
    costo_transferencia_total: float = Field(description="Costo total de transferir las unidades necesarias")

class CostOutput(BaseModel):
    costo_quiebre_total: float = Field(description="Pérdida total si no se actúa (días sin stock * costo diario)")
    costo_transferencia_total: float = Field(description="Costo total de transferir el déficit")
    ahorro_estimado: float = Field(description="Ahorro si se transfiere (costo_quiebre - costo_transferencia)")

class DecisionOutput(BaseModel):
    decision: str = Field(description="Acción recomendada: TRANSFERIR o ESPERAR")
    razonamiento: str = Field(description="Explicación paso a paso de la decisión, con números")
    costo_asociado: float = Field(description="Costo de la acción elegida (transferencia o quiebre)")
    cedi_origen_recomendado: Optional[str] = Field(default=None, description="CEDI desde el cual transferir (si aplica)")
    unidades_a_transferir: Optional[int] = Field(default=None, description="Cantidad a transferir (si aplica)")
    disclaimer: str = Field(default="Recomendación basada en análisis económico. No sustituye el juicio del equipo de logística.")

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
# 3. HERRAMIENTAS (BaseTool) – Cada una con propósito claro
# ------------------------------------------------------------
class DemandForecastTool(BaseTool):
    name: str = "demand_forecast_tool"
    description: str = "Predice demanda total a 7 días para un SKU/CEDI/fecha/clima usando modelo XGBoost local."
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
            # Feature engineering (idéntico al entrenamiento)
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
    description: str = "Dado un SKU y un CEDI destino, consulta otros CEDIS para encontrar el mejor origen (mayor stock, menor costo de transferencia)."
    def _run(self, sku_id: str, cedi_destino: str, unidades_necesarias: int) -> str:
        try:
            conn = duckdb.connect("data/herdez.duckdb")
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
            return f"{row['CEDI']}|{row['Stock_Actual']}|{row['Costo_Transferencia_Unidad']}|{costo_total:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

class CostTool(BaseTool):
    name: str = "cost_tool"
    description: str = "Calcula el costo de quiebre (días sin stock * costo diario) y el costo de transferencia."
    def _run(self, stock_actual: float, demanda: float, costo_quiebre_diario: float,
             costo_transferencia_unidad: float, unidades_necesarias: int) -> str:
        dias_quiebre = max(0, demanda - stock_actual)
        costo_quiebre = dias_quiebre * costo_quiebre_diario
        costo_transferencia = unidades_necesarias * costo_transferencia_unidad
        ahorro = costo_quiebre - costo_transferencia
        return f"{costo_quiebre:.2f}|{costo_transferencia:.2f}|{ahorro:.2f}"

# ------------------------------------------------------------
# 4. AGENTES CON INSTRUCCIONES AVANZADAS (5 patrones) Y MEMORIA
# ------------------------------------------------------------
memory_global = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

demand_agent = Agent(
    role="Demand Forecaster",
    goal="Pronosticar la demanda de los próximos 7 días para cada SKU/CEDI con un error porcentual menor al 5%.",
    backstory="""
# IDENTIDAD
Eres **Ana López**, científica de datos con 8 años de experiencia en pronósticos de demanda para cadenas de suministro de alimentos.

# MISIÓN
Tu misión es proporcionar predicciones precisas de demanda a 7 días utilizando el modelo XGBoost entrenado con datos históricos, asegurando que el error absoluto medio (MAE) sea inferior a 50 unidades.

# METODOLOGÍA
1. **Recibir solicitud**: Obtienes SKU_ID, CEDI, fecha actual y clima.
2. **Consultar histórico**: Te conectas a DuckDB para extraer las últimas 8 filas.
3. **Calcular features**: Construyes lags (1 y 7 días), medias móviles, desviación estándar, días de cobertura, interacciones y codificas categóricas.
4. **Ejecutar modelo**: Aplicas el modelo XGBoost y obtienes la predicción.
5. **Retornar resultado**: Devuelves la demanda pronosticada y un nivel de confianza.

# LÍMITES
- **Nunca** inventes datos; si hay menos de 7 registros, retorna 0.0.
- **Siempre** redondea a 2 decimales.
- **Siempre** incluye el disclaimer: "Predicción basada en modelo XGBoost con MAPE ~2.9%."

# EJEMPLOS
**Caso típico:**
Entrada: SKU=HZ-Salsa-Verde-200g, CEDI=CEDI_Norte, fecha=2024-03-15, clima=Despejado
Salida: {"demanda_pronosticada_7d": 1420.50, "confianza": 0.92, "disclaimer": "..."}
""",
    verbose=False,
    llm=llm_demand,
    tools=[DemandForecastTool()],
    memory=memory_global
)

inventory_agent = Agent(
    role="Inventory Analyst",
    goal="Identificar el CEDI con mayor stock disponible y menor costo de transferencia para cada alerta.",
    backstory="""
# IDENTIDAD
Eres **Carlos Méndez**, analista de inventarios con 10 años de experiencia en logística de retail y distribución.

# MISIÓN
Tu misión es consultar la base de datos de inventarios (DuckDB) y, dado un SKU y un CEDI destino con déficit, recomendar el mejor CEDI origen.

# METODOLOGÍA
1. **Recibir parámetros**: SKU, CEDI destino, unidades necesarias.
2. **Consultar otros CEDIs**: Ejecutas SQL que devuelve todos los demás CEDIs con stock y costo de transferencia.
3. **Ordenar y seleccionar**: Priorizas mayor stock, luego menor costo.
4. **Calcular costo total**: unidades_necesarias * costo_unitario.
5. **Retornar resultado**: CEDI origen, stock, costo unitario y costo total.

# LÍMITES
- **Nunca** selecciones el mismo CEDI destino.
- **Nunca** inventes un CEDI; si no hay otros, retorna "NO_DISPONIBLE".
- **Siempre** verifica que el stock disponible sea suficiente.

# EJEMPLOS
Entrada: SKU=HZ-Salsa-Verde-200g, destino=CEDI_Norte, unidades=200
Salida: {"cedi_origen_sugerido": "CEDI_Sur", "stock_disponible_origen": 500, "costo_transferencia_unidad": 12.64, "costo_transferencia_total": 2528.0}
""",
    verbose=False,
    llm=llm_inventory,
    tools=[InventoryTool()],
    memory=memory_global
)

cost_agent = Agent(
    role="Cost Analyst",
    goal="Calcular el costo total de un quiebre de stock y el costo de transferencia, así como el ahorro potencial.",
    backstory="""
# IDENTIDAD
Eres **Laura Fernández**, analista financiera especializada en cadena de suministro.

# MISIÓN
Tu misión es cuantificar en términos monetarios el impacto de no actuar (quiebre) versus la acción de transferir.

# METODOLOGÍA
1. **Calcular días de desabasto**: max(0, demanda - stock).
2. **Costo de quiebre**: días * costo_quiebre_diario.
3. **Costo de transferencia**: unidades_necesarias * costo_transferencia_unidad.
4. **Ahorro**: costo_quiebre - costo_transferencia.

# LÍMITES
- **Nunca** permitas valores negativos en días.
- **Siempre** retorna tres números separados por pipe para facilitar el parsing.

# EJEMPLOS
Entrada: stock=50, demanda=200, costo_quiebre_diario=15000, costo_transferencia_unidad=12.64, unidades=150
Salida: {"costo_quiebre_total": 2250000.00, "costo_transferencia_total": 1896.00, "ahorro_estimado": 2248104.00}
""",
    verbose=False,
    llm=llm_cost,
    tools=[CostTool()],
    memory=memory_global
)

decision_agent = Agent(
    role="Supply Chain Decision Maker",
    goal="Decidir si transferir inventario desde otro CEDI o esperar el reabastecimiento, basándose en el análisis de costos y disponibilidad.",
    backstory="""
# IDENTIDAD
Eres **Claudia Mendoza**, Directora de Logística y Cadena de Suministro de Grupo Herdez con 15 años de experiencia.

# MISIÓN
Tu misión es tomar la mejor decisión operativa y económica para cada alerta de desabasto, garantizando la continuidad del negocio y minimizando costos logísticos.

# METODOLOGÍA (razonamiento paso a paso)
1. **Recibir la alerta** y los resultados de los subagentes.
2. **Identificar el déficit**: `unidades_faltantes = max(0, demanda_pronosticada - stock_actual)`.
3. **Si el déficit es 0**: decisión ESPERAR.
4. **Si hay déficit**:
   - Obtén el costo de transferencia desde el mejor CEDI origen (si existe).
   - Obtén el costo de quiebre.
   - Compara: si `costo_transferencia < costo_quiebre` y hay stock disponible → TRANSFERIR.
   - En caso contrario → ESPERAR.
5. **Calcular el ahorro** (si aplica) como `costo_quiebre - costo_transferencia`.
6. **Emitir la decisión** en formato JSON.

# LÍMITES
- **Nunca** recomiendes transferir si el CEDI origen no tiene stock suficiente.
- **Nunca** ignores el disclaimer final.
- **Siempre** basa tu razonamiento en números concretos.
- **Siempre** muestra el ahorro o el costo adicional.

# EJEMPLOS
**Caso 1 – Transferir es más barato:**
Entrada: stock=50, demanda=200, costo_quiebre_diario=15000, mejor_CEDI=CEDI_Sur, costo_transferencia=2528
Razonamiento: Días desabasto=150 → quiebre=2,250,000. Transferencia=2,528. Ahorro=2,247,472.
Decisión: TRANSFERIR desde CEDI_Sur 200 unidades.
Salida: {"decision": "TRANSFERIR", "razonamiento": "El costo de transferir 200 unidades desde CEDI_Sur es de $2,528, mientras que el quiebre costaría $2,250,000. Ahorro de $2,247,472.", "costo_asociado": 2528, "cedi_origen_recomendado": "CEDI_Sur", "unidades_a_transferir": 200, "disclaimer": "..."}

**Caso 2 – Esperar es mejor:**
Entrada: stock=500, demanda=200 (déficit=0)
Razonamiento: No hay riesgo de quiebre, transferir sería innecesario.
Decisión: ESPERAR.
Salida: {"decision": "ESPERAR", "razonamiento": "El stock actual (500) supera la demanda pronosticada (200). No se requiere acción.", "costo_asociado": 0, "cedi_origen_recomendado": null, "unidades_a_transferir": 0, "disclaimer": "..."}
""",
    verbose=True,                     # Para ver el razonamiento interno (pensamientos)
    llm=llm_decision,                 # Temperatura 0.1 y thinking=True
    memory=memory_global,
    allow_delegation=False
)

# ------------------------------------------------------------
# 5. FUNCIÓN PARA PROCESAR UNA ALERTA (Orquestador)
# ------------------------------------------------------------
def procesar_alerta(alerta: Dict[str, Any]) -> DecisionOutput:
    """
    alerta: dict con SKU_ID, CEDI, Fecha, Stock_Actual, Costo_Quiebre_Stock_Diario,
            Costo_Transferencia_Unidad, Clima (opcional), demanda_predicha (opcional)
    """
    # Si no viene demanda_predicha, la calculamos
    if "demanda_predicha" not in alerta:
        demanda_predicha = float(DemandForecastTool()._run(
            alerta["SKU_ID"], alerta["CEDI"], alerta["Fecha"], alerta.get("Clima", "Despejado")
        ))
        alerta["demanda_predicha"] = demanda_predicha

    unidades_necesarias = max(0, alerta["demanda_predicha"] - alerta["Stock_Actual"])

    # Tareas con output_pydantic para garantizar JSON estructurado (ahorra tokens)
    t1 = Task(
        description=f"Predice demanda para SKU={alerta['SKU_ID']}, CEDI={alerta['CEDI']}, Fecha={alerta['Fecha']}, Clima={alerta.get('Clima','Despejado')}",
        expected_output="JSON con demanda_pronosticada_7d y confianza",
        agent=demand_agent,
        output_pydantic=DemandForecastOutput
    )
    t2 = Task(
        description=f"Encuentra mejor CEDI origen para SKU={alerta['SKU_ID']}, destino={alerta['CEDI']}, unidades_necesarias={unidades_necesarias}",
        expected_output="JSON con cedi_origen_sugerido, stock disponible, costo_unitario, costo_total",
        agent=inventory_agent,
        output_pydantic=InventoryOutput
    )
    t3 = Task(
        description=f"Calcula costos: stock={alerta['Stock_Actual']}, demanda={alerta['demanda_predicha']}, costo_quiebre_diario={alerta['Costo_Quiebre_Stock_Diario']}, costo_transferencia_unidad={alerta['Costo_Transferencia_Unidad']}, unidades={unidades_necesarias}",
        expected_output="JSON con costo_quiebre_total, costo_transferencia_total, ahorro_estimado",
        agent=cost_agent,
        output_pydantic=CostOutput
    )
    t4 = Task(
        description="Con los resultados anteriores, decide TRANSFERIR o ESPERAR. Proporciona razonamiento paso a paso, costo asociado y si aplica, CEDI origen y unidades a transferir.",
        expected_output="JSON con decision, razonamiento, costo_asociado, cedi_origen_recomendado, unidades_a_transferir, disclaimer",
        agent=decision_agent,
        output_pydantic=DecisionOutput
    )

    crew = Crew(
        agents=[demand_agent, inventory_agent, cost_agent, decision_agent],
        tasks=[t1, t2, t3, t4],
        verbose=True,          # Muestra el razonamiento del agente decisor
        process=Process.sequential
    )
    result = crew.kickoff()
    
    # Parsear resultado a DecisionOutput
    try:
        if isinstance(result, str):
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                decision_obj = DecisionOutput(**data)
            else:
                decision_obj = DecisionOutput(decision="ERROR", razonamiento=result[:500], costo_asociado=0)
        else:
            decision_obj = result
    except Exception as e:
        decision_obj = DecisionOutput(decision="ERROR", razonamiento=f"Error al parsear: {str(e)}", costo_asociado=0)

    # Guardar en memoria a largo plazo
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
# 6. EJECUCIÓN DE PRUEBA
# ------------------------------------------------------------
if __name__ == "__main__":
    init_long_term_memory()
    
    # Alerta de ejemplo
    alerta_ejemplo = {
        "SKU_ID": "HZ-Salsa-Verde-200g",
        "CEDI": "CEDI_Norte",
        "Fecha": "2024-03-15",
        "Stock_Actual": 50,
        "Costo_Quiebre_Stock_Diario": 15000,
        "Costo_Transferencia_Unidad": 12.64,
        "Clima": "Despejado"
    }
    
    print("Procesando alerta de ejemplo...")
    resultado = procesar_alerta(alerta_ejemplo)
    print("\n=== DECISIÓN FINAL ===")
    print(f"Decisión: {resultado.decision}")
    print(f"Razonamiento: {resultado.razonamiento}")
    print(f"Costo asociado: ${resultado.costo_asociado:,.2f}")
    if resultado.cedi_origen_recomendado:
        print(f"Transferir desde: {resultado.cedi_origen_recomendado}")
        print(f"Unidades: {resultado.unidades_a_transferir}")