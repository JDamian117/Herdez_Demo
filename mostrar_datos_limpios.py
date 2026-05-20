# =============================================================
# PIPELINE DE LIMPIEZA Y FEATURES — GRUPO HERDEZ
# =============================================================
# Objetivo: Automatizar la limpieza y preparación de datos
# cada vez que llegue un Excel nuevo — sin intervención manual.
#
# En producción GCP esto sería:
# DuckDB local → BigQuery + Vertex AI Pipelines automático
# =============================================================

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# CONFIGURACIÓN
# =============================================================

EXCEL_PATH = "data/Data_Prueba_Tecnica_Herdez_IA.xlsx"
DB_PATH    = "data/herdez.duckdb"  # base de datos local persistente

# =============================================================
# 1. CONEXIÓN A DUCKDB
# =============================================================

def conectar_db():
    """
    Crea o conecta a la base de datos DuckDB local.
    En producción GCP: reemplazar por conexión a BigQuery.
    """
    conn = duckdb.connect(DB_PATH)
    return conn


# =============================================================
# 2. CARGA DEL EXCEL A DUCKDB
# =============================================================

def cargar_excel(conn):
    """
    Lee el Excel y lo carga en DuckDB como tabla.
    DuckDB lee Excel via pandas — luego todo es SQL.
    En producción GCP: Cloud Storage → BigQuery via Dataflow.
    """
    print("📂 Cargando Excel en DuckDB...")

    # Leer Excel con pandas
    df_raw = pd.read_excel(EXCEL_PATH)

    # Registrar como tabla en DuckDB
    conn.execute("DROP TABLE IF EXISTS inventario_raw")
    conn.execute("CREATE TABLE inventario_raw AS SELECT * FROM df_raw")

    total = conn.execute("SELECT COUNT(*) FROM inventario_raw").fetchone()[0]
    print(f"✅ {total} registros cargados en DuckDB")

    return df_raw


# =============================================================
# 3. LIMPIEZA AUTOMÁTICA CON SQL
# =============================================================

def limpiar_datos(conn):
    """
    Limpieza automática con SQL en DuckDB.
    Ventaja: mismo SQL funciona en BigQuery en producción.
    """
    print("\n🧹 Limpiando datos automáticamente...")

    conn.execute("DROP TABLE IF EXISTS inventario_limpio")

    conn.execute("""
        CREATE TABLE inventario_limpio AS
        SELECT
            -- Convertir fecha correctamente
            CAST(Fecha AS DATE)                    AS Fecha,
            SKU_ID,
            CEDI,

            -- Ventas: no pueden ser negativas
            CASE
                WHEN Ventas_Unidades < 0 THEN 0
                ELSE Ventas_Unidades
            END                                    AS Ventas_Unidades,

            -- Stock: no puede ser negativo
            CASE
                WHEN Stock_Actual < 0 THEN 0
                ELSE Stock_Actual
            END                                    AS Stock_Actual,

            -- Lead time: mínimo 1 día
            CASE
                WHEN Lead_Time_Dias <= 0 THEN 1
                ELSE Lead_Time_Dias
            END                                    AS Lead_Time_Dias,

            -- Promoción: solo 0 o 1
            CASE
                WHEN Promocion_Activa NOT IN (0, 1) THEN 0
                ELSE Promocion_Activa
            END                                    AS Promocion_Activa,

            -- Precio combustible: no puede ser negativo
            CASE
                WHEN Precio_Combustible_MXN <= 0 THEN NULL
                ELSE Precio_Combustible_MXN
            END                                    AS Precio_Combustible_MXN,

            -- Clima: valor directo
            Clima,

            -- Costos: no pueden ser negativos
            CASE
                WHEN Costo_Quiebre_Stock_Diario < 0 THEN 0
                ELSE Costo_Quiebre_Stock_Diario
            END                                    AS Costo_Quiebre_Stock_Diario,

            CASE
                WHEN Costo_Transferencia_Unidad < 0 THEN 0
                ELSE Costo_Transferencia_Unidad
            END                                    AS Costo_Transferencia_Unidad

        FROM inventario_raw

        -- Solo registros con datos mínimos válidos
        WHERE Fecha IS NOT NULL
          AND SKU_ID IS NOT NULL
          AND CEDI IS NOT NULL

        -- Eliminar duplicados exactos (el WHERE ya filtró nulos)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY Fecha, SKU_ID, CEDI
            ORDER BY Fecha
        ) = 1
    """)

    total_limpio = conn.execute("SELECT COUNT(*) FROM inventario_limpio").fetchone()[0]
    print(f"✅ {total_limpio} registros después de limpieza")

    # Reporte de calidad
    reporte = conn.execute("""
        SELECT
            COUNT(*)                                        AS total_registros,
            COUNT(DISTINCT SKU_ID)                         AS skus_unicos,
            COUNT(DISTINCT CEDI)                           AS cedis_unicos,
            MIN(Fecha)                                     AS fecha_inicio,
            MAX(Fecha)                                     AS fecha_fin,
            ROUND(AVG(Ventas_Unidades), 1)                AS ventas_promedio,
            ROUND(AVG(Stock_Actual), 1)                   AS stock_promedio,
            SUM(CASE WHEN Stock_Actual = 0 THEN 1 END)    AS dias_sin_stock
        FROM inventario_limpio
    """).df()

    print("\n📊 REPORTE DE CALIDAD:")
    print(reporte.to_string(index=False))

    return conn.execute("SELECT * FROM inventario_limpio ORDER BY SKU_ID, CEDI, Fecha").df()


# =============================================================
# 4. IMPUTACIÓN DE NULOS
# =============================================================

def imputar_nulos(conn):
    """
    Imputa valores faltantes por mediana de SKU.
    Respeta el comportamiento individual de cada producto.
    """
    print("\n🔧 Imputando valores faltantes...")

    conn.execute("DROP TABLE IF EXISTS inventario_imputado")

    conn.execute("""
        CREATE TABLE inventario_imputado AS
        SELECT
            Fecha,
            SKU_ID,
            CEDI,
            Ventas_Unidades,
            Stock_Actual,
            Lead_Time_Dias,
            Promocion_Activa,

            -- Imputa precio combustible con mediana del CEDI
            COALESCE(
                Precio_Combustible_MXN,
                MEDIAN(Precio_Combustible_MXN) OVER (PARTITION BY CEDI)
            )                                  AS Precio_Combustible_MXN,

            -- Imputa clima con moda global
            COALESCE(
                Clima,
                (SELECT Clima FROM inventario_limpio
                 GROUP BY Clima ORDER BY COUNT(*) DESC LIMIT 1)
            )                                  AS Clima,

            Costo_Quiebre_Stock_Diario,
            Costo_Transferencia_Unidad

        FROM inventario_limpio
    """)

    nulos = conn.execute("""
        SELECT COUNT(*) - COUNT(Precio_Combustible_MXN) AS nulos_combustible,
               COUNT(*) - COUNT(Clima)                  AS nulos_clima
        FROM inventario_imputado
    """).df()

    print(f"✅ Nulos restantes: {nulos.to_string(index=False)}")

    return conn.execute("SELECT * FROM inventario_imputado ORDER BY SKU_ID, CEDI, Fecha").df()


# =============================================================
# 5. FEATURE ENGINEERING
# =============================================================

def crear_features(df):
    """
    Crea variables temporales que el modelo XGBoost necesita.
    XGBoost no ve el tiempo solo — hay que dárselo como features.
    """
    print("\n⚙️  Creando features temporales...")

    df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).reset_index(drop=True)

    # Log de ventas — suaviza distribución sesgada
    df["ventas_log"] = np.log1p(df["Ventas_Unidades"])

    # Lags — ventas de días anteriores
    df["ventas_lag_1"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].shift(1)
    df["ventas_lag_7"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].shift(7)

    # Promedios móviles — tendencia reciente
    df["ventas_media_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["ventas_std_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    # Días de cobertura — feature más directa de riesgo de quiebre
    df["dias_cobertura"] = df["Stock_Actual"] / (df["Ventas_Unidades"] + 1)

    # Interacciones de negocio
    df["promo_x_ventas"]    = df["Promocion_Activa"] * df["Ventas_Unidades"]
    df["leadtime_x_ventas"] = df["Lead_Time_Dias"]   * df["Ventas_Unidades"]

    # Encoding de clima
    clima_map = {"Despejado": 0, "Nublado": 1, "Lluvioso": 2, "Tormenta": 3}
    df["clima_encoded"] = df["Clima"].map(clima_map).fillna(0)

    print("✅ Features creadas correctamente")
    return df


# =============================================================
# 6. CONSTRUCCIÓN DEL TARGET
# =============================================================

def crear_target(df, horizonte=7):
    """
    Target: demanda total en los próximos N días por SKU/CEDI.
    Si demanda_predicha > Stock_Actual → riesgo de quiebre.
    """
    print(f"\n🎯 Creando target — demanda próximos {horizonte} días...")

    def demanda_futura(series, h):
        return (
            series.shift(-1)
                  .iloc[::-1]
                  .rolling(h, min_periods=h)
                  .sum()
                  .iloc[::-1]
        )

    df["target_demanda_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
        lambda s: demanda_futura(s, horizonte)
    )

    df_model = df.dropna().copy()
    print(f"✅ Dataset final: {df_model.shape[0]} registros listos para XGBoost")

    return df_model


# =============================================================
# 7. FUNCIÓN PRINCIPAL — EJECUTA TODO EL PIPELINE
# =============================================================

def ejecutar_pipeline():
    """
    Pipeline completo de extremo a extremo.
    Se llama con un botón desde Streamlit o automáticamente.

    En producción GCP:
    → Vertex AI Pipelines ejecuta esto automáticamente
       cuando llega un CSV nuevo a Cloud Storage.
    """
    print("=" * 60)
    print("🚀 INICIANDO PIPELINE DE DATOS — GRUPO HERDEZ")
    print("=" * 60)

    # Conectar a DuckDB
    conn = conectar_db()

    # Ejecutar pipeline paso a paso
    cargar_excel(conn)
    df_limpio    = limpiar_datos(conn)
    df_imputado  = imputar_nulos(conn)
    df_features  = crear_features(df_imputado)
    df_final     = crear_target(df_features)

    # Guardar dataset final en DuckDB
    conn.execute("DROP TABLE IF EXISTS dataset_modelo")
    conn.execute("CREATE TABLE dataset_modelo AS SELECT * FROM df_final")

    conn.close()

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO — DATOS LISTOS PARA XGBOOST")
    print(f"   Registros: {df_final.shape[0]}")
    print(f"   Features:  {df_final.shape[1]}")
    print("=" * 60)

    return df_final


# =============================================================
# EJECUTAR DIRECTAMENTE
# =============================================================

if __name__ == "__main__":
    df = ejecutar_pipeline()
    print("\n🔍 MUESTRA DEL DATASET FINAL:")
    print(df[["Fecha", "SKU_ID", "CEDI", "Stock_Actual",
              "dias_cobertura", "target_demanda_7d"]].head(10).to_string(index=False))
    