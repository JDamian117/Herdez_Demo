import duckdb

conn = duckdb.connect("data/herdez.duckdb")

# Ver todas las columnas de dataset_modelo
print("📋 COLUMNAS EN DATASET_MODELO:")
columnas = conn.execute("DESCRIBE dataset_modelo").df()
print(columnas)

# Obtener el dataset completo
df_modelo = conn.execute("SELECT * FROM dataset_modelo").df()

# Ver primeras filas
print("\n🔍 PRIMERAS 5 FILAS (SOLO NOMBRES DE COLUMNAS):")
print(df_modelo.columns.tolist())

# Exportar a CSV con todas las columnas extra
df_modelo.to_csv("dataset_con_features.csv", index=False)
print("\n💾 Dataset completo con features guardado en: dataset_con_features.csv")

conn.close()