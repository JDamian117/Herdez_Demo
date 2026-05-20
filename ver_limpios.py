import duckdb

conn = duckdb.connect("data/herdez.duckdb")

print("📄 DATOS LIMPIOS (primeras 30 filas):")
print("="*60)
df = conn.execute("SELECT * FROM inventario_limpio LIMIT 30").df()
print(df.to_string())

print("\n📊 ESTADÍSTICAS BÁSICAS:")
print(conn.execute("DESCRIBE inventario_limpio").df())