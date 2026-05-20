import duckdb

conn = duckdb.connect("data/herdez.duckdb")
df_limpio = conn.execute("SELECT * FROM inventario_limpio").df()
conn.close()

df_limpio.to_csv("inventario_limpio.csv", index=False)
print("✅ Datos limpios exportados a 'inventario_limpio.csv'")
print(f"📊 Total filas: {len(df_limpio)}")

