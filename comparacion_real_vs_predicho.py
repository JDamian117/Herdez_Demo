# diagnostico_modelo.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Cargar modelo y encoders
model = joblib.load("modelo_xgboost_local.pkl")
encoders = joblib.load("encoders.pkl")

# 2. Cargar datos originales
df = pd.read_excel("data/Data_Prueba_Tecnica_Herdez_IA.xlsx")
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).reset_index(drop=True)

# 3. Crear target (demanda 7 días)
def demanda_futura(series, horizonte=7):
    return series.shift(-horizonte).rolling(horizonte).sum().shift(-horizonte+1)

df["target_demanda_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda s: demanda_futura(s, 7)
)

# 4. Crear features
df["ventas_lag_1"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].shift(1)
df["ventas_lag_7"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].shift(7)
df["ventas_media_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
df["ventas_std_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda x: x.shift(1).rolling(7).std()
)
df["dias_cobertura"] = df["Stock_Actual"] / (df["Ventas_Unidades"] + 1)
df["promo_x_ventas"] = df["Promocion_Activa"] * df["Ventas_Unidades"]
df["leadtime_x_ventas"] = df["Lead_Time_Dias"] * df["Ventas_Unidades"]

# Codificar categóricas
df["SKU_encoded"] = encoders["sku"].transform(df["SKU_ID"])
df["CEDI_encoded"] = encoders["cedi"].transform(df["CEDI"])
df["Clima_encoded"] = encoders["clima"].transform(df["Clima"])

# Features finales
features = ["Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
            "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
            "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
            "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
            "SKU_encoded", "CEDI_encoded", "Clima_encoded"]

# 5. Eliminar filas con NaN (por lags iniciales)
df_model = df.dropna(subset=features + ["target_demanda_7d"]).copy()
X = df_model[features]
y = df_model["target_demanda_7d"]

# 6. Usar el mismo split que en entrenamiento (primer 80% train, último 20% test)
split_idx = int(0.8 * len(X))
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# 7. Predicciones
y_pred = model.predict(X_test)

# 8. DataFrame comparativo
comp = pd.DataFrame({
    'Real': y_test.values,
    'Predicho': y_pred,
    'SKU_ID': df_model.iloc[split_idx:]["SKU_ID"].values,
    'CEDI': df_model.iloc[split_idx:]["CEDI"].values,
    'Fecha': df_model.iloc[split_idx:]["Fecha"].values
})
print("Primeras 10 filas de comparación:")
print(comp.head(10))

# Guardar CSV
comp.to_csv("comparacion_real_vs_predicho.csv", index=False)
print("\n✅ Archivo 'comparacion_real_vs_predicho.csv' guardado.")

# 9. Gráfico de dispersión
plt.figure(figsize=(8,6))
plt.scatter(comp['Real'], comp['Predicho'], alpha=0.5)
min_val = min(comp['Real'].min(), comp['Predicho'].min())
max_val = max(comp['Real'].max(), comp['Predicho'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title('Predicción vs Real')
plt.legend()
plt.show()

# 10. Calcular métricas manualmente
errores = comp['Real'] - comp['Predicho']
mae_manual = errores.abs().mean()
rmse_manual = (errores**2).mean()**0.5
ss_res = (errores**2).sum()
ss_tot = ((comp['Real'] - comp['Real'].mean())**2).sum()
r2_manual = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

print(f"\nMétricas calculadas sobre este conjunto:")
print(f"MAE  : {mae_manual:.2f}")
print(f"RMSE : {rmse_manual:.2f}")
print(f"R²   : {r2_manual:.4f}")
print(f"SS_res: {ss_res:.2f}, SS_tot: {ss_tot:.2f}")