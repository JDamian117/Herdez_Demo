# evaluar_modelo_completo.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# 1. Cargar datos crudos
df = pd.read_excel("data/Data_Prueba_Tecnica_Herdez_IA.xlsx")
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).reset_index(drop=True)

# 2. Crear target (demanda acumulada a 7 días)
def demanda_futura(series, horizonte=7):
    return series.shift(-horizonte).rolling(horizonte).sum().shift(-horizonte+1)

df["target_demanda_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda s: demanda_futura(s, 7)
)

# 3. Crear features
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

# Codificación de categóricas (usar los encoders guardados, o reentrenar si no existen)
# Primero intenta cargar encoders, si no, entrena nuevos
try:
    encoders = joblib.load("encoders.pkl")
    le_sku = encoders["sku"]
    le_cedi = encoders["cedi"]
    le_clima = encoders["clima"]
    print("Encoders cargados desde disco.")
except:
    print("No se encontraron encoders, entrenando nuevos...")
    from sklearn.preprocessing import LabelEncoder
    le_sku = LabelEncoder()
    le_cedi = LabelEncoder()
    le_clima = LabelEncoder()
    le_sku.fit(df["SKU_ID"])
    le_cedi.fit(df["CEDI"])
    le_clima.fit(df["Clima"])
    # Guardar para futuras ejecuciones
    joblib.dump({"sku": le_sku, "cedi": le_cedi, "clima": le_clima}, "encoders.pkl")

df["SKU_encoded"] = le_sku.transform(df["SKU_ID"])
df["CEDI_encoded"] = le_cedi.transform(df["CEDI"])
df["Clima_encoded"] = le_clima.transform(df["Clima"])

# Lista de features (misma que en entrenamiento)
features = ["Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
            "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
            "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
            "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
            "SKU_encoded", "CEDI_encoded", "Clima_encoded"]

# Eliminar filas con NaN (las primeras filas de cada grupo por los lags)
df_model = df.dropna(subset=features + ["target_demanda_7d"]).reset_index(drop=True)

X = df_model[features]
y = df_model["target_demanda_7d"]

# 4. Split temporal (80% train, 20% test) - mismo que en entrenamiento
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 5. Cargar modelo
model = joblib.load("modelo_xgboost_local.pkl")

# 6. Predicciones
y_pred = model.predict(X_test)

# 7. Métricas
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
max_err = np.max(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

# Métricas normalizadas
mean_true = y_test.mean()
mae_norm = mae / mean_true
rmse_norm = rmse / mean_true

# Sesgo
bias = np.mean(y_pred - y_test)

# Correlaciones
pearson_corr, _ = pearsonr(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

# Baseline (predecir media)
baseline_pred = np.full_like(y_test, y_train.mean())  # usar media de train
baseline_mae = mean_absolute_error(y_test, baseline_pred)
improvement = (baseline_mae - mae) / baseline_mae * 100

# 8. Resultados
print("\n" + "="*50)
print("EVALUACIÓN DEL MODELO LOCAL")
print("="*50)
print(f"MAE  (Error Absoluto Medio): {mae:.2f} unidades")
print(f"MedAE (Error Absoluto Mediano): {medae:.2f} unidades")
print(f"RMSE (Raíz Error Cuadrático Medio): {rmse:.2f} unidades")
print(f"Max Error: {max_err:.2f} unidades")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")
print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")
print(f"MAE Normalizado: {mae_norm:.3f}")
print(f"RMSE Normalizado: {rmse_norm:.3f}")
print(f"Sesgo (MFE): {bias:.2f} unidades")
print(f"Correlación de Pearson: {pearson_corr:.3f}")
print(f"Correlación de Spearman: {spearman_corr:.3f}")
print(f"\nBaseline (media): MAE = {baseline_mae:.2f} unidades")
print(f"🚀 Mejora sobre baseline: {improvement:.1f}%")
print("="*50)

# 9. Guardar predicciones
results_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'residual': y_test - y_pred
})
results_df.to_csv('predicciones_vs_reales.csv', index=False)
print("\n💾 Predicciones guardadas en 'predicciones_vs_reales.csv'")