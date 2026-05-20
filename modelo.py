# entrenar_modelo_local.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# 1. Cargar datos
df = pd.read_excel("data/Data_Prueba_Tecnica_Herdez_IA.xlsx")
df["Fecha"] = pd.to_datetime(df["Fecha"])

# 2. Ordenar por SKU, CEDI y Fecha
df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).reset_index(drop=True)

# 3. Crear target: demanda total de los próximos 7 días (igual que antes)
def demanda_futura(series, horizonte=7):
    return series.shift(-horizonte).rolling(horizonte).sum().shift(-horizonte+1)

df["target_demanda_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda s: demanda_futura(s, 7)
)

# 4. Crear features (mismas que el pipeline original, pero sin limpieza extra)
# Features básicas numéricas
feature_cols = ["Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
                "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad"]

# Variables temporales: lags y medias móviles
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

# Codificación de categóricas
le_sku = LabelEncoder()
le_cedi = LabelEncoder()
le_clima = LabelEncoder()
df["SKU_encoded"] = le_sku.fit_transform(df["SKU_ID"])
df["CEDI_encoded"] = le_cedi.fit_transform(df["CEDI"])
df["Clima_encoded"] = le_clima.fit_transform(df["Clima"])

# Lista completa de features
features = ["Ventas_Unidades", "Stock_Actual", "Lead_Time_Dias", "Promocion_Activa",
            "Precio_Combustible_MXN", "Costo_Quiebre_Stock_Diario", "Costo_Transferencia_Unidad",
            "ventas_lag_1", "ventas_lag_7", "ventas_media_7d", "ventas_std_7d",
            "dias_cobertura", "promo_x_ventas", "leadtime_x_ventas",
            "SKU_encoded", "CEDI_encoded", "Clima_encoded"]

# 5. Eliminar filas con NaN (por los lags iniciales)
df_model = df.dropna(subset=features + ["target_demanda_7d"])

X = df_model[features]
y = df_model["target_demanda_7d"]

# 6. Dividir respetando orden temporal (último 20% para test)
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 7. Entrenar XGBoost (parámetros estándar)
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric="mae"
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# 8. Guardar modelo y encoders
joblib.dump(model, "modelo_xgboost_local.pkl")
joblib.dump({"sku": le_sku, "cedi": le_cedi, "clima": le_clima}, "encoders.pkl")

print("✅ Modelo entrenado y guardado como 'modelo_xgboost_local.pkl'")