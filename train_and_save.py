import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
)
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

os.makedirs("artifacts", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/supply_chain_intelligence_Dataset.csv")

# 2. Missing value treatment
num_col = df.select_dtypes(include="number").columns.drop(["Delivery_Time_days"])
cat_col = df.select_dtypes(include="object").columns.drop(["Delivery_Status"])

si_num = SimpleImputer(strategy="median")
df[num_col] = si_num.fit_transform(df[num_col])

si_cat = SimpleImputer(strategy="most_frequent")
df[cat_col] = si_cat.fit_transform(df[cat_col])

# 3. Feature / target split
X = df.drop(columns=["Delivery_Time_days", "Delivery_Status"])
y_reg = df["Delivery_Time_days"]
y_clf = df["Delivery_Status"]

# 4. Log transform (skew reduction)
log_col = ["Order_Quantity", "Distance_km", "Transportation_Cost",
           "Inventory_Level", "Demand_Forecast"]
log_transformer = FunctionTransformer(np.log1p)
X[log_col] = log_transformer.transform(X[log_col])

# 5. One-hot encode categoricals
ohe_col = ["Product_Category", "Supplier_Region", "Shipping_Mode", "Warehouse_Type"]
ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
X_cat = pd.DataFrame(
    ohe.fit_transform(X[ohe_col]),
    columns=ohe.get_feature_names_out(ohe_col),
    index=X.index,
)
X = pd.concat([X.drop(columns=ohe_col), X_cat], axis=1)

# 6. Scale numeric columns
scaler = StandardScaler()
X[num_col] = scaler.fit_transform(X[num_col])

# 7. Train the winning regression model (delivery time)
svr = SVR(kernel="rbf", C=10, epsilon=0.3)
svr.fit(X, y_reg)

# 8. Train the winning classification model (delivery status)
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
xgb_clf = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    random_state=42, eval_metric="mlogloss"
)
xgb_clf.fit(X, y_clf_enc)

# 9. Shipment clustering (KMeans) — fit on the same 8 numeric columns
cluster_cols = list(num_col)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_cols])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

# 10. Save everything the app will need
joblib.dump(si_num, "artifacts/si_num.joblib")
joblib.dump(si_cat, "artifacts/si_cat.joblib")
joblib.dump(ohe, "artifacts/ohe.joblib")
joblib.dump(scaler, "artifacts/scaler.joblib")
joblib.dump(svr, "artifacts/svr_model.joblib")
joblib.dump(xgb_clf, "artifacts/xgb_model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib")
joblib.dump(scaler_cluster, "artifacts/scaler_cluster.joblib")
joblib.dump(kmeans, "artifacts/kmeans_model.joblib")

meta = {
    "num_col": list(num_col),
    "cat_col": list(cat_col),
    "log_col": log_col,
    "ohe_col": ohe_col,
    "cluster_cols": cluster_cols,
    "feature_columns": list(X.columns),   # exact column order the models expect
}
joblib.dump(meta, "artifacts/meta.joblib")

print("Training complete. Artifacts saved to /artifacts")