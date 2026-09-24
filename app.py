import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Supply Chain Intelligence", layout="centered")

@st.cache_resource   # loads the models once, reuses them for every visitor
def load_artifacts():
    return {
        "si_num": joblib.load("artifacts/si_num.joblib"),
        "si_cat": joblib.load("artifacts/si_cat.joblib"),
        "ohe": joblib.load("artifacts/ohe.joblib"),
        "scaler": joblib.load("artifacts/scaler.joblib"),
        "svr": joblib.load("artifacts/svr_model.joblib"),
        "xgb": joblib.load("artifacts/xgb_model.joblib"),
        "le": joblib.load("artifacts/label_encoder.joblib"),
        "scaler_cluster": joblib.load("artifacts/scaler_cluster.joblib"),
        "kmeans": joblib.load("artifacts/kmeans_model.joblib"),
        "meta": joblib.load("artifacts/meta.joblib"),
    }

art = load_artifacts()
meta = art["meta"]

st.title("📦 Supply Chain Intelligence")
st.caption("Predict delivery time, delivery status, and shipment profile for a new order.")

with st.form("shipment_form"):
    col1, col2 = st.columns(2)
    with col1:
        product_category = st.selectbox("Product Category", ["Apparel", "Automotive Parts", "Electronics", "Furniture", "Industrial Equipment", "Perishable Food", "Pharmaceuticals"])
        supplier_region = st.selectbox("Supplier Region", ["East Asia", "Europe", "Latin America", "North America", "South Asia", "Southeast Asia"])
        shipping_mode = st.selectbox("Shipping Mode", ["Air", "Rail", "Road", "Sea"])
        warehouse_type = st.selectbox("Warehouse Type", ["Automated", "Cross-Dock", "Hybrid", "Manual"])
    with col2:
        order_quantity = st.number_input("Order Quantity", min_value=1, value=500)
        distance_km = st.number_input("Distance (km)", min_value=0.0, value=800.0)
        lead_time = st.number_input("Supplier Lead Time (days)", min_value=0.0, value=7.0)
        transport_cost = st.number_input("Transportation Cost", min_value=0.0, value=2000.0)

    inventory_level = st.number_input("Inventory Level", min_value=0.0, value=300.0)
    demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=450.0)
    reliability = st.slider("Supplier Reliability Score", 0.0, 100.0, 85.0)  
    delay_rate = st.slider("Historical Delay Rate", 0.0, 1.0, 0.1) 

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = pd.DataFrame([{
        "Product_Category": product_category,
        "Supplier_Region": supplier_region,
        "Shipping_Mode": shipping_mode,
        "Warehouse_Type": warehouse_type,
        "Order_Quantity": order_quantity,
        "Distance_km": distance_km,
        "Supplier_Lead_Time_days": lead_time,
        "Transportation_Cost": transport_cost,
        "Inventory_Level": inventory_level,
        "Demand_Forecast": demand_forecast,
        "Supplier_Reliability_Score": reliability,
        "Historical_Delay_Rate": delay_rate,
    }])

    X = raw.copy()
    X[meta["log_col"]] = np.log1p(X[meta["log_col"]])

    X_cat = pd.DataFrame(
        art["ohe"].transform(X[meta["ohe_col"]]),
        columns=art["ohe"].get_feature_names_out(meta["ohe_col"]),
    )
    X = pd.concat([X.drop(columns=meta["ohe_col"]).reset_index(drop=True), X_cat], axis=1)

    X[meta["num_col"]] = art["scaler"].transform(X[meta["num_col"]])

    X = X[meta["feature_columns"]]

    pred_days = art["svr"].predict(X)[0]

    pred_status_enc = art["xgb"].predict(X)[0]
    pred_status = art["le"].inverse_transform([pred_status_enc])[0]

    cluster_input = raw[meta["cluster_cols"]]
    cluster_scaled = art["scaler_cluster"].transform(cluster_input)
    cluster_id = art["kmeans"].predict(cluster_scaled)[0]

    st.success("Prediction complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Delivery Time", f"{pred_days:.1f} days")
    c2.metric("Delivery Status", pred_status)
    c3.metric("Shipment Profile", f"Cluster {cluster_id}") 