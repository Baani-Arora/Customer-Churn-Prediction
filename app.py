import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "models" / "ECommerce_xgb_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Feature Engineering
# -----------------------------
def create_features(data):
    data["recency_bucket"] = pd.cut(
        data["DaySinceLastOrder"],
        bins=[-1, 3, 7, 15, 50],
        labels=["0-3d", "4-7d", "8-15d", "16d+"]
    )

    data["tenure_bucket"] = pd.cut(
        data["Tenure"],
        bins=[-1, 3, 6, 12, 24, 60],
        labels=["0-3mo", "4-6mo", "7-12mo", "13-24mo", "25mo+"]
    )

    # Same threshold used during model training
    data["frequent_shopper"] = (
        data["OrderCount"] > 2.0
    ).astype(int)

    data["inactivity_flag"] = (
        data["DaySinceLastOrder"] > 15
    ).astype(int)

    data["low_satisfaction_flag"] = (
        data["SatisfactionScore"] <= 2
    ).astype(int)

    data["high_support_contact_flag"] = data["Complain"]

    data["unhappy_and_complained"] = (
        data["low_satisfaction_flag"] &
        data["high_support_contact_flag"]
    ).astype(int)

    # Same 75th percentile threshold used during training
    data["high_cashback"] = (
        data["CashbackAmount"] > 196.3925
    ).astype(int)

    return data


# -----------------------------
# UI
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the probability "
    "that the customer may churn."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=60,
        value=10
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    warehouse_to_home = st.number_input(
        "Warehouse to Home (km)",
        min_value=0.0,
        value=15.0
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

with col2:
    login_device = st.selectbox(
        "Preferred Login Device",
        ["Mobile Phone", "Computer"]
    )

    payment_mode = st.selectbox(
        "Preferred Payment Mode",
        [
            "Credit Card",
            "Cash on Delivery",
            "Debit Card",
            "UPI",
            "E wallet"
        ]
    )

    order_category = st.selectbox(
        "Preferred Order Category",
        [
            "Laptop & Accessory",
            "Mobile Phone",
            "Fashion",
            "Grocery",
            "Others"
        ]
    )

    satisfaction = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    devices_registered = st.number_input(
        "Number of Devices Registered",
        min_value=1,
        max_value=10,
        value=3
    )

with col3:
    hours_on_app = st.number_input(
        "Hours Spent on App",
        min_value=0.0,
        max_value=10.0,
        value=2.0
    )

    number_of_address = st.number_input(
        "Number of Addresses",
        min_value=1,
        max_value=25,
        value=2
    )

    complain = st.selectbox(
        "Has the customer complained?",
        ["No", "Yes"]
    )

    order_amount_hike = st.number_input(
        "Order Amount Increase From Last Year (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

    coupon_used = st.number_input(
        "Coupons Used",
        min_value=0,
        max_value=100,
        value=2
    )

# -----------------------------
# Additional Information
# -----------------------------
st.subheader("Purchase & Activity Information")

col4, col5, col6 = st.columns(3)

with col4:
    order_count = st.number_input(
        "Order Count",
        min_value=0,
        max_value=100,
        value=2
    )

with col5:
    days_since_last_order = st.number_input(
        "Days Since Last Order",
        min_value=0,
        max_value=50,
        value=5
    )

with col6:
    cashback = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=500.0,
        value=150.0
    )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔮 Predict Churn Risk", use_container_width=True):

    # Convert Yes/No to 1/0
    complain_value = 1 if complain == "Yes" else 0

    # Create dataframe with the exact model input columns
    customer = pd.DataFrame([{
        "Tenure": tenure,
        "PreferredLoginDevice": login_device,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "PreferredPaymentMode": payment_mode,
        "Gender": gender,
        "HourSpendOnApp": hours_on_app,
        "NumberOfDeviceRegistered": devices_registered,
        "PreferedOrderCat": order_category,
        "SatisfactionScore": satisfaction,
        "MaritalStatus": marital_status,
        "NumberOfAddress": number_of_address,
        "Complain": complain_value,
        "OrderAmountHikeFromlastYear": order_amount_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_since_last_order,
        "CashbackAmount": cashback
    }])

    # Apply the same feature engineering used during training
    customer = create_features(customer)

    # Prediction
    probability = model.predict_proba(customer)[0][1]
    prediction = model.predict(customer)[0]

    probability_percent = probability * 100

    # Risk category
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.70:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # -----------------------------
    # Result
    # -----------------------------
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

    with result_col2:
        st.metric(
            "Risk Level",
            risk
        )

    if risk == "High Risk":
        st.error(
            "⚠️ This customer has a high probability of churning."
        )
    elif risk == "Medium Risk":
        st.warning(
            "⚠️ This customer has a moderate churn risk."
        )
    else:
        st.success(
            "✅ This customer has a low churn risk."
        )

    # -----------------------------
    # Risk Factors
    # -----------------------------
    st.subheader("Possible Risk Factors")

    factors = []

    if days_since_last_order > 15:
        factors.append("Customer has been inactive for a long time.")

    if satisfaction <= 2:
        factors.append("Low customer satisfaction.")

    if complain_value == 1:
        factors.append("Customer has reported a complaint.")

    if tenure <= 3:
        factors.append("Customer has a short tenure.")

    if order_count <= 2:
        factors.append("Customer has relatively low order frequency.")

    if not factors:
        factors.append(
            "No major risk factors identified from the provided inputs."
        )

    for factor in factors:
        st.write("•", factor)

st.divider()

st.caption(
    "Customer Churn Prediction | E-Commerce XGBoost Model"
)