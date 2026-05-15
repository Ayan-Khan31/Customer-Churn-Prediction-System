import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND FEATURE NAMES
# These files are saved from the training notebook.
# - churn_model.pkl   : trained RandomForest model
# - feature_names.pkl : list of columns the model expects
# ============================================================

@st.cache_resource  # cache so model loads only once, not on every interaction
def load_model():
    model = pickle.load(open("models/churn_model.pkl", "rb"))
    feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
    return model, feature_names

model, feature_names = load_model()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_input_row(gender, tenure, monthly_charges, contract,
                    internet_service, payment_method, dependents):
    """
    Converts raw user inputs into a one-row DataFrame
    that matches the exact feature columns the model was trained on.
    All features start at 0, then we fill in the relevant ones.
    """
    input_dict = {feature: 0 for feature in feature_names}

    # Numerical features — filled directly
    input_dict['Tenure Months'] = tenure
    input_dict['Monthly Charges'] = monthly_charges

    # One-hot encoded categorical features
    # We check if the column exists before setting it (safe approach)
    if 'Gender_Male' in feature_names:
        input_dict['Gender_Male'] = 1 if gender == "Male" else 0

    if f'Contract_{contract}' in feature_names:
        input_dict[f'Contract_{contract}'] = 1

    if f'Internet Service_{internet_service}' in feature_names:
        input_dict[f'Internet Service_{internet_service}'] = 1

    if f'Payment Method_{payment_method}' in feature_names:
        input_dict[f'Payment Method_{payment_method}'] = 1

    if 'Dependents_Yes' in feature_names:
        input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0

    # Return as DataFrame with columns in the correct order
    return pd.DataFrame([input_dict])[feature_names]


def get_risk_level(probability):
    """Returns a risk label string based on churn probability."""
    if probability > 0.7:
        return "High Risk"
    elif probability > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


def show_recommendation(probability):
    """Displays a business recommendation based on churn probability."""
    if probability > 0.7:
        st.error("🚨 Offer retention discount and immediate customer support.")
    elif probability > 0.4:
        st.warning("⚠️ Provide loyalty rewards and personalized engagement.")
    else:
        st.success("✅ Customer appears stable. Maintain engagement quality.")


def predict_batch(df_upload):
    """
    Takes an uploaded DataFrame, maps columns to model features,
    runs predictions, and returns the DataFrame with results added.
    """
    # Start with all zeros for every model feature
    batch_input = pd.DataFrame(0, index=df_upload.index, columns=feature_names)

    # Map numerical columns if they exist in the uploaded file
    for col in ['Tenure Months', 'Monthly Charges']:
        if col in df_upload.columns:
            batch_input[col] = df_upload[col].values

    # Map Gender — one-hot: Gender_Male = 1 if Male
    if 'Gender' in df_upload.columns and 'Gender_Male' in feature_names:
        batch_input['Gender_Male'] = (df_upload['Gender'] == 'Male').astype(int).values

    # Map Contract type — one-hot encode each possible value
    if 'Contract' in df_upload.columns:
        for val in ['One year', 'Two year']:
            col = f'Contract_{val}'
            if col in feature_names:
                batch_input[col] = (df_upload['Contract'] == val).astype(int).values

    # Map Internet Service — one-hot encode each possible value
    if 'Internet Service' in df_upload.columns:
        for val in ['Fiber optic', 'No']:
            col = f'Internet Service_{val}'
            if col in feature_names:
                batch_input[col] = (df_upload['Internet Service'] == val).astype(int).values

    # Map Payment Method — one-hot encode each possible value
    if 'Payment Method' in df_upload.columns:
        for val in ['Credit card (automatic)', 'Electronic check', 'Mailed check']:
            col = f'Payment Method_{val}'
            if col in feature_names:
                batch_input[col] = (df_upload['Payment Method'] == val).astype(int).values

    # Map Dependents
    if 'Dependents' in df_upload.columns and 'Dependents_Yes' in feature_names:
        batch_input['Dependents_Yes'] = (df_upload['Dependents'] == 'Yes').astype(int).values

    # Run model predictions
    predictions = model.predict(batch_input)
    probabilities = model.predict_proba(batch_input)[:, 1]

    # Add results back to the original uploaded dataframe
    result_df = df_upload.copy()
    result_df['Churn Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in predictions]
    result_df['Churn Probability (%)'] = (probabilities * 100).round(2)
    result_df['Risk Level'] = [get_risk_level(p) for p in probabilities]

    return result_df, probabilities


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Single Prediction", "Batch CSV Upload", "Business Insights"]
)

# ============================================================
# PAGE 1 — SINGLE CUSTOMER PREDICTION
# ============================================================

if page == "Single Prediction":

    st.title("📊 Customer Churn Prediction System")
    st.write("Fill in the customer details below to predict churn risk.")

    # --- INPUT FORM ---
    col_left, col_right = st.columns(2)

    with col_left:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure Months", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col_right:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    # --- PREDICT BUTTON ---
    if st.button("Predict Churn", type="primary"):

        # Build model input and run prediction
        input_data = build_input_row(
            gender, tenure, monthly_charges,
            contract, internet_service, payment_method, dependents
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        # Main result banner
        if prediction == 1:
            st.error("⚠️ Customer Likely to Churn")
        else:
            st.success("✅ Customer Likely to Stay")

        st.write(f"**Churn Probability:** {probability:.2%}")

        # KPI cards
        k1, k2, k3 = st.columns(3)
        k1.metric("Churn Probability", f"{probability:.2%}")
        k2.metric("Tenure", f"{tenure} Months")
        k3.metric("Monthly Charges", f"${monthly_charges:.2f}")

        # Risk level badge
        st.progress(float(probability))
        risk = get_risk_level(probability)
        if risk == "High Risk":
            st.error(f"🔴 {risk} Customer")
        elif risk == "Medium Risk":
            st.warning(f"🟡 {risk} Customer")
        else:
            st.success(f"🟢 {risk} Customer")

        # Pie chart — churn vs retention probability
        fig = px.pie(
            values=[probability, 1 - probability],
            names=["Churn Risk", "Retention"],
            title="Customer Risk Distribution",
            color_discrete_sequence=["#1f77b4", "#aec7e8"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Business recommendation
        st.subheader("💡 Recommended Business Action")
        show_recommendation(probability)


# ============================================================
# PAGE 2 — BATCH CSV UPLOAD
# ============================================================

elif page == "Batch CSV Upload":

    st.title("📂 Batch Customer Churn Prediction")
    st.write("Upload a CSV file with multiple customers to predict churn for all of them at once.")

    # --- EXPECTED FORMAT INFO ---
    with st.expander("📋 Expected CSV Format (click to expand)"):
        st.write("Your CSV should contain these columns:")
        st.code("""Gender, Tenure Months, Monthly Charges, Contract,
Internet Service, Payment Method, Dependents""")
        # Show a small sample so user knows exactly what format to use
        sample = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Tenure Months": [12, 36],
            "Monthly Charges": [70.0, 95.0],
            "Contract": ["Month-to-month", "One year"],
            "Internet Service": ["Fiber optic", "DSL"],
            "Payment Method": ["Electronic check", "Mailed check"],
            "Dependents": ["No", "Yes"]
        })
        st.dataframe(sample)

        # Download sample CSV button
        sample_csv = sample.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_csv,
            file_name="sample_customers.csv",
            mime="text/csv"
        )

    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:

        # Read the uploaded file
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"✅ File uploaded successfully — **{len(df_upload)} customers** found.")
        st.dataframe(df_upload.head())

        # Run batch predictions
        if st.button("Run Batch Prediction", type="primary"):

            with st.spinner("Running predictions..."):
                result_df, probabilities = predict_batch(df_upload)

            st.divider()
            st.subheader("📊 Batch Prediction Results")

            # --- SUMMARY KPI CARDS ---
            total = len(result_df)
            high_risk = (result_df['Risk Level'] == 'High Risk').sum()
            medium_risk = (result_df['Risk Level'] == 'Medium Risk').sum()
            low_risk = (result_df['Risk Level'] == 'Low Risk').sum()
            will_churn = (result_df['Churn Prediction'] == 'Will Churn').sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers", total)
            m2.metric("Predicted to Churn", will_churn)
            m3.metric("High Risk", high_risk)
            m4.metric("Churn Rate", f"{(will_churn / total * 100):.1f}%")

            # --- RESULTS TABLE ---
            st.subheader("Customer-Level Results")

            # Color code the Risk Level column for easy reading
            def color_risk(val):
                if val == "High Risk":
                    return "background-color: #ffcccc"
                elif val == "Medium Risk":
                    return "background-color: #fff3cc"
                else:
                    return "background-color: #ccffcc"

            styled_df = result_df.style.applymap(color_risk, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True)

            # --- RISK DISTRIBUTION CHART ---
            risk_counts = result_df['Risk Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']

            fig_risk = px.bar(
                risk_counts,
                x='Risk Level',
                y='Count',
                color='Risk Level',
                title="Customer Risk Distribution",
                color_discrete_map={
                    "High Risk": "#e74c3c",
                    "Medium Risk": "#f39c12",
                    "Low Risk": "#2ecc71"
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # --- CHURN PROBABILITY DISTRIBUTION ---
            fig_hist = px.histogram(
                result_df,
                x='Churn Probability (%)',
                nbins=20,
                title="Churn Probability Distribution Across Customers",
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- DOWNLOAD RESULTS ---
            st.subheader("⬇️ Download Results")
            result_csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=result_csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )


# ============================================================
# PAGE 3 — BUSINESS INSIGHTS DASHBOARD
# ============================================================

elif page == "Business Insights":

    st.title("📈 Business Insights Dashboard")
    st.write("Key findings from the trained churn model.")

    # --- KPI CARDS ---
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Model Accuracy", "79.84%")
    kpi2.metric("High Risk Drivers", "Tenure & Charges")
    kpi3.metric("Retention Opportunity", "High")

    st.divider()

    # --- FEATURE IMPORTANCE CHART ---
    # These importance values come from the trained RandomForest model
    st.subheader("Top Churn Drivers")

    feature_data = pd.DataFrame({
        "Feature": [
            "Tenure Months",
            "Monthly Charges",
            "CLTV",
            "Contract",
            "Payment Method"
        ],
        "Importance": [0.106, 0.079, 0.078, 0.029, 0.026]
    }).sort_values("Importance", ascending=True)  # ascending so largest bar is on top

    fig_feat = px.bar(
        feature_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Random Forest)",
        color="Importance",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.divider()

    # --- KEY FINDINGS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Key Findings")
        st.markdown("""
        - Customers with **low tenure** churn more frequently
        - **Month-to-month** contracts have the highest churn rate
        - **High monthly charges** correlate strongly with churn
        - **Electronic check** users are at higher risk than other payment methods
        - Customers **without dependents** tend to churn more
        """)

    with col2:
        st.subheader("✅ Retention Recommendations")
        st.markdown("""
        - Offer discounts for switching to **long-term contracts**
        - Improve **onboarding experience** for new customers (first 3 months are critical)
        - Target high-risk users with **loyalty rewards**
        - Provide **flexible payment options** to reduce friction
        - Send **proactive support** to fiber optic users with high charges
        """)