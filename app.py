import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="CreditWise — Intelligent Loan Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_artifacts():
    return {
        "log_model": joblib.load(MODEL_DIR / "log_model.pkl"),
        "ohe": joblib.load(MODEL_DIR / "ohe.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "feature_columns": joblib.load(MODEL_DIR / "feature_columns.pkl"),
    }


artifacts = load_artifacts()

log_model = artifacts["log_model"]
ohe = artifacts["ohe"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]


EMPLOYMENT = ["Contract", "Salaried", "Self-employed", "Unemployed"]
MARITAL = ["Married", "Single"]
PURPOSE = ["Business", "Car", "Education", "Home", "Personal"]
PROPERTY = ["Rural", "Semiurban", "Urban"]
EDUCATION = ["Graduate", "Not Graduate"]
GENDER = ["Female", "Male"]
EMPLOYER = ["Business", "Government", "MNC", "Private", "Unemployed"]


# -----------------------------
# Premium UI
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(61, 92, 255, .13), transparent 30%),
            radial-gradient(circle at 90% 12%, rgba(0, 214, 170, .09), transparent 28%),
            #07111f;
        color: #edf3ff;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 28px;
        padding: 34px 38px 32px;
        background: linear-gradient(135deg, rgba(19,35,61,.92), rgba(9,22,39,.78));
        box-shadow: 0 22px 70px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }

    .hero:after {
        content: '';
        position: absolute;
        width: 280px;
        height: 280px;
        right: -100px;
        top: -140px;
        border-radius: 50%;
        border: 1px solid rgba(120,150,255,.18);
        box-shadow:
            0 0 0 35px rgba(120,150,255,.035),
            0 0 0 70px rgba(120,150,255,.025);
    }

    .brand {
        color:#9aaeff;
        letter-spacing: .18em;
        text-transform:uppercase;
        font-size:.76rem;
        font-weight:700;
    }

    .hero h1 {
        font-family:'Space Grotesk',sans-serif;
        font-size:2.7rem;
        line-height:1.05;
        margin:9px 0 10px;
        color:#fff;
    }

    .hero p {
        color:#aebbd0;
        max-width:720px;
        font-size:1rem;
        margin:0;
    }

    .badge {
        display:inline-flex;
        align-items:center;
        gap:8px;
        margin-top:20px;
        padding:7px 12px;
        border-radius:999px;
        background:rgba(28,208,169,.09);
        border:1px solid rgba(28,208,169,.22);
        color:#76ebcf;
        font-size:.78rem;
        font-weight:600;
    }

    .dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:#31dfb5;
        box-shadow:0 0 12px #31dfb5;
    }

    .section-card {
        border:1px solid rgba(255,255,255,.08);
        border-radius:22px;
        padding:23px 25px 18px;
        background:rgba(13,27,46,.72);
        box-shadow:0 15px 45px rgba(0,0,0,.14);
        margin:12px 0;
    }

    .section-title {
        font-family:'Space Grotesk',sans-serif;
        font-size:1.12rem;
        font-weight:600;
        color:#fff;
        margin-bottom:2px;
    }

    .section-subtitle {
        color:#778aa5;
        font-size:.78rem;
        margin-bottom:14px;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background:#0b1a2d !important;
        border-color:#203651 !important;
        color:#edf3ff !important;
        border-radius:12px !important;
    }

    label {
        color:#9eb0c9 !important;
        font-size:.83rem !important;
    }

    .stNumberInput button {
        color:#9eb0c9 !important;
    }

    div[data-testid="stForm"] {
        border:0;
        padding:0;
    }

    .stButton > button {
        border:0;
        border-radius:14px;
        min-height:50px;
        font-weight:700;
        font-size:.94rem;
        color:#06131f;
        background:linear-gradient(135deg,#77e9d0,#6ca7ff);
        box-shadow:0 12px 30px rgba(65,166,225,.20);
        transition:.2s ease;
    }

    .stButton > button:hover {
        transform:translateY(-1px);
        box-shadow:0 16px 36px rgba(65,166,225,.28);
    }

    .metric-card {
        border:1px solid rgba(255,255,255,.08);
        border-radius:20px;
        padding:24px;
        background:linear-gradient(
            160deg,
            rgba(17,36,60,.92),
            rgba(9,23,40,.92)
        );
        min-height:210px;
    }

    .metric-name {
        color:#9fb0c8;
        font-size:.8rem;
        font-weight:600;
    }

    .prediction {
        font-family:'Space Grotesk';
        font-size:2rem;
        font-weight:700;
        margin:18px 0 5px;
    }

    .approved {
        color:#6fe5c8;
    }

    .rejected {
        color:#ff8497;
    }

    .prob-label {
        color:#7387a3;
        font-size:.73rem;
        margin-top:16px;
    }

    .bar {
        height:8px;
        border-radius:20px;
        background:#16283e;
        overflow:hidden;
        margin-top:7px;
    }

    .bar-fill {
        height:100%;
        border-radius:20px;
        background:linear-gradient(90deg,#6ca7ff,#77e9d0);
    }

    .summary {
        border:1px solid rgba(119,233,208,.16);
        border-radius:22px;
        padding:24px 26px;
        background:linear-gradient(
            135deg,
            rgba(21,61,69,.50),
            rgba(10,30,47,.72)
        );
        margin-top:18px;
    }

    .summary-kicker {
        text-transform:uppercase;
        letter-spacing:.13em;
        color:#73d9c3;
        font-size:.7rem;
        font-weight:700;
    }

    .summary h2 {
        font-family:'Space Grotesk';
        margin:7px 0 3px;
        font-size:1.8rem;
    }

    .summary p {
        color:#9fb0c8;
        margin:0;
    }

    .agreement {
        margin-top:16px;
        color:#cad5e5;
        font-size:.84rem;
    }

    .footer-note {
        color:#657892;
        font-size:.72rem;
        margin-top:22px;
        text-align:center;
    }

    div[data-testid="stTabs"] button {
        color:#8295ae;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color:#fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Hero section
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="brand">CreditWise — Loan Approval Prediction System · </div>

      <h1>Make a smarter loan decision.</h1>

      <p>
        Predict loan approval using applicant financial, 
        demographic, and credit information.
      </p>

      <div class="badge">
        <span class="dot"></span>
        Logistic Regression model online
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Feature construction
# -----------------------------
def make_features(values):

    raw = pd.DataFrame([values])

    # Education encoding
    raw["Education_Level"] = raw["Education_Level"].map({
        "Graduate": 0,
        "Not Graduate": 1
    })

    # Categorical columns
    categorical_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    # Apply the SAME encoder used during training
    encoded = ohe.transform(raw[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=raw.index,
    )

    # Remove original categorical columns
    raw = raw.drop(columns=categorical_cols)

    # Combine numerical + encoded features
    X = pd.concat([raw, encoded_df], axis=1)

    # Feature engineering
    X["DTI_Ratio_sq"] = X["DTI_Ratio"] ** 2
    X["Credit_Score_sq"] = X["Credit_Score"] ** 2

    # Remove original variables
    X = X.drop(
        columns=[
            "Credit_Score",
            "DTI_Ratio"
        ]
    )

    # Ensure exact training feature order
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the SAME scaler used during training
    X_scaled = scaler.transform(X)

    return X_scaled


# -----------------------------
# Applicant form
# -----------------------------
st.markdown(
    '<div class="section-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Applicant profile</div>'
    '<div class="section-subtitle">'
    'Tell CreditWise who is applying.'
    '</div>',
    unsafe_allow_html=True
)


with st.form("loan_form"):

    tab1, tab2, tab3 = st.tabs(
        [
            "01 · Personal",
            "02 · Financial",
            "03 · Loan request"
        ]
    )

    # -------------------------
    # Personal
    # -------------------------
    with tab1:

        a, b, c = st.columns(3)

        with a:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1
            )

            gender = st.selectbox(
                "Gender",
                GENDER
            )

        with b:
            marital_status = st.selectbox(
                "Marital Status",
                MARITAL
            )

            dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

        with c:
            education_level = st.selectbox(
                "Education Level",
                EDUCATION
            )

            employment_status = st.selectbox(
                "Employment Status",
                EMPLOYMENT
            )


    # -------------------------
    # Financial
    # -------------------------
    with tab2:

        a, b, c = st.columns(3)

        with a:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            coapplicant_income = st.number_input(
                "Co-applicant Income",
                min_value=0.0,
                value=0.0,
                step=1000.0
            )

        with b:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300.0,
                max_value=900.0,
                value=700.0,
                step=1.0
            )

            existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

        with c:
            dti_ratio = st.number_input(
                "DTI Ratio",
                min_value=0.0,
                max_value=10.0,
                value=0.30,
                step=0.01
            )

            savings = st.number_input(
                "Savings",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )


    # -------------------------
    # Loan request
    # -------------------------
    with tab3:

        a, b, c = st.columns(3)

        with a:
            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=200000.0,
                step=1000.0
            )

            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=1,
                max_value=480,
                value=60,
                step=1
            )

        with b:
            loan_purpose = st.selectbox(
                "Loan Purpose",
                PURPOSE
            )

            property_area = st.selectbox(
                "Property Area",
                PROPERTY
            )

        with c:
            collateral_value = st.number_input(
                "Collateral Value",
                min_value=0.0,
                value=100000.0,
                step=1000.0
            )

            employer_category = st.selectbox(
                "Employer Category",
                EMPLOYER
            )


    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "RUN CREDIT ANALYSIS  →",
        type="primary",
        use_container_width=True
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Prediction
# -----------------------------
if submitted:

    values = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    try:

        # -----------------------------
        # Prepare applicant data
        # -----------------------------
        X_new = make_features(values)

        # -----------------------------
        # Logistic Regression
        # -----------------------------
        prediction = int(log_model.predict(X_new)[0])

        approval_probability = float(
            log_model.predict_proba(X_new)[0, 1]
        )

        probability_pct = approval_probability * 100

        # -----------------------------
        # Analysis Result
        # -----------------------------
        st.markdown("## Analysis Result")

        st.caption(
            "Prediction generated using the trained Logistic Regression model."
        )

        # -----------------------------
        # Decision
        # -----------------------------
        if prediction == 1:

            st.success("### ✓ LOAN APPROVED")

        else:

            st.error("### ✕ LOAN REJECTED")

        # -----------------------------
        # Metrics
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Approval Probability",
                f"{probability_pct:.1f}%"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                "50%"
            )

        # Probability bar
        st.progress(
            min(max(approval_probability, 0.0), 1.0)
        )

        # -----------------------------
        # Explanation
        # -----------------------------
        if prediction == 1:

            st.info(
                f"""
                **CreditWise assessment**

                The Logistic Regression model estimates an approval
                probability of **{probability_pct:.1f}%**.

                Since this probability is above the current 50% decision
                threshold, the application is classified as **Approved**.
                """
            )

        else:

            st.warning(
                f"""
                **CreditWise assessment**

                The Logistic Regression model estimates an approval
                probability of **{probability_pct:.1f}%**.

                Since this probability is below the current 50% decision
                threshold, the application is classified as **Rejected**.
                """
            )

        # -----------------------------
        # Footer
        # -----------------------------
        st.caption(
            "CreditWise is a machine-learning decision-support demonstration. "
            "Final lending decisions require appropriate human and institutional review."
        )

    except Exception as e:

        st.error("Prediction could not be completed.")

        st.exception(e)