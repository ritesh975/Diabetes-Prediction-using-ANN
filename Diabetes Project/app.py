import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction - ANN",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "diabetes_ann_model.keras"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Artificial Neural Network (ANN) with Sigmoid Activation</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever",
            "No Info"
        ]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=10.0,
        value=5.5,
        step=0.1
    )

    glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400,
        value=120
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
):

    # Create input dataframe

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    # --------------------------------------------------
    # ENCODE CATEGORICAL DATA
    # --------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "gender",
            "smoking_history"
        ],
        drop_first=True
    )

    # --------------------------------------------------
    # MATCH TRAINING FEATURES
    # --------------------------------------------------

    # IMPORTANT:
    # These columns should match the columns used
    # while training the ANN.

    training_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Male",
        "gender_Other",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current"
    ]

    input_data = input_data.reindex(
        columns=training_columns,
        fill_value=0
    )

    # --------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )

    # --------------------------------------------------
    # ANN PREDICTION
    # --------------------------------------------------

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    probability_percentage = probability * 100

    if probability >= 0.5:

        st.error(
            f"⚠️ Diabetes Detected\n\n"
            f"Probability: {probability_percentage:.2f}%"
        )

    else:

        st.success(
            f"✅ No Diabetes Detected\n\n"
            f"Probability: {probability_percentage:.2f}%"
        )

    # --------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------

    st.progress(
        float(probability)
    )

    st.write(
        f"**Diabetes Probability:** "
        f"{probability_percentage:.2f}%"
    )

    st.write(
        f"**No Diabetes Probability:** "
        f"{100 - probability_percentage:.2f}%"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Powered by Artificial Neural Network (ANN) • "
    "Sigmoid Activation • TensorFlow • Streamlit"
)