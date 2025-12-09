import streamlit as st
import pandas as pd
import joblib
import pickle
# -----------------------------
# Load model + preprocessing
# -----------------------------
model = joblib.load("/content/loan_model.pkl")
encoder = joblib.load("/content/encoder.pkl")
cat_imputer = joblib.load("/content/cat_imputer.pkl")
num_imputer = joblib.load("/content/num_imputer.pkl")

# Columns used during training — MUST MATCH EXACTLY
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History']

cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area']

# -----------------------------
# UI
# -----------------------------
st.title("Loan Eligibility Prediction System")
st.write("Enter applicant details to check loan eligibility:")

# Numeric inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Co-applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (Months)", min_value=0)

# Missing numeric feature
Credit_History = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])

# Categorical inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Eligibility"):
    
    # Create raw input row EXACTLY like training
    input_df = pd.DataFrame([{
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'Property_Area': Property_Area
    }])

    # 1️⃣ Numeric imputation
    df_num = pd.DataFrame(
        num_imputer.transform(input_df[num_cols]),
        columns=num_cols
    )

    # 2️⃣ Categorical imputation
    df_cat = pd.DataFrame(
        cat_imputer.transform(input_df[cat_cols]),
        columns=cat_cols
    )

    # 3️⃣ One-hot encoding
    df_cat_enc = pd.DataFrame(
        encoder.transform(df_cat),
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # 4️⃣ Final processed input
    final_input = pd.concat([df_num, df_cat_enc], axis=1)

    # 5️⃣ Prediction
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]

    if pred == 1:
        st.success(f"Loan Approved ✅ (Probability: {prob:.2f})")
    else:
        st.error(f"Loan Not Approved ❌ (Probability: {prob:.2f})")
    import shap
    import matplotlib.pyplot as plt
    
    st.markdown("---")
    st.subheader("📌 Prediction Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_input)

        # CASE 1: shap_values is list → binary classification
        if isinstance(shap_values, list):
            shap_single = shap_values[1][0]
            base_value = explainer.expected_value[1]

        # CASE 2: shape (1, 10, 2)
        elif shap_values.ndim == 3 and shap_values.shape[2] == 2:
            shap_single = shap_values[0, :, 1]
            base_value = explainer.expected_value[0]

        # CASE 3: already matrix
        else:
            shap_single = shap_values[0]
            base_value = explainer.expected_value
        shap_single = shap_single * -1
        shap_exp = shap.Explanation(
            values=shap_single,
            base_values=base_value,
            data=final_input.iloc[0],
            feature_names=final_input.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP could not generate plots.")
        st.code(str(e))

