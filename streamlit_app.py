import joblib
import pandas as pd
import streamlit as st

bundle = joblib.load("penguin_model.joblib")
pipe, cols, classes = bundle["pipeline"], bundle["feature_cols"], bundle["classes"]

st.title("Pygoscelis Penguin Species Identifier")
st.write("Enter morphometric measurements to predict Adelie, Chinstrap or Gentoo.")

bill_len = st.number_input("Bill length (mm)", value=45.0)
bill_depth = st.number_input("Bill depth (mm)", value=17.0)
flipper = st.number_input("Flipper length (mm)", value=200.0)
mass = st.number_input("Body mass (g)", value=4000.0)
island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
sex = st.radio("Sex", ["Female", "Male"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "bill_length_mm": bill_len, "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper, "body_mass_g": mass,
        "island_Dream": int(island == "Dream"),
        "island_Torgersen": int(island == "Torgersen"),
        "sex_MALE": int(sex == "Male")}])[cols]
    probs = pipe.predict_proba(row)[0]
    st.success(f"Predicted species: {classes[probs.argmax()]}")
    st.bar_chart(pd.Series(probs, index=classes))