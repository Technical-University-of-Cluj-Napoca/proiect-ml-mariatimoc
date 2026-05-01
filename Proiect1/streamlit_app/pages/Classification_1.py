import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Classification", layout="wide")

st.title("Classification - Predictia nivelului de stres")

st.write("""
Aceasta pagina prezinta problema de clasificare pentru predictia nivelului de stres
pe baza somnului, activitatii fizice, timpului petrecut pe ecran si social media.
""")

df = pd.read_csv("Datasets/social_media_sleep_stress_productivity_11000.csv")
df = df.sample(n=2000, random_state=42)

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Grafice EDA")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="stress_level", ax=ax1)
    ax1.set_title("Distributia claselor stress_level")
    st.pyplot(fig1)

with col2:
    numeric_df = df.select_dtypes(include="number")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax2)
    ax2.set_title("Matricea de corelatie")
    st.pyplot(fig2)

st.subheader("Top 5 modele analizate")

classification_results = pd.DataFrame({
    "Model": ["CatBoost", "Random Forest", "XGBoost", "SVM", "Gaussian Naive Bayes"],
    "Metric principal": ["F1 Score", "F1 Score", "F1 Score", "F1 Score", "F1 Score"],
    "Observatie": [
        "model performant dupa optimizare",
        "model stabil, dar cu risc de overfitting",
        "model performant pe date complexe",
        "model echilibrat",
        "model simplu si rapid"
    ]
})

st.dataframe(classification_results)

models = {
    "CatBoost": joblib.load("saved_models/classification_catboost.pkl"),
    "XGBoost": joblib.load("saved_models/classification_xgboost.pkl"),
    "Random Forest": joblib.load("saved_models/classification_random_forest.pkl"),
    "SVM": joblib.load("saved_models/classification_svm.pkl"),
    "Gaussian Naive Bayes": joblib.load("saved_models/classification_nb.pkl")
}

scaler = joblib.load("saved_models/classification_scaler.pkl")
target_encoder = joblib.load("saved_models/classification_target_encoder.pkl")
platform_encoder = joblib.load("saved_models/classification_platform_encoder.pkl")

classification_columns = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "sleep_hours",
    "exercise_minutes",
    "study_work_hours",
    "productivity_score",
    "platform"
]

st.subheader("Selectare model")

selected_model_name = st.selectbox("Alege modelul:", list(models.keys()))
model = models[selected_model_name]

st.subheader("Introducere date pentru predictie")

age = st.number_input("Age", min_value=10, max_value=80, value=25)
daily_screen_time_hours = st.number_input("Daily screen time hours", min_value=0.0, max_value=24.0, value=6.0)
social_media_hours = st.number_input("Social media hours", min_value=0.0, max_value=24.0, value=3.0)
sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=15.0, value=7.0)
exercise_minutes = st.number_input("Exercise minutes", min_value=0, max_value=300, value=30)
study_work_hours = st.number_input("Study/work hours", min_value=0.0, max_value=24.0, value=6.0)
productivity_score = st.number_input("Productivity score", min_value=0.0, max_value=100.0, value=70.0)

platform = st.selectbox("Platform", list(platform_encoder.classes_))
platform_encoded = platform_encoder.transform([platform])[0]

input_data = pd.DataFrame(columns=classification_columns)
input_data.loc[0] = 0

input_data["age"] = age
input_data["daily_screen_time_hours"] = daily_screen_time_hours
input_data["social_media_hours"] = social_media_hours
input_data["sleep_hours"] = sleep_hours
input_data["exercise_minutes"] = exercise_minutes
input_data["study_work_hours"] = study_work_hours
input_data["productivity_score"] = productivity_score
input_data["platform"] = platform_encoded

if st.button("Prezice nivelul de stres"):
    if selected_model_name in ["SVM", "Gaussian Naive Bayes"]:
        input_for_prediction = scaler.transform(input_data)
    else:
        input_for_prediction = input_data

    prediction = model.predict(input_for_prediction)

    prediction_value = prediction[0]

    if isinstance(prediction_value, str):
        prediction_label = prediction_value

    elif isinstance(prediction_value, (list, tuple, np.ndarray)):
        prediction_label = target_encoder.inverse_transform([np.argmax(prediction_value)])[0]

    else:
        prediction_label = target_encoder.inverse_transform([int(prediction_value)])[0]
    st.success(f"Nivelul de stres prezis este: {prediction_label}")

st.subheader("Hiperparametri model selectat")
st.write(model.get_params())

st.subheader("Learning Curve")
st.write("""
Curbele de invatare au fost analizate in notebook pentru cele mai bune 5 modele.
Acestea au fost folosite pentru observarea overfitting-ului si a capacitatii de generalizare.
""")

st.subheader("SHAP")
st.write("""
Analiza SHAP a fost realizata pentru CatBoost, XGBoost si Random Forest.

Concluzie principala:
- sleep_hours este cea mai importanta caracteristica
- exercise_minutes si study_work_hours apar frecvent printre factorii importanti
- factorii digitali, precum social_media_hours si daily_screen_time_hours, influenteaza predictia in functie de model
""")