import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Regression", layout="wide")

st.title("Regression - Predictia scorului la examen")

st.write("""
Aceasta pagina prezinta problema de regresie pentru predictia scorului final obtinut de studenti,
folosind factori educationali, sociali si personali.
""")

df = pd.read_csv("Datasets/StudentPerformanceFactors.csv")
df = df.sample(n=2000, random_state=42)

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Grafice EDA")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df, x="Exam_Score", kde=True, ax=ax1)
    ax1.set_title("Distributia variabilei tinta Exam_Score")
    st.pyplot(fig1)

with col2:
    numeric_df = df.select_dtypes(include="number")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax2)
    ax2.set_title("Matricea de corelatie")
    st.pyplot(fig2)

st.subheader("Top 5 modele analizate")

regression_results = pd.DataFrame({
    "Model": ["EBM", "SVR", "CatBoost", "XGBoost", "Linear Regression"],
    "Metric principal": ["RMSE / R2", "RMSE / R2", "RMSE / R2", "RMSE / R2", "RMSE / R2"],
    "Observatie": [
        "cel mai bun rezultat general",
        "performanta buna dupa optimizare",
        "model performant si interpretabil",
        "model bun pe relatii complexe",
        "model simplu si usor de interpretat"
    ]
})

st.dataframe(regression_results)

models = {
    "EBM": joblib.load("saved_models/regression_ebm.pkl"),
    "SVR": joblib.load("saved_models/regression_svr.pkl"),
    "CatBoost": joblib.load("saved_models/regression_catboost.pkl"),
    "XGBoost": joblib.load("saved_models/regression_xgboost.pkl"),
    "Linear Regression": joblib.load("saved_models/regression_linear.pkl")
}

scaler = joblib.load("saved_models/regression_scaler.pkl")

regression_columns = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

st.subheader("Selectare model")

selected_model_name = st.selectbox("Alege modelul:", list(models.keys()))
model = models[selected_model_name]

st.subheader("Introducere date pentru predictie")

Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=50, value=20)
Attendance = st.number_input("Attendance", min_value=0, max_value=100, value=80)
Sleep_Hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
Previous_Scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
Tutoring_Sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1)
Physical_Activity = st.number_input("Physical Activity", min_value=0, max_value=10, value=3)

input_data = pd.DataFrame(columns=regression_columns)
input_data.loc[0] = 0

input_data["Hours_Studied"] = Hours_Studied
input_data["Attendance"] = Attendance
input_data["Sleep_Hours"] = Sleep_Hours
input_data["Previous_Scores"] = Previous_Scores
input_data["Tutoring_Sessions"] = Tutoring_Sessions
input_data["Physical_Activity"] = Physical_Activity

# valori default pentru coloanele categorice codificate in notebook
input_data["Parental_Involvement"] = 1
input_data["Access_to_Resources"] = 1
input_data["Extracurricular_Activities"] = 1
input_data["Motivation_Level"] = 1
input_data["Internet_Access"] = 1
input_data["Family_Income"] = 1
input_data["Teacher_Quality"] = 1
input_data["School_Type"] = 1
input_data["Peer_Influence"] = 1
input_data["Learning_Disabilities"] = 0
input_data["Parental_Education_Level"] = 1
input_data["Distance_from_Home"] = 1
input_data["Gender"] = 1

input_data = input_data[regression_columns]

if st.button("Prezice scorul"):
    if selected_model_name in ["SVR", "Linear Regression"]:
        input_for_prediction = scaler.transform(input_data)
    else:
        input_for_prediction = input_data

    prediction = model.predict(input_for_prediction)[0]

    st.success(f"Scorul prezis la examen este: {prediction:.2f}")

st.subheader("Hiperparametri model selectat")
st.write(model.get_params())

st.subheader("Learning Curve")
st.write("""
Curbele de invatare au fost analizate in notebook pentru primele 5 modele.
Acestea au aratat comportamentul modelelor in raport cu overfitting-ul si generalizarea.
""")

st.subheader("SHAP")
st.write("""
Analiza SHAP a fost realizata pentru EBM, SVR si CatBoost.

Concluzie principala:
- Attendance este cea mai importanta caracteristica
- Hours_Studied influenteaza puternic scorul prezis
- Previous_Scores este important mai ales pentru SVR si CatBoost
- modelele folosesc factori educationali logici pentru predictie
""")