import streamlit as st

st.set_page_config(
    page_title="Proiect Machine Learning",
    page_icon="📊",
    layout="wide"
)

st.title("Proiect Machine Learning")
st.subheader("Analiza comparata a modelelor de machine learning")

st.write("""
Aceasta aplicatie prezinta doua probleme de machine learning:

- Clasificare: predictia nivelului de stres
- Regresie: predictia scorului final la examen
""")

st.info("Alege din meniul din stanga pagina pentru Classification sau Regression.")

st.markdown("### Student")
st.write("Maria Timoc")

st.markdown("### Continut aplicatie")
st.write("""
Aplicatia include:
- descrierea problemei si a dataset-ului
- grafice EDA relevante
- selectarea interactiva a modelelor
- introducerea valorilor pentru predictie
- afisarea rezultatului prezis
""")