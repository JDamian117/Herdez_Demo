import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("Interfaz de prueba")

st.write("Si ves esto, Streamlit funciona correctamente.")

with st.sidebar:
    st.header("Configuración")
    valor = st.slider("Selecciona un número", 0, 100, 50)
    st.write(f"Valor seleccionado: {valor}")

chat = st.chat_input("Escribe algo...")
if chat:
    st.chat_message("user").write(chat)
    st.chat_message("assistant").write(f"Recibí tu mensaje: {chat}")