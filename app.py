import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configuração da página Streamlit
st.set_page_config(page_title="Detector de Objetos IA", layout="centered")

@st.cache_resource
def load_model():
    # Carrega o modelo Nano (leve e otimizado para CPU)
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo YOLO: {e}")
    st.stop()

st.title("🔍 Detecção de Objetos em Tempo Real")
st.write("Faça o upload de uma imagem para que a IA identifique os objetos presentes.")

# Componente de Upload
uploaded_file = st.file_uploader("Escolha uma imagem (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Conversão do arquivo de upload para imagem PIL e depois OpenCV
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # YOLO espera RGB (padrão do PIL), mas se a imagem tiver canal Alpha (RGBA), converte para RGB
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    st.subheader("Processando Imagem...")
    
    # Inferência
    results = model(img_array)
    
    # Renderização dos resultados na imagem
    res_plotted = results[0].plot()
    
    # Exibição dos resultados no Streamlit
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Imagem Original", use_container_width=True)
        
    with col2:
        st.image(res_plotted, caption="Objetos Detectados", use_container_width=True)
        
    # Exibir métricas e classes encontradas abaixo
    st.subheader("📋 Objetos Identificados:")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.info("Nenhum objeto conhecido foi detectado na imagem.")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            st.write(f"- **{label.capitalize()}** (Confiança: {conf:.2%})")