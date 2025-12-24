import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from collections import deque

# ================= CONFIG =================
st.set_page_config(
    page_title="FruitNet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ESTILO CLARO =================
st.markdown("""
<style>
body {
    background-color: #f4f6fa;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.header {
    text-align: center;
    margin-bottom: 1.5rem;
}
.header h1 {
    color: #1f2937;
    font-weight: 700;
}
.header p {
    color: #6b7280;
}
.badge {
    background-color: #e0f2fe;
    color: #0369a1;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    display: inline-block;
}
.state-good {
    color: #15803d;
    font-weight: bold;
}
.state-bad {
    color: #b91c1c;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 📊 Estado del Modelo")
    st.success("Modelo cargado")
    st.caption("CNN con Transfer Learning")

    st.markdown("### 📦 Clases")
    st.markdown("- Manzana")
    st.markdown("- Banana")
    st.markdown("- Naranja")

    st.markdown("---")
    st.markdown("### 📝 Instrucciones")
    st.caption("1. Coloque la fruta al centro")
    st.caption("2. Use buena iluminación")
    st.caption("3. Fondo claro recomendado")

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h1>🍎Clasificación de Frutas Frescas o Malas</h1>
    <p>Clasificación automática de frutas por tipo y estado en tiempo real</p>
</div>
""", unsafe_allow_html=True)

# ================= ESTADOS =================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="card"><b>Tipo</b><br><span class="badge">Auto</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><b>Estado</b><br><span class="badge">Auto</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><b>Conf. Tipo</b><br><span class="badge">%</span></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="card"><b>Conf. Estado</b><br><span class="badge">%</span></div>', unsafe_allow_html=True)

# ================= MODELO =================
MODEL_PATH = "fruit_classifier_v3_transfer_learning.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
input_h, input_w = model.input_shape[1], model.input_shape[2]

CLASS_NAMES = [
    "Manzana fresca", "Banana fresca", "Naranja fresca",
    "Manzana podrida", "Banana podrida", "Naranja podrida"
]

MIN_FRUIT_AREA = 5000
EMA_ALPHA = 0.6
CONFIRM_FRAMES = 3

# ================= VIDEO =================
class FruitVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.ema_pred = None
        self.class_counter = deque(maxlen=CONFIRM_FRAMES)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        roi_size = 300
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2, y2 = x1 + roi_size, y1 + roi_size

        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, sat, _ = cv2.split(hsv)
        _, thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large = [c for c in contours if cv2.contourArea(c) > MIN_FRUIT_AREA]

        clase, estado, prob = "No clara", "N/A", 0.0
        color = (255, 165, 0)

        if len(large) == 1:
            c = max(large, key=cv2.contourArea)
            x, y, w2, h2 = cv2.boundingRect(c)
            crop = roi[y:y+h2, x:x+w2]

            if crop.size > 0:
                img_in = cv2.resize(crop, (input_w, input_h))
                img_in = tf.expand_dims(img_in, axis=0)
                pred = model.predict(img_in, verbose=0)[0]

                self.ema_pred = pred if self.ema_pred is None else EMA_ALPHA * pred + (1 - EMA_ALPHA) * self.ema_pred
                idx = int(np.argmax(self.ema_pred))
                clase = CLASS_NAMES[idx]
                estado = "Buena" if "fresca" in clase else "Podrida"
                color = (0, 180, 90) if estado == "Buena" else (200, 50, 50)
                prob = float(self.ema_pred[idx])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{clase} ({prob*100:.1f}%)", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Estado: {estado}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# ================= CÁMARA =================
st.markdown('<div class="card"><h3>📷 Cámara en Vivo</h3></div>', unsafe_allow_html=True)

webrtc_streamer(
    key="fruitnet",
    video_processor_factory=FruitVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ================= GUIA =================
st.markdown("## 🍏 Guía de Uso con Frutas Reales")

g1, g2 = st.columns(2)
with g1:
    st.markdown('<div class="card">🍌 <b>Plátano</b><br>Fresco: Amarillo brillante<br>Malogrado: manchas oscuras</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">🍎 <b>Manzana</b><br>Fresca: Firme y lisa<br>Malograda: Arrugada</div>', unsafe_allow_html=True)

with g2:
    st.markdown('<div class="card">🍊 <b>Naranja</b><br>Fresco: Amarillo intenso<br>Malogrado: Manchas oscuras</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">🍊 <b>Otros </b><br>Fresca: Lisa <br>Malograda: Manchas oscuros</div>', unsafe_allow_html=True)

#streamlit run app.py  // para ejecutar//