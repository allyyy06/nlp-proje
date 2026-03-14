import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Gemini API Yapılandırması
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
else:
    model = None

# Sayfa yapılandırması
st.set_page_config(
    page_title="Duygu Analizi - Gemini AI",
    page_icon="🤖",
    layout="wide"
)

# Stil düzenlemeleri
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4F8BF9;
        color: white;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Ali İhsan Çetin - AI Duygu Analizi")
st.subheader("Yapay zeka destekli metin analiz araçları.")

# Sidebar - Geçmiş
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("📜 Analiz Geçmişi")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.info(f"**{item['text'][:30]}...**\nResult: {item['sentiment']}")
    else:
        st.write("Henüz analiz yapılmadı.")

def fallback_analysis(text):
    positive_words = ['iyi', 'güzel', 'harika', 'muhteşem', 'memnun', 'başarılı', 'seviyorum', 'teşekkür', 'mükemmel']
    negative_words = ['kötü', 'berbat', 'rezelat', 'memnun değilim', 'hayal kırıklığı', 'çalışmıyor', 'hata', 'korkunç']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {"sentiment": "Pozitif", "score": 0.7, "explanation": "Yerel analiz: Pozitif anahtar kelimeler tespit edildi. (Gemini API şu an kullanılamıyor, çevrimdışı mod)"}
    elif neg_count > pos_count:
        return {"sentiment": "Negatif", "score": 0.7, "explanation": "Yerel analiz: Negatif anahtar kelimeler tespit edildi. (Gemini API şu an kullanılamıyor, çevrimdışı mod)"}
    else:
        return {"sentiment": "Nötr", "score": 0.5, "explanation": "Yerel analiz: Belirgin bir duygu tonu saptanamadı. (Gemini API şu an kullanılamıyor, çevrimdışı mod)"}

def analyze_sentiment(text):
    if not model:
        return fallback_analysis(text)
        
    try:
        prompt = f"""
        Aşağıdaki metnin duygu analizini yap. 
        Sonucu mutlaka şu JSON formatında döndür:
        {{
            "sentiment": "Pozitif" | "Negatif" | "Nötr",
            "score": 0.0-1.0 arası güven skoru,
            "explanation": "Kısa Türkçe açıklama"
        }}
        
        Metin: {text}
        """
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # JSON temizleme (Gemini bazen markdown içinde döndürür)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(result_text)
    except Exception as e:
        return fallback_analysis(text)

# Ana Ekran
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Analiz edilecek metni buraya girin:", height=200, placeholder="Örn: Bu ürün gerçekten harika, çok memnun kaldım!")
    
    if st.button("Analiz Et"):
        if user_input.strip():
            with st.spinner("Analiz ediliyor..."):
                result = analyze_sentiment(user_input.strip())
                
                st.session_state.history.append({
                    "text": user_input,
                    "sentiment": result['sentiment'],
                    "score": result['score']
                })
                
                # Sonuç kartı
                sentiment = result['sentiment']
                color = "#d4edda" if sentiment == "Pozitif" else "#f8d7da" if sentiment == "Negatif" else "#fff3cd"
                text_color = "#155724" if sentiment == "Pozitif" else "#721c24" if sentiment == "Negatif" else "#856404"
                
                st.markdown(f"""
                    <div class="sentiment-box" style="background-color: {color}; color: {text_color}; border: 1px solid {text_color};">
                        <h1 style="margin:0;">{sentiment}</h1>
                        <p style="font-size: 1.2em;">Güven Skoru: %{result['score']*100:.1f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success(f"**Analiz Sonucu:** {result['explanation']}")
        else:
            st.warning("Lütfen bir metin girin.")

with col2:
    st.header("📊 İstatistikler")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = px.pie(df, names='sentiment', title='Duygu Dağılımı', color='sentiment',
                     color_discrete_map={'Pozitif':'green', 'Negatif':'red', 'Nötr':'gray'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Veri bekleniyor...")

st.markdown("---")
st.caption("Geliştiricisi: Ali İhsan ÇETİN | Powered by Gemini 2.0 Flash Lite")
