# Ali İhsan Çetin - AI Duygu Analizi Projesi (Gemini AI)

Bu proje, Google Gemini 2.0 Flash Lite modelini kullanarak metinlerin duygu analizini (Pozitif, Negatif, Nötr) gerçekleştiren modern bir web uygulamasıdır.

## 🚀 Özellikler
- **Doğrudan Gemini Entegrasyonu:** Herhangi bir harici backend sunucusuna ihtiyaç duymadan doğrudan Streamlit üzerinden çalışır.
- **Akıllı Fallback Sistemi:** API kotası dolsa veya internet kesilse bile yerel anahtar kelime analizi ile kesintisiz hizmet verir.
- **Streamlit Frontend:** Modern ve etkileşimli kullanıcı arayüzü.
- **Duygu Dağılımı:** Plotly ile görselleştirilmiş analiz geçmişi.

## 🛠️ Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/allyyy06/nlp-proje.git
   cd nlp-proje
   ```

2. `.env` dosyasını oluşturun ve API anahtarınızı ekleyin:
   ```env
   GEMINI_API_KEY=Sizin_API_Anahtariniz
   ```

3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Uygulamayı Çalıştırma

Uygulamayı başlatmak için terminale şu komutu yazın:
```bash
streamlit run streamlit_app.py
```

Uygulamanız şu anki yapılandırmada varsayılan olarak **http://localhost:8505** adresinde yayına girecektir.

Geliştiren: ALİ İHSAN ÇETİN
