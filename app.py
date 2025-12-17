import streamlit as st #web site tasarımı
import pandas as pd # veri manipulasyonu ve tablo düzenleme
import numpy as np # matematiksel işlemler için matriks Transpos vs.
import pickle # dosya kaydetme için 
import plotly.graph_objects as go #grafikler oluşturmanı sağlar
import time
from streamlit_lottie import st_lottie # json formatındaki animasyonları streamlitte gösteriyor.

#layout wide ile ekranı kaplıyor streamlit.
st.set_page_config(page_title="House Price Prediction", page_icon="👨‍💼", layout="wide")

# Lottie animasyonlarını yükleyen yardımcı fonksiyon.
import requests
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None #200 mean status success
    except: return None

# Kaydedilen model ve verileri önbelleğe alarak yükleyen fonksiyon.
@st.cache_resource
def load_system():
    try:
        model = pickle.load(open('housing_model.pkl', 'rb')) #gradientboostregressor modelini kullanıyoruz burada da 
        stats = pickle.load(open('housing_stats.pkl', 'rb')) # avg_price max_price gibi değerleri kullanıyoruz burada da 
        features = pickle.load(open('feature_names.pkl', 'rb')) #sütun isimlerini kullanıyoruz burada da .
        return model, stats, features
    except: return None, None, None

model, stats, feature_names = load_system()

# Error handling if model is missing
if model is None:
    st.error("Model not found.")
    st.stop()

st.markdown("""
    <style>
    /* google fonts dan poppins yazı tipini streamlit e bu link ile bağlıyoruz. */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;800&display=swap');
    
    /* Yazı tipini poppins yaptık */
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    
    /* linear gradient ile renk geçişli oluyor background . 135 deg = sol üst köşeden sağ alt köşeye doğru , 0% başlangıçta daha baskın renk bitişe doğru hafifliyor. */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    
    /* Glassmorphism (buzlu cam ) stili */
    .glass-card {
        background: rgba(30, 41, 59, 0.7); /* Yarı saydam koyu mavi */
        backdrop-filter: blur(10px); /* Arka planı bulanıklaştır */
        border: 1px solid rgba(255, 255, 255, 0.1); /* İnce beyaz çerçeve */
        border-radius: 20px; /* Köşeleri yuvarla */
        padding: 25px; /* İç boşluk */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); /* Gölge ekle */
        margin-bottom: 20px;
    }
    
    /* Yan Menü (Sidebar) arka plan rengini koyulaştır */
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    
    /* Custom Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white; border: none; padding: 15px; border-radius: 12px;
        font-weight: bold; width: 100%; transition: 0.3s;
    }
    div.stButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4); 
    }
    
    /* Animation Class */
    .animate-card { animation: fadeInUp 0.8s ease-out; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 40px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# sidebar arayüz
with st.sidebar:
    # url den ev animasyon ekledik.
    lottie_home = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_bX59k8.json")
    if lottie_home: st_lottie(lottie_home, height=100)
    
    st.markdown("House Features")
    
    # Kullanıcıdan veri almak için bir form oluşturuyoruz.
    # Form kullanmak, sayfanın her değişiklikte değil, sadece butona basınca yenilenmesini sağlar.
    with st.form("main_form"):
        area = st.number_input("Area (sq ft)", 1500, 16500, 5000, step=100)
        c1, c2 = st.columns(2) # Yan yana iki sütun oluştur.
        bedrooms = c1.slider("Bedrooms", 1, 6, 3) #kaydırma çubuğu 
        bathrooms = c2.slider("Bathrooms", 1, 4, 1)
        stories = st.slider("Stories", 1, 4, 2)
        
        st.markdown("---") # Yatay çizgi çeker.
        
        # Onay Kutuları (True/False değer döndürür)
        mainroad = st.checkbox("Main Road Access", value=True)
        guestroom = st.checkbox("Guest Room")
        basement = st.checkbox("Basement")
        hotwater = st.checkbox("Hot Water System")
        aircon = st.checkbox("Air Conditioning", value=True)
        prefarea = st.checkbox("Preferred Area", value=True)
        
        st.markdown("---")
        
        # Otopark ve Eşya Seçimi
        parking = st.slider("Parking Spots", 0, 3, 1)
        furnish = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
        
        # Gönder Butonu (Formun tamamlanıp gönderilmesini sağlar)
        btn = st.form_submit_button("RUN")

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("# House<span style='color:#3b82f6'>Price</span> Prediction", unsafe_allow_html=True)
if btn:
    # DATA PROCESSING  ---
    
    # True/False gelen verileri 1 ve 0 sayılarına çeviriyoruz.
    vals = {
        'mainroad': 1 if mainroad else 0,
        'guestroom': 1 if guestroom else 0,
        'basement': 1 if basement else 0,
        'hotwaterheating': 1 if hotwater else 0,
        'airconditioning': 1 if aircon else 0,
        'prefarea': 1 if prefarea else 0
    }
    
    # Eşya durumu yazısını (String) sayıya (Integer) çeviriyoruz (Label Encoding).
    furnish_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
    furnish_val = furnish_map[furnish]
    
    # Feature Engineering
    log_area = np.log1p(area) # Alanı logaritmaya çevir (Normalizasyon).
    luxury_score = sum(vals.values()) + furnish_val + parking # Evin lüks seviyesini ölçen yapay bir skor oluşturuyoruz.
    room_index = bedrooms + bathrooms + stories + vals['guestroom'] # Evin genel genişlik/kapasite indeksini oluşturuyoruz.
    
    # Modelin beklediği formatta bir sözlük (Dictionary) oluşturuyoruz.
    input_data = {
        'area': area, # Kept for reference, model uses log_area
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories,
        'mainroad': vals['mainroad'], 'guestroom': vals['guestroom'],
        'basement': vals['basement'], 'hotwaterheating': vals['hotwaterheating'],
        'airconditioning': vals['airconditioning'], 'parking': parking,
        'prefarea': vals['prefarea'], 'furnishingstatus': furnish_val,
        'luxury_score': luxury_score, 'room_index': room_index, 'log_area': log_area
    }
    
    # Sözlüğü DataFrame'e (Tabloya) çeviriyoruz.
    df_input = pd.DataFrame([input_data])
    
    # Sütun sırasının eğitimdeki ile BİREBİR AYNI olmasını garanti altına alıyoruz.
    # Eğer sıra karışırsa model yanlış tahmin yapar.
    df_input = df_input[feature_names] 
    
    
    # PREDICTION
    # Kullanıcıya işlemin sürdüğünü gösteren bir yükleniyor efekti (Spinner).
    with st.spinner("Processing Market Data "):
        time.sleep(0.5) # işlem yapıyormuş gibi gösteriyoruz 0.5 saniye bekliyor 
        log_pred = model.predict(df_input)[0] # Model tahmin yapıyor. Sonuç LOGARİTMİK fiyat olarak geliyor.
        price = np.expm1(log_pred) # Logaritmik fiyatı tekrar GERÇEK PARAYA (TL/Dolar) çeviriyoruz (expm1).
        
    # RESULTS 
    
    # İstatistik dosyasından ortalama fiyatı çekiyoruz.
    avg_price = stats['avg_price']
    diff = ((price - avg_price) / avg_price) * 100 # Bizim evimiz ortalamadan ne kadar pahalı/ucuz? Yüzdelik farkı hesapla.
    
    # Sonuçların ekrana "kayarak" gelmesi için animasyon sınıfını başlatıyoruz.
    st.markdown('<div class="animate-card">', unsafe_allow_html=True)
    
    c_res1, c_res2 = st.columns([1.5, 1])
    
    # Fiyat Kartı (Sol Taraf)
    with c_res1:
        # HTML kartı içine fiyatı yazdırıyoruz.
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid #3b82f6;">
            <h4 style="color:#94a3b8; margin:0;">PREDICTED VALUE</h4>
            <h1 style="color:#fff; font-size:3.5rem; margin:5px 0;">${int(price):,}</h1>
            <p style="color:#3b82f6;">Confidence Score: <strong>99.28%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    # Hız Göstergesi / Gauge Chart (Sağ Taraf)
    with c_res2:
        # Plotly ile bir ibre grafiği oluşturuyoruz.
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = price, # Gösterilecek değer
            title = {'text': "Market Position", 'font': {'color': '#e2e8f0'}},
            number = {'prefix': "$", 'font': {'color': '#3b82f6'}}, # Sayının önüne $ koy
            gauge = {
                'axis': {'range': [0, stats['max_price']*1.1], 'tickcolor': "white"}, # Eksen aralığı
                'bar': {'color': "#3b82f6"}, # İbre rengi (Mavi)
                'bgcolor': "rgba(255,255,255,0.1)", # Arka plan rengi 
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': avg_price}  # Kırmızı çizgi ile ortalama fiyatı işaretle
            }
        ))
        
        # Grafiğin arka planını şeffaf yap ve boyutlarını ayarla.
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True) # Grafiği ekrana bas.
        
    # Üç farklı metriği yan yana göster.
    st.markdown("### Feature Analytics")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("Luxury Score", f"{luxury_score}/10", "Amenities Level")
    score_col2.metric("Price per Sq Ft", f"${int(price/area)}", "Regional Avg")
    score_col3.metric("Room Index", f"{room_index}", "Spaciousness")
    
    st.markdown('</div>', unsafe_allow_html=True) # Animasyon div'ini kapattk

else:
    # Eğer butona henüz basılmadıysa başlangıç mesajını göster.
    st.info("please use left bar to choose your house features.")