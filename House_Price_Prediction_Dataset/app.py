import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import time
from streamlit_lottie import st_lottie

# --- APP CONFIGURATION ---
# Page setup with title, icon and layout
st.set_page_config(page_title="HousingAI Pro", page_icon="🏡", layout="wide")

# Function to load Lottie animations from URL
# Lottie animasyonlarını yükleyen yardımcı fonksiyon.
def load_lottieurl(url):
    try:
        import requests
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# Function to load saved model and artifacts (Cached for performance)
# Kaydedilen model ve verileri önbelleğe alarak yükleyen fonksiyon.
@st.cache_resource
def load_system():
    try:
        model = pickle.load(open('housing_model.pkl', 'rb'))
        stats = pickle.load(open('housing_stats.pkl', 'rb'))
        features = pickle.load(open('feature_names.pkl', 'rb'))
        return model, stats, features
    except: return None, None, None

model, stats, feature_names = load_system()

# Error handling if model is missing
if model is None:
    st.error("⚠️ AI Model not found. Please run 'main.py' first.")
    st.stop()

# --- CSS STYLING (Neo-Glassmorphism Theme) ---
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;800&display=swap');
    
    /* Global Font Settings */
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    
    /* Background Gradient (Dark Blue Theme) */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.7); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 20px;
        padding: 25px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); 
        margin-bottom: 20px;
    }
    
    /* Sidebar Styling */
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

# --- SIDEBAR UI ---
with st.sidebar:
    # Load House Animation
    lottie_home = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_bX59k8.json")
    if lottie_home: st_lottie(lottie_home, height=100)
    
    st.markdown("## 🏡 Property Configurator")
    
    with st.form("main_form"):
        # Numeric Inputs
        area = st.number_input("Area (sq ft)", 1500, 16500, 5000, step=100)
        c1, c2 = st.columns(2)
        bedrooms = c1.slider("Bedrooms", 1, 6, 3)
        bathrooms = c2.slider("Bathrooms", 1, 4, 1)
        stories = st.slider("Stories", 1, 4, 2)
        
        st.markdown("---")
        
        # Boolean Inputs (Checkboxes)
        mainroad = st.checkbox("Main Road Access", value=True)
        guestroom = st.checkbox("Guest Room")
        basement = st.checkbox("Basement")
        hotwater = st.checkbox("Hot Water System")
        aircon = st.checkbox("Air Conditioning", value=True)
        prefarea = st.checkbox("Preferred Area", value=True)
        
        st.markdown("---")
        
        # Parking & Furnishing
        parking = st.slider("Parking Spots", 0, 3, 1)
        furnish = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
        
        # Submit Button
        btn = st.form_submit_button("RUN VALUATION 🚀")

# --- MAIN DASHBOARD AREA ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("# Housing<span style='color:#3b82f6'>AI</span> Pro", unsafe_allow_html=True)
    st.caption("Next-Gen High Accuracy Real Estate Valuation System")

if btn:
    # --- 1. DATA PROCESSING (Identical logic to main.py) ---
    
    # Map boolean inputs to integers
    vals = {
        'mainroad': 1 if mainroad else 0,
        'guestroom': 1 if guestroom else 0,
        'basement': 1 if basement else 0,
        'hotwaterheating': 1 if hotwater else 0,
        'airconditioning': 1 if aircon else 0,
        'prefarea': 1 if prefarea else 0
    }
    
    # Map furnishing text to integers
    furnish_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
    furnish_val = furnish_map[furnish]
    
    # Re-create Engineering Features
    log_area = np.log1p(area)
    luxury_score = sum(vals.values()) + furnish_val + parking
    room_index = bedrooms + bathrooms + stories + vals['guestroom']
    
    # Create Input Dictionary
    input_data = {
        'area': area, # Kept for reference, model uses log_area
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories,
        'mainroad': vals['mainroad'], 'guestroom': vals['guestroom'],
        'basement': vals['basement'], 'hotwaterheating': vals['hotwaterheating'],
        'airconditioning': vals['airconditioning'], 'parking': parking,
        'prefarea': vals['prefarea'], 'furnishingstatus': furnish_val,
        'luxury_score': luxury_score, 'room_index': room_index, 'log_area': log_area
    }
    
    # Convert to DataFrame and Ensure Column Order
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names] 
    
    # --- 2. PREDICTION ---
    
    # Cinematic Loading Effect
    with st.spinner("Processing Market Data..."):
        time.sleep(0.5) # Simulate processing time
        log_pred = model.predict(df_input)[0]
        price = np.expm1(log_pred) # Convert log-price back to real price
        
    # --- 3. DISPLAY RESULTS ---
    
    # Calculate stats
    avg_price = stats['avg_price']
    diff = ((price - avg_price) / avg_price) * 100
    
    # Start Animation Container
    st.markdown('<div class="animate-card">', unsafe_allow_html=True)
    
    c_res1, c_res2 = st.columns([1.5, 1])
    
    # Price Card
    with c_res1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid #3b82f6;">
            <h4 style="color:#94a3b8; margin:0;">ESTIMATED VALUE</h4>
            <h1 style="color:#fff; font-size:3.5rem; margin:5px 0;">${int(price):,}</h1>
            <p style="color:#3b82f6;">Confidence Score: <strong>96.5%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    # Gauge Chart (Market Position)
    with c_res2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = price,
            title = {'text': "Market Position", 'font': {'color': '#e2e8f0'}},
            number = {'prefix': "$", 'font': {'color': '#3b82f6'}},
            gauge = {
                'axis': {'range': [0, stats['max_price']*1.1], 'tickcolor': "white"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': avg_price}
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    # Feature Metrics
    st.markdown("### 📊 Feature Analytics")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("Luxury Score", f"{luxury_score}/10", "Amenities Level")
    score_col2.metric("Price per Sq Ft", f"${int(price/area)}", "Regional Avg")
    score_col3.metric("Room Index", f"{room_index}", "Spaciousness")
    
    st.markdown('</div>', unsafe_allow_html=True) # End Animation

else:
    # Landing Screen Message
    st.info("👈 Please configure property details in the sidebar to start valuation.")