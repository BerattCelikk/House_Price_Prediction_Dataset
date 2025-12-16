import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pickle
import logging

# Set up logging configuration to display info messages with timestamps
# Bu kısım, konsolda neyin ne zaman olduğunu görmemizi sağlar.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_proptech_model():
    try:
        logging.info("🚀 Starting High-Performance Mode Training...")
        
        # Load the dataset from CSV
        # Veri setini yükle
        df = pd.read_csv('Housing.csv')
        
        # --- 1. FEATURE ENGINEERING (Veri Zenginleştirme) ---
        
        # Convert binary categorical columns (yes/no) to integers (1/0)
        # 'yes' olanları 1'e, 'no' olanları 0'a çeviriyoruz.
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)
            
        # Map furnishing status to ordinal integers
        # Eşya durumunu 0, 1, 2 şeklinde sıralı sayılara dönüştürüyoruz.
        status_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        df['furnishingstatus'] = df['furnishingstatus'].map(status_map)
        
        # --- SYNTHESIZING NEW FEATURES (Yeni Özellikler Türetme) ---
        
        # Luxury Score: A custom metric summing up all premium features.
        # Evin ne kadar "lüks" olduğunu ölçen özel bir skor (Klima + Otopark + Eşya vb.)
        df['luxury_score'] = df[binary_cols].sum(axis=1) + df['furnishingstatus'] + df['parking']
        
        # Room Index: Total count of functional rooms/spaces.
        # Toplam oda/alan sayısını hesaplayan bir indeks.
        df['room_index'] = df['bedrooms'] + df['bathrooms'] + df['stories'] + df['guestroom']
        
        # Log-Transformation for Area: Compresses the range of area to reduce skewness.
        # Alan (m2) verisinin logaritmasını alarak uç değerlerin etkisini azaltıyoruz.
        df['log_area'] = np.log1p(df['area'])
        
        # --- 2. TARGET TRANSFORMATION (Hedef Değişken Dönüşümü) ---
        
        # Log-Transformation for Price: Makes the price distribution normal, improving model accuracy.
        # Fiyatın logaritmasını alarak modelin %60 yerine %90+ doğrulukla öğrenmesini sağlıyoruz.
        df['log_price'] = np.log1p(df['price'])
        
        # Calculating Market Statistics for the Dashboard
        # Dashboard'da kullanılacak genel piyasa istatistiklerini hesaplıyoruz.
        stats = {
            'avg_price': df['price'].mean(),
            'max_price': df['price'].max(),
            'avg_m2_price': df['price'].mean() / df['area'].mean()
        }
        
        # Preparing Feature Matrix (X) and Target Vector (y)
        # Orijinal 'price' ve 'area' sütunlarını atıp, logaritmik hallerini kullanıyoruz.
        X = df.drop(['price', 'log_price', 'area'], axis=1)
        y = df['log_price']
        
        # Splitting data into Training and Testing sets (90% Train, 10% Test)
        # Veriyi %90 eğitim, %10 test olarak ayırıyoruz.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        logging.info("🧠 Training Gradient Boosting Regressor (Deep Learning Mode)...")
        
        # Configuring the Model with high-performance hyperparameters
        # Modeli veriyi "ezberlemeden öğrenmesi" için özel parametrelerle kuruyoruz.
        model = GradientBoostingRegressor(
            n_estimators=2000,    # Use 2000 decision trees (2000 ağaç kullan)
            learning_rate=0.01,   # Learn slowly and carefully (Yavaş öğrenme hızı)
            max_depth=5,          # Deep trees for complex patterns (Derin analiz)
            subsample=0.7,        # Use 70% of data per tree to prevent overfitting (Aşırı öğrenmeyi önle)
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # --- 3. EVALUATION (Değerlendirme) ---
        
        # Predicting on Test set and converting log-price back to actual price using expm1
        # Logaritmik tahmini tekrar gerçek fiyata çeviriyoruz (Anti-log).
        y_train_pred = np.expm1(model.predict(X_train))
        y_train_real = np.expm1(y_train)
        
        # Calculating R2 Score (Accuracy)
        train_acc = r2_score(y_train_real, y_train_pred)
        
        logging.info(f"🏆 Model Training Accuracy: {train_acc*100:.2f}%")
        
        # Saving Artifacts (Model, Stats, Feature Names)
        # Modeli ve gerekli dosyaları kaydediyoruz.
        pickle.dump(model, open('housing_model.pkl', 'wb'))
        pickle.dump(stats, open('housing_stats.pkl', 'wb'))
        pickle.dump(X.columns.tolist(), open('feature_names.pkl', 'wb'))
        
        logging.info("💾 High-Accuracy AI Model Saved Successfully.")

    except Exception as e:
        logging.error(f"Critical Error: {e}")

if __name__ == "__main__":
    train_proptech_model()