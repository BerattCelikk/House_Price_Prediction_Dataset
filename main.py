import pandas as pd #veri manipulasyonu
import numpy as np #matematiksel işemler
from sklearn.model_selection import train_test_split #modeli eğitmek için train test ayırıcaz
from sklearn.ensemble import GradientBoostingRegressor #bir sürü küçük karar ağacı var katlanarak doğruluk artıyor
from sklearn.metrics import r2_score
import pickle
import logging # log a alıyor nerede hata var vs. anlıyorsun

#bu kısımda log bir olayın ne zaman olduğunu gösteriyor.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_proptech_model():
    try:
        logging.info("Starting Training")
        
        # Veri setini yükledik
        df = pd.read_csv('Housing.csv')
        
        # --- 1. FEATURE ENGINEERING (Veri Zenginleştirme) ---
        
        # 'yes' olanları 1'e, 'no' olanları 0'a çeviriyoruz.
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)
            
        # Eşya durumunu 0, 1, 2 şeklinde sıralı sayılara dönüştürüyoruz.
        status_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        df['furnishingstatus'] = df['furnishingstatus'].map(status_map)
        
        # Yeni Özellikler Türetme
        
        # Evin ne kadar "lüks" olduğunu ölçen özel bir skor (Klima + Otopark + Eşya vb.)
        df['luxury_score'] = df[binary_cols].sum(axis=1) + df['furnishingstatus'] + df['parking']
        
        # Toplam oda/alan sayısını hesaplayan bir indeks.
        df['room_index'] = df['bedrooms'] + df['bathrooms'] + df['stories'] + df['guestroom']
        
        #logaritmasını alarak alanların normalization yapıyoruz bilhassa . 
        # Küçük sayıları az küçültürken , büyük sayıları çok küçültür.
        df['log_area'] = np.log1p(df['area']) 
        
        
        # Target Değişken Dönüşümü
        # Fiyatın logaritmasını alarak modelin %60 yerine %90+ doğrulukla öğrenmesini sağlıyoruz.
        df['log_price'] = np.log1p(df['price'])
        
        #Dashboarddaki ortalamyı hesaplıyoruz .
        stats = {
            'avg_price': df['price'].mean(),
            'max_price': df['price'].max(),
            'avg_m2_price': df['price'].mean() / df['area'].mean()
        }
        
        # Orijinal 'price' ve 'area' sütunlarını atıp, logaritmik hallerini kullanıyoruz.
        X = df.drop(['price', 'log_price', 'area'], axis=1)
        y = df['log_price']
        
        #veriyi %80 ini eğit , 20 sini test etmek için kullanıyoruz.
        #random_state = 42 => bu verileri random seçiyor 42 nin olayı her bir sayı bir random tutuyor yani o sayıyı girince yine aynı random sayılar geliyor .
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Training Gradient Boosting Regressor")
        
        # Modeli veriyi "ezberlemeden öğrenmesi" için özel parametrelerle kuruyoruz.
        model = GradientBoostingRegressor(
            n_estimators=2000,    # 2000 karar ağacı kullan.
            learning_rate=0.01,   # yavaş ama dikkatlı öğrenecek 0.1 ile 1 e yaklaşınca tam tersi oluyor 
            max_depth=5,          # ağaç 5 depth dallanıyor 
            subsample=0.7,        # Verinin %70 ini öğren.Overfitting i engellemek için(aşırı öğrenme)
            random_state=42 
        )
        model.fit(X_train, y_train) #modeli eğitiyoruz.
        
        #Değerlendirme
        
        # Logaritmik tahmini tekrar gerçek fiyata çeviriyoruz (Anti-log).
        #exp = log'u geri alıyor
        #m1 = +1 yapmıştık başta onun için -1 ekliyor
        y_train_pred = np.expm1(model.predict(X_train)) #predict yapıyor 
        y_train_real = np.expm1(y_train) 
        
        #r2 score ile accurate score ölçük ama mape falan da kullanılabilir.en yaygını bu .
        train_acc = r2_score(y_train_real, y_train_pred)
        
        logging.info(f"Training Accuracy: {train_acc*100:.2f}%")
        
        # Modeli ve gerekli dosyaları kaydediyoruz.
        pickle.dump(model, open('housing_model.pkl', 'wb')) #gradientboostregressor modeli 
        pickle.dump(stats, open('housing_stats.pkl', 'wb')) #avg price , max price hesapladık ya buraya gidiyor 
        pickle.dump(X.columns.tolist(), open('feature_names.pkl', 'wb')) #kullanılan sütunlar burada
        
        logging.info("Model Saved Successfully")

    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    train_proptech_model()