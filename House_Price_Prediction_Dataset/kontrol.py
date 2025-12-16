import pandas as pd

try:
    # CSV dosyasını oku
    df = pd.read_csv('House_Price_Prediction_Dataset.csv')
    
    print("\n" + "="*40)
    print("📂 CSV DOSYASINDAKİ SÜTUN İSİMLERİ:")
    print("="*40)
    
    # Sütunları listele
    for col in df.columns:
        print(f"- {col}")
        
    print("="*40 + "\n")
    
    print("İlk 3 satır örneği:")
    print(df.head(3))

except FileNotFoundError:
    print("❌ HATA: 'House_Price_Prediction_Dataset.csv' dosyası bulunamadı.")