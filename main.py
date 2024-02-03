import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


konut = pd.read_csv('Data/konut.csv')

data = pd.DataFrame(konut)

########### SORU 1 /a  ###########
#Cevap:

# Bağımsız değişkenler (X) ve bağımlı değişkenler (Y) olarak ayır. # 'Fiyat' ve 'Port_no' sütunlarını çıkararak bağımsız değişkenleri alıyorum.

X = data.drop(['Fiyat', 'Port_no'], axis=1)

# 'Fiyat' ve 'Port_no' sütunlarını dataframe den çıkarıyoruz.'Fiyat' bağımlı değişken olarak alıyoruz. 'Port_no' ise kategori değişkeni olduğu için tahmin edilmesi mantıklı olmayabilir.

Y = data['Fiyat']
# Bağımsız ve bağımlı değişkenleri yazdıran kodu yaz.
print(X)
print(Y)



########### SORU 1 /b ###########
#Cevap:

# MinMaxScaler nesnesini oluştur
scaler = MinMaxScaler()

# Bağımsız değişkenleri ölçeklendir
X_normalized = scaler.fit_transform(X)

# Normalleştirilmiş bağımsız değişkenlerin DataFrame'e dönüştürülmesi
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Normalleştirilmiş bağımsız değişkenleri yazdır
print(X_normalized_df)

########### SORU 1 /c ###########
#Cevap:

#Ortalama Mutlak Hata (MAE): 0.5311643817546448
#Tahmin Edilen Değerler: [4.13164983 3.97660644 3.67657094 ... 0.17125141 0.31910524 0.51580363]

lr_model1 = LinearRegression()
lr_model1.fit(X_normalized_df, Y)

# Aynı bağımsız değişkenler üzerinde tahmini değerleri hesaplama
y_pred1 = lr_model1.predict(X_normalized_df)

# Ortalama mutlak hata (MAE) değerini hesaplama
mae1 = mean_absolute_error(Y, y_pred1)

print(f"Ortalama Mutlak Hata (MAE): {mae1}")
print(f"Tahmin Edilen Değerler: {y_pred1}")


########### SORU 1 /d ###########
#Cevap:
# öğrenci numarası: 20221500130
# rakam toplamları: 2+0+2+2+1+5+0+0+1+3+0 = 16
# test_size = 16/100 = 0.16

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.16, random_state=42)

############# SORU 1 /e ############
#Cevap:
#Test ve train Verileri Üzerinde Ortalama Mutlak Hata (MAE): 0.5367525346722181
#Test Tahmin Edilen Değerler: [0.72921491 1.7542562  2.67436509 ... 2.52703171 2.97491884 1.71792032]

# Lineer regresyon modelini oluştur ve eğitme işlemi
lr_model2 = LinearRegression()
lr_model2.fit(X_train, y_train)

y_pred2 = lr_model2.predict(X_test)
mae2 = mean_absolute_error(y_test, y_pred2)

print(f"Test ve train Verileri Üzerinde Ortalama Mutlak Hata (MAE): {mae2}")
print(f"Test Tahmin Edilen Değerler: {y_pred2}")



############# SORU 1 /f ############
#Cevap:
#            katsayi_model1 katsayi_model2
#OrtGel             6.33214       0.441671
#Bina_Yasi         0.481225       0.009482
#OrtOda          -15.139162      -0.110158
#Ort_Y            21.760216       0.647729
#Nufus            -0.141874      -0.000004
#Ort_Hane         -4.705313      -0.003569
#Enlem            -3.964568      -0.419629
#Boylam           -4.362518      -0.433181
#hata_model1       0.531164           <NA>
#hata_model2           <NA>       0.536753

# Katsayılar için DataFrame oluşturma işlemi
df_katsayilar = pd.DataFrame(lr_model1.coef_, index=X.columns, columns=['katsayi_model1'])

# katsayi_model2 sütununu lr_model2'nin katsayıları ile güncelleme işlemi
df_katsayilar['katsayi_model2'] = lr_model2.coef_

# Hata metrikleri için DataFrame oluşturma işlemi
df_hatalar = pd.DataFrame({
    'katsayi_model1': [mae1, pd.NA],
    'katsayi_model2': [pd.NA, mae2]
}, index=['hata_model1', 'hata_model2'])

# Katsayılar ve hatalar DataFrame'lerini birleştirme işlemi
df_karsilastirma = pd.concat([df_katsayilar, df_hatalar])

print(df_karsilastirma)