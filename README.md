# 🎵 Ses Benzerlik Analizi

Bu uygulama, ses dosyalarını yükleyip sınıflandıran ve benzerlik analizi yapabilen bir AI tabanlı ses analiz sistemidir.

## 🚀 Canlı Demo

Uygulamayı Streamlit Cloud'da kullanabilirsiniz: [Demo Linki](https://github.com/canpasa1940/BenzerlikDenemeV3)

## 📋 Özellikler

- **Ses Sınıflandırma**: 7 farklı perküsyon sesi (Bass, Clap, Cymbal, Hat, Kick, Rims, Snare)
- **Benzerlik Analizi**: Yüklenen sesler arasında benzerlik karşılaştırması
- **Toplu İşleme**: Birden fazla dosyayı aynı anda yükleme ve analiz
- **Görselleştirme**: PCA analizi, sınıf dağılımı ve dalga formu görselleri
- **Ses Çalar**: Yüklenen dosyaları doğrudan dinleme

## 🎯 Desteklenen Ses Sınıfları

1. **Bass** - Bas davul sesleri
2. **Clap** - El çırpma/clap sesleri
3. **Cymbal** - Zil sesleri
4. **Hat** - Hi-hat sesleri
5. **Kick** - Kick davul sesleri
6. **Rims** - Rim shot sesleri
7. **Snare** - Snare davul sesleri

## 🔧 Teknik Detaylar

- **Model**: TensorFlow/Keras tabanlı derin öğrenme modeli
- **Özellik Çıkarma**: MFCC, RMS, ZCR, Spektral özellikler
- **Benzerlik Ölçümü**: Cosine similarity ve Euclidean distance
- **Görselleştirme**: PCA ile 2D boyut azaltma

## 📁 Dosya Formatları

- Desteklenen format: **WAV**
- Maksimum dosya boyutu: 200MB
- Önerilen örnekleme oranı: 44.1kHz

## 🎵 Kullanım Alanları

- Müzik prodüktörleri için benzer davul sesi bulma
- Ses tasarımcıları için kütüphane organizasyonu
- Araştırmacılar için ses özellik analizi
- Müzisyenler için uyumlu perküsyon keşfi

## 📊 Nasıl Çalışır?

1. **Yükleme**: WAV formatında ses dosyalarınızı yükleyin
2. **Analiz**: AI modeli ses özelliklerini çıkarır ve sınıflandırır
3. **Sonuçlar**: Tahmin edilen sınıf ve güven skorunu görün
4. **Benzerlik**: Diğer yüklenmiş seslerle benzerlik analizi yapın
5. **Görselleştirme**: PCA ve grafiklerde sonuçları inceleyin

## 🛠️ Local Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/canpasa1940/BenzerlikDenemeV3.git
cd BenzerlikDenemeV3

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

**Geliştirici:** Can Paşa  
**Son Güncelleme:** 2024  
**Versiyon:** 3.0
