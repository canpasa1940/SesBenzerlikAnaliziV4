# 🛠️ Local Kurulum Rehberi

Bu rehber, Ses Benzerlik Analizi uygulamasını kendi bilgisayarınızda çalıştırmanız için hazırlanmıştır.

## 📋 Gereksinimler

- **Python 3.11 veya 3.12** (önerilen: 3.12)
- **Git** (kod indirmek için)
- **İnternet bağlantısı** (bağımlılıkları indirmek için)

## 🔧 Kurulum Adımları

### 1. Python'u İndirin ve Kurun

**Windows:**
- [python.org](https://python.org) adresinden Python 3.12 indirin
- Kurulum sırasında "Add Python to PATH" işaretleyin

**macOS:**
```bash
# Homebrew ile
brew install python@3.12

# Veya python.org'dan indirin
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-pip python3.12-venv
```

### 2. Projeyi İndirin

```bash
# Terminal/Command Prompt'da çalıştırın:
git clone https://github.com/canpasa1940/BenzerlikDenemeV3.git
cd BenzerlikDenemeV3
```

**Git yoksa:** 
- GitHub'dan ZIP indirin: [Download ZIP](https://github.com/canpasa1940/BenzerlikDenemeV3/archive/refs/heads/main.zip)
- ZIP'i açın ve klasöre girin

### 3. Sanal Ortam Oluşturun (Önerilen)

```bash
# Sanal ortam oluştur
python -m venv ses_analizi_env

# Sanal ortamı aktifleştir
# Windows:
ses_analizi_env\Scripts\activate

# macOS/Linux:
source ses_analizi_env/bin/activate
```

### 4. Bağımlılıkları Yükleyin

```bash
# Gerekli kütüphaneleri yükle
pip install --upgrade pip
pip install -r requirements.txt
```

**Not:** Bu adım 5-10 dakika sürebilir (TensorFlow büyük bir kütüphane).

### 5. Uygulamayı Başlatın

```bash
streamlit run app.py
```

### 6. Tarayıcıda Açın

Uygulama otomatik olarak tarayıcıda açılacak. Açılmazsa:
- **URL:** `http://localhost:8501`

## 🎵 Kullanım

1. **WAV dosyası yükleyin** (perküsyon sesleri önerilir)
2. **Sınıflandırma sonuçlarını** görün
3. **Benzerlik analizi** yapın
4. **Görselleştirmeleri** inceleyin

## 🔍 Olası Sorunlar ve Çözümler

### Hata: "Python bulunamadı"
```bash
# Python yolunu kontrol edin:
python --version
# veya
python3 --version
```

### Hata: "TensorFlow yüklenemedi"
```bash
# Pip'i güncelleyin:
pip install --upgrade pip
pip install tensorflow==2.16.0
```

### Hata: "librosa yüklenemedi"
```bash
# Sistem bağımlılıklarını yükleyin:
# Windows: Visual C++ redistributable gerekebilir
# macOS: Xcode command line tools
# Linux: sudo apt install libsndfile1
```

### Hata: "Streamlit bulunamadı"
```bash
pip install streamlit
```

## 🚀 Performans İpuçları

- **RAM:** En az 4GB önerilir
- **İşlemci:** Model yükleme için güçlü işlemci avantajlı
- **İlk çalıştırma:** Model yükleme 30-60 saniye sürebilir

## 📂 Dosya Yapısı

```
BenzerlikDenemeV3/
├── app.py                    # Ana uygulama
├── audio_classifier.py       # Sınıflandırıcı
├── feature_extractor.py      # Özellik çıkarıcı
├── my_enhanced_audio_model.h5 # Eğitilmiş model
├── scaler.pkl               # Ölçekleyici
├── label_encoder.pkl        # Kodlayıcı
├── requirements.txt         # Python bağımlılıkları
└── README.md               # Proje açıklaması
```

## 🆘 Yardım

Sorun yaşarsanız:

1. **GitHub Issues:** [Sorun Bildir](https://github.com/canpasa1940/BenzerlikDenemeV3/issues)
2. **Email:** [İletişim bilginiz]
3. **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)

## 🔄 Güncelleme

Yeni versiyonları almak için:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**💡 İpucu:** İlk kurulum biraz zaman alabilir ama sonrasında hızlı çalışacak!

**🎵 Keyifli analizler!** 