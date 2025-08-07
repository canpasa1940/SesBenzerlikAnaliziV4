# 🧠 Embedding Tabanlı Ses Sınıflandırıcı

Bu klasör, **son katman öncesi embedding'leri** benzerlik analizi için kullanan gelişmiş bir ses sınıflandırıcı içerir.

## 🎯 Özellikler

### 🔧 Mevcut Yöntem vs 🧠 Embedding Yöntemi

| Özellik | Mevcut Yöntem | Embedding Yöntemi |
|---------|---------------|-------------------|
| **Benzerlik Analizi** | 42 özellik vektörü | Son katman öncesi embedding |
| **Boyut** | 42 boyut | Model'e göre değişir (genellikle 64-128) |
| **Anlam** | Ham özellikler | Öğrenilmiş temsiller |
| **Performans** | İyi | Daha iyi (beklenen) |

## 📁 Dosya Yapısı

```
embedding_classifier/
├── embedding_audio_classifier.py    # Ana sınıflandırıcı sınıfı
├── test_embedding_classifier.py     # Test ve karşılaştırma dosyası
├── README.md                        # Bu dosya
└── embedding_visualization.png      # Test sonuçları (otomatik oluşur)
```

## 🚀 Kullanım

### 1. Temel Kullanım

```python
from embedding_audio_classifier import EmbeddingAudioClassifier

# Sınıflandırıcıyı yükle
classifier = EmbeddingAudioClassifier()

# Ses dosyasını analiz et
predicted_class, confidence, features, embedding = classifier.predict_single_with_embedding(
    "ses_dosyasi.wav"
)

print(f"Sınıf: {predicted_class}")
print(f"Güven: {confidence:.3f}")
print(f"Embedding boyutu: {embedding.shape}")
```

### 2. Benzerlik Analizi

```python
# Veritabanına ses ekle
classifier.add_to_database(audio_file, predicted_class, features, embedding)

# Embedding tabanlı benzerlik
similar_sounds = classifier.find_similar_sounds_embedding(
    target_embedding, target_class, top_k=5
)

# Özellik tabanlı benzerlik (karşılaştırma için)
feature_similar_sounds = classifier.find_similar_sounds_features(
    target_features, target_class, top_k=5
)
```

### 3. Yöntem Karşılaştırması

```python
# Her iki yöntemi karşılaştır
comparison = classifier.compare_similarity_methods(
    target_features, target_embedding, target_class, top_k=5
)

print("Özellik tabanlı:", comparison['feature_based'])
print("Embedding tabanlı:", comparison['embedding_based'])
```

## 🧪 Test Etme

Test dosyasını çalıştırmak için:

```bash
cd embedding_classifier
python test_embedding_classifier.py
```

Bu test:
- ✅ Embedding model oluşturmayı test eder
- ✅ Benzerlik yöntemlerini karşılaştırır
- ✅ Görselleştirme oluşturur
- ✅ Performans analizi yapar

## 🔍 Teknik Detaylar

### Embedding Model Oluşturma

```python
def _create_embedding_model(self):
    """Son katman öncesi embedding model oluştur"""
    embedding_model = tf.keras.Model(
        inputs=self.model.input,
        outputs=self.model.layers[-2].output  # Son katman öncesi
    )
    self.embedding_model = embedding_model
```

### Embedding Çıkarma

```python
def get_embedding(self, feature_vector_scaled):
    """Özellik vektöründen embedding çıkar"""
    embedding = self.embedding_model.predict(feature_vector_scaled, verbose=0)
    return embedding[0]
```

## 📊 Beklenen Avantajlar

### 1. **Daha İyi Temsil**
- Embedding'ler model tarafından öğrenilmiş anlamlı temsillerdir
- Ham özelliklerden daha soyut ve güçlüdür

### 2. **Boyut Optimizasyonu**
- Embedding boyutu genellikle 42'den küçüktür
- Daha hızlı benzerlik hesaplama

### 3. **Transfer Learning**
- Farklı modellerle kullanılabilir
- Pre-trained embedding'ler kullanılabilir

## 🔬 Karşılaştırma Metrikleri

Test dosyası şu metrikleri karşılaştırır:

1. **Cosine Similarity** - Açısal benzerlik
2. **Euclidean Distance** - Öklid mesafesi
3. **PCA Görselleştirme** - 2D temsil
4. **Performans Analizi** - Hız ve doğruluk

## 📈 Sonuçlar

Test sonuçları `embedding_visualization.png` dosyasında saklanır:

- **Sol grafik**: PCA ile embedding'lerin 2D görselleştirmesi
- **Sağ grafik**: Özellik vs embedding tabanlı benzerlik karşılaştırması

## 🎯 Kullanım Senaryoları

1. **Müzik Prodüksiyonu**: Benzer davul sesleri bulma
2. **Ses Tasarımı**: Kütüphane organizasyonu
3. **Araştırma**: Ses özellik analizi
4. **Müzik Keşfi**: Uyumlu perküsyon bulma

## 🔧 Gereksinimler

Ana proje ile aynı gereksinimler:
- TensorFlow/Keras
- Librosa
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

## 📝 Notlar

- Embedding boyutu model mimarisine bağlıdır
- Son katman öncesi katman seçimi önemlidir
- Test dosyası simüle edilmiş veriler kullanır
- Gerçek ses dosyaları ile test etmek için ana uygulamayı kullanın 