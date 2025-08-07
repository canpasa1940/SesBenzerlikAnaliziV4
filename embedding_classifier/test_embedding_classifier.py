#!/usr/bin/env python3
"""
Embedding tabanlı ses sınıflandırıcı test dosyası
Son katman öncesi embedding'leri benzerlik analizi için kullanır
"""

import sys
import os
sys.path.append('..')  # Ana dizini path'e ekle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from embedding_audio_classifier import EmbeddingAudioClassifier
import warnings

# Ana dizine geç (embedding_classifier klasöründen çık)
if os.path.basename(os.getcwd()) == 'embedding_classifier':
    os.chdir('..')

warnings.filterwarnings("ignore")

def test_embedding_classifier():
    """Embedding tabanlı sınıflandırıcıyı test et"""
    print("🎵 Embedding Tabanlı Ses Sınıflandırıcı Testi")
    print("=" * 50)
    
    try:
        # Sınıflandırıcıyı yükle
        print("📦 Sınıflandırıcı yükleniyor...")
        print(f"📁 Çalışma dizini: {os.getcwd()}")
        print(f"📁 Model dosyası var mı: {os.path.exists('my_enhanced_audio_model.h5')}")
        classifier = EmbeddingAudioClassifier()
        
        print(f"✅ Model yüklendi!")
        print(f"📊 Embedding boyutu: {classifier.embedding_dim}")
        print(f"🎯 Sınıflar: {classifier.classes}")
        
        # Test embedding çıkarma
        print("\n🔍 Test embedding çıkarma...")
        sample_features = np.random.randn(1, 42)  # 42 özellik
        sample_embedding = classifier.get_embedding(sample_features)
        print(f"✅ Embedding boyutu: {sample_embedding.shape}")
        
        # Embedding istatistikleri
        print(f"📈 Embedding istatistikleri:")
        print(f"   Ortalama: {np.mean(sample_embedding):.4f}")
        print(f"   Standart sapma: {np.std(sample_embedding):.4f}")
        print(f"   Min: {np.min(sample_embedding):.4f}")
        print(f"   Max: {np.max(sample_embedding):.4f}")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

def test_similarity_comparison(classifier):
    """Özellik ve embedding tabanlı benzerlik yöntemlerini karşılaştır"""
    print("\n🔄 Benzerlik Yöntemleri Karşılaştırması")
    print("=" * 50)
    
    # Test verileri oluştur
    print("📊 Test verileri oluşturuluyor...")
    
    # Farklı sınıflardan test sesleri oluştur
    test_classes = classifier.classes[:3]  # İlk 3 sınıf
    
    for class_name in test_classes:
        print(f"\n🎯 {class_name} sınıfı için test:")
        
        # Aynı sınıftan 5 farklı "ses" oluştur (simüle edilmiş)
        for i in range(5):
            # Rastgele özellikler oluştur
            features = np.random.randn(42)
            features_scaled = classifier.scaler.transform(features.reshape(1, -1))
            embedding = classifier.get_embedding(features_scaled)
            
            # Veritabanına ekle
            classifier.add_to_database(
                f"test_{class_name}_{i}.wav", 
                class_name, 
                features_scaled[0], 
                embedding
            )
        
        print(f"   ✅ {class_name} sınıfından 5 test sesi eklendi")
    
    # Test hedef sesi oluştur
    target_features = np.random.randn(42)
    target_features_scaled = classifier.scaler.transform(target_features.reshape(1, -1))
    target_embedding = classifier.get_embedding(target_features_scaled)
    target_class = test_classes[0]  # İlk sınıfı hedef al
    
    print(f"\n🎯 Hedef ses: {target_class} sınıfı")
    
    # Benzerlik karşılaştırması
    comparison = classifier.compare_similarity_methods(
        target_features_scaled[0], 
        target_embedding, 
        target_class, 
        top_k=3
    )
    
    # Sonuçları göster
    print("\n📊 Karşılaştırma Sonuçları:")
    print("-" * 30)
    
    print("🔧 Özellik Tabanlı Benzerlik:")
    for i, sim in enumerate(comparison['feature_based']):
        print(f"   {i+1}. {sim['filename']}: {sim['cosine_similarity']:.4f}")
    
    print("\n🧠 Embedding Tabanlı Benzerlik:")
    for i, sim in enumerate(comparison['embedding_based']):
        print(f"   {i+1}. {sim['filename']}: {sim['cosine_similarity']:.4f}")
    
    return comparison

def visualize_embeddings(classifier, comparison):
    """Embedding'leri görselleştir"""
    print("\n📈 Embedding Görselleştirme")
    print("=" * 50)
    
    try:
        # PCA ile 2D görselleştirme
        target_class = classifier.classes[0]
        
        # Aynı sınıftan embedding'leri al
        same_class_embeddings = []
        same_class_names = []
        
        for ref in classifier.reference_database:
            if ref['class'] == target_class:
                same_class_embeddings.append(ref['embedding'])
                same_class_names.append(ref['filename'])
        
        if len(same_class_embeddings) >= 3:
            # PCA uygula
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(same_class_embeddings)
            
            # Görselleştir
            plt.figure(figsize=(10, 6))
            
            # PCA scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], alpha=0.7)
            for i, name in enumerate(same_class_names):
                plt.annotate(name.split('_')[-1], (pca_embeddings[i, 0], pca_embeddings[i, 1]))
            plt.title(f'{target_class} Sınıfı - PCA Görselleştirme')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # Benzerlik karşılaştırması
            plt.subplot(1, 2, 2)
            feature_sims = [sim['cosine_similarity'] for sim in comparison['feature_based']]
            embedding_sims = [sim['cosine_similarity'] for sim in comparison['embedding_based']]
            
            x = np.arange(len(feature_sims))
            width = 0.35
            
            plt.bar(x - width/2, feature_sims, width, label='Özellik Tabanlı', alpha=0.8)
            plt.bar(x + width/2, embedding_sims, width, label='Embedding Tabanlı', alpha=0.8)
            
            plt.xlabel('Benzer Ses Sırası')
            plt.ylabel('Cosine Similarity')
            plt.title('Benzerlik Yöntemleri Karşılaştırması')
            plt.legend()
            plt.xticks(x, [f'{i+1}' for i in range(len(feature_sims))])
            
            plt.tight_layout()
            plt.savefig('embedding_classifier/embedding_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Görselleştirme kaydedildi: embedding_visualization.png")
            
        else:
            print("⚠️ Görselleştirme için yeterli veri yok")
            
    except Exception as e:
        print(f"❌ Görselleştirme hatası: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 Embedding Tabanlı Ses Sınıflandırıcı Test Başlıyor")
    print("=" * 60)
    
    # Sınıflandırıcıyı test et
    classifier = test_embedding_classifier()
    
    if classifier is None:
        print("❌ Test başarısız!")
        return
    
    # Benzerlik karşılaştırması
    comparison = test_similarity_comparison(classifier)
    
    # Görselleştirme
    visualize_embeddings(classifier, comparison)
    
    # Özet
    print("\n📋 Test Özeti")
    print("=" * 30)
    print(f"✅ Embedding boyutu: {classifier.embedding_dim}")
    print(f"✅ Veritabanındaki ses sayısı: {len(classifier.reference_database)}")
    print(f"✅ Test başarıyla tamamlandı!")
    
    # Veritabanı özeti
    summary = classifier.get_database_summary()
    print(f"\n📊 Veritabanı Özeti:")
    for class_name, count in summary.items():
        print(f"   {class_name}: {count} ses")

if __name__ == "__main__":
    main() 