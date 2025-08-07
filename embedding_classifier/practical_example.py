#!/usr/bin/env python3
"""
Embedding Tabanlı Ses Sınıflandırıcı - Pratik Kullanım Örneği
"""

import sys
import os
import numpy as np
import tempfile
import warnings

# Ana dizini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_audio_classifier import EmbeddingAudioClassifier

warnings.filterwarnings("ignore")

def demo_embedding_classifier():
    """Embedding tabanlı sınıflandırıcı demo"""
    print("🎵 Embedding Tabanlı Ses Sınıflandırıcı Demo")
    print("=" * 50)
    
    # 1. Sınıflandırıcıyı yükle
    print("📦 Sınıflandırıcı yükleniyor...")
    classifier = EmbeddingAudioClassifier()
    print(f"✅ Model yüklendi! Embedding boyutu: {classifier.embedding_dim}")
    
    # 2. Test verileri oluştur (gerçek ses dosyaları yerine simüle edilmiş)
    print("\n🎯 Test verileri oluşturuluyor...")
    
    # Farklı sınıflardan test "sesleri" oluştur
    test_data = []
    for class_name in classifier.classes[:3]:  # İlk 3 sınıf
        for i in range(3):  # Her sınıftan 3 ses
            # Rastgele özellikler oluştur (gerçek ses yerine)
            features = np.random.randn(42)
            features_scaled = classifier.scaler.transform(features.reshape(1, -1))
            embedding = classifier.get_embedding(features_scaled)
            
            # Sınıflandırma yap
            prediction = classifier.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = classifier.classes[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]
            
            test_data.append({
                'filename': f"test_{class_name}_{i}.wav",
                'class': predicted_class,
                'confidence': confidence,
                'features': features_scaled[0],
                'embedding': embedding
            })
            
            # Veritabanına ekle
            classifier.add_to_database(
                f"test_{class_name}_{i}.wav", 
                predicted_class, 
                features_scaled[0], 
                embedding
            )
    
    print(f"✅ {len(test_data)} test sesi oluşturuldu")
    
    # 3. Hedef ses oluştur ve analiz et
    print("\n🎯 Hedef ses analizi...")
    
    # Hedef ses için özellikler oluştur
    target_features = np.random.randn(42)
    target_features_scaled = classifier.scaler.transform(target_features.reshape(1, -1))
    target_embedding = classifier.get_embedding(target_features_scaled)
    
    # Sınıflandırma yap
    target_prediction = classifier.model.predict(target_features_scaled, verbose=0)
    target_class_idx = np.argmax(target_prediction[0])
    target_class = classifier.classes[target_class_idx]
    target_confidence = target_prediction[0][target_class_idx]
    
    print(f"🎯 Hedef Ses Analizi:")
    print(f"   📁 Dosya: target_sound.wav")
    print(f"   🎵 Sınıf: {target_class}")
    print(f"   📊 Güven: {target_confidence:.3f}")
    print(f"   🧠 Embedding boyutu: {target_embedding.shape}")
    
    # 4. Benzerlik analizi
    print(f"\n🔍 Benzerlik Analizi ({target_class} sınıfı)...")
    
    # Embedding tabanlı benzerlik
    similar_sounds = classifier.find_similar_sounds_embedding(
        target_embedding, target_class, top_k=5
    )
    
    print("🧠 Embedding Tabanlı Benzerlik:")
    for i, sim in enumerate(similar_sounds):
        print(f"   {i+1}. {sim['filename']}: {sim['cosine_similarity']:.4f}")
    
    # Özellik tabanlı benzerlik (karşılaştırma)
    feature_similar_sounds = classifier.find_similar_sounds_features(
        target_features_scaled[0], target_class, top_k=5
    )
    
    print("\n🔧 Özellik Tabanlı Benzerlik:")
    for i, sim in enumerate(feature_similar_sounds):
        print(f"   {i+1}. {sim['filename']}: {sim['cosine_similarity']:.4f}")
    
    # 5. Yöntem karşılaştırması
    print(f"\n📊 Yöntem Karşılaştırması:")
    
    comparison = classifier.compare_similarity_methods(
        target_features_scaled[0], target_embedding, target_class, top_k=3
    )
    
    print("   En İyi Embedding Benzerlikleri:")
    for i, sim in enumerate(comparison['embedding_based'][:3]):
        print(f"     {i+1}. {sim['cosine_similarity']:.4f}")
    
    print("   En İyi Özellik Benzerlikleri:")
    for i, sim in enumerate(comparison['feature_based'][:3]):
        print(f"     {i+1}. {sim['cosine_similarity']:.4f}")
    
    # 6. PCA görselleştirme
    print(f"\n📈 PCA Görselleştirme...")
    
    pca_data = classifier.get_pca_visualization_data_embedding(
        target_embedding, target_class
    )
    
    if pca_data[0] is not None:
        pca_features, names, variance_ratio, pca = pca_data
        print(f"   ✅ PCA uygulandı!")
        print(f"   📊 Açıklanan varyans: {variance_ratio[0]:.2%}, {variance_ratio[1]:.2%}")
        print(f"   🎯 Görselleştirme için {len(names)} ses")
    else:
        print("   ⚠️ PCA için yeterli veri yok")
    
    # 7. Özet
    print(f"\n📋 Demo Özeti:")
    print(f"   ✅ Embedding boyutu: {classifier.embedding_dim}")
    print(f"   ✅ Veritabanındaki ses: {len(classifier.reference_database)}")
    print(f"   ✅ Hedef sınıf: {target_class}")
    print(f"   ✅ En iyi embedding benzerlik: {similar_sounds[0]['cosine_similarity']:.4f}")
    print(f"   ✅ En iyi özellik benzerlik: {feature_similar_sounds[0]['cosine_similarity']:.4f}")
    
    return classifier, target_embedding, target_class

def interactive_demo():
    """İnteraktif demo"""
    print("\n🎮 İnteraktif Demo")
    print("=" * 30)
    
    classifier, target_embedding, target_class = demo_embedding_classifier()
    
    while True:
        print(f"\n🔍 Ne yapmak istiyorsunuz?")
        print("1. Benzerlik analizi yap")
        print("2. Veritabanı özetini göster")
        print("3. Yeni test sesi ekle")
        print("4. Çıkış")
        
        choice = input("Seçiminiz (1-4): ").strip()
        
        if choice == "1":
            print(f"\n🎯 {target_class} sınıfından benzer sesler:")
            similar_sounds = classifier.find_similar_sounds_embedding(
                target_embedding, target_class, top_k=5
            )
            for i, sim in enumerate(similar_sounds):
                print(f"   {i+1}. {sim['filename']}: {sim['cosine_similarity']:.4f}")
                
        elif choice == "2":
            summary = classifier.get_database_summary()
            print(f"\n📊 Veritabanı Özeti:")
            for class_name, count in summary.items():
                print(f"   {class_name}: {count} ses")
                
        elif choice == "3":
            print(f"\n➕ Yeni test sesi ekleniyor...")
            # Yeni rastgele ses oluştur
            new_features = np.random.randn(42)
            new_features_scaled = classifier.scaler.transform(new_features.reshape(1, -1))
            new_embedding = classifier.get_embedding(new_features_scaled)
            
            new_prediction = classifier.model.predict(new_features_scaled, verbose=0)
            new_class_idx = np.argmax(new_prediction[0])
            new_class = classifier.classes[new_class_idx]
            new_confidence = new_prediction[0][new_class_idx]
            
            classifier.add_to_database(
                f"new_test_{len(classifier.reference_database)}.wav",
                new_class, new_features_scaled[0], new_embedding
            )
            
            print(f"   ✅ Yeni ses eklendi: {new_class} (güven: {new_confidence:.3f})")
            
        elif choice == "4":
            print("👋 Demo sonlandırılıyor...")
            break
            
        else:
            print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    # Ana dizine geç
    if os.path.basename(os.getcwd()) == 'embedding_classifier':
        os.chdir('..')
    
    # Demo çalıştır
    interactive_demo() 