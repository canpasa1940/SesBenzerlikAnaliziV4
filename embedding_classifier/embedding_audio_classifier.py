import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import librosa
import sys
import os

# Ana dizini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import extract_from_file, extract_features

class EmbeddingAudioClassifier:
    def __init__(self, model_path="my_enhanced_audio_model.h5", 
                 scaler_path="scaler.pkl", 
                 label_encoder_path="label_encoder.pkl"):
        """Ses sınıflandırıcı ve embedding tabanlı benzerlik analizi sınıfı"""
        warnings.filterwarnings("ignore")
        
        # Model ve ön işleme araçlarını yükle
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)
        
        # PKL dosyalarını joblib ile yükle
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
        
        self.classes = self.label_encoder.classes_
        print(f"Model yüklendi. Sınıflar: {self.classes}")
        
        # Embedding model oluştur (son katman öncesi)
        self._create_embedding_model()
        
        # Referans ses veritabanı (embedding'ler ile)
        self.reference_database = []
        
    def _create_embedding_model(self):
        """Son katman öncesi embedding model oluştur"""
        # Model'in son katmanını çıkar
        embedding_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output  # Son katman öncesi
        )
        self.embedding_model = embedding_model
        
        # Embedding boyutunu al
        sample_input = np.zeros((1, 42))  # 42 özellik
        sample_embedding = self.embedding_model.predict(sample_input, verbose=0)
        self.embedding_dim = sample_embedding.shape[1]
        
        print(f"🎯 Embedding model oluşturuldu. Boyut: {self.embedding_dim}")
        
    def remove_silence(self, y, sr=22050, frame_length=2048, hop_length=512):
        """Sessizlik bölümlerini temizle (Voice Activity Detection)"""
        # RMS tabanlı VAD
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Dinamik threshold hesaplama
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        threshold = rms_mean + 0.1 * rms_std  # Adaptif threshold
        
        # En az minimum threshold
        min_threshold = 0.01
        threshold = max(threshold, min_threshold)
        
        # Aktif ses bölgelerini bul
        active_frames = rms > threshold
        
        # Frame'leri zaman indeksine çevir
        active_samples = []
        for i, is_active in enumerate(active_frames):
            if is_active:
                start_sample = i * hop_length
                end_sample = min((i + 1) * hop_length, len(y))
                active_samples.extend(range(start_sample, end_sample))
        
        if len(active_samples) == 0:
            # Hiç aktif bölge bulunamadıysa orijinal sesi döndür
            return y
        
        # Aktif bölgeleri birleştir
        cleaned_audio = y[active_samples]
        
        # Çok kısa temizlenmiş ses durumunda
        if len(cleaned_audio) < sr * 0.5:  # 0.5 saniyeden kısa
            return y  # Orijinal sesi döndür
            
        return cleaned_audio
    
    def segment_audio(self, y, sr=22050, segment_duration=5.0, overlap=0.5):
        """Uzun sesi parçalara böl"""
        segment_samples = int(segment_duration * sr)
        overlap_samples = int(overlap * segment_samples)
        step_samples = segment_samples - overlap_samples
        
        segments = []
        start = 0
        
        while start < len(y):
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            
            # Segment çok kısaysa padding yap
            if len(segment) < segment_samples * 0.5:
                break
                
            segments.append(segment)
            start += step_samples
            
        return segments
    
    def detect_first_pattern(self, y, sr=22050, max_pattern_duration=10.0):
        """İlk karakteristik pattern'i tespit et ve çıkar"""
        # Onset detection ile vuruşları tespit et
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units='frames', 
            hop_length=512, backtrack=True,
            pre_max=20, post_max=20, pre_avg=100, post_avg=100,
            delta=0.1, wait=10
        )
        
        if len(onset_frames) < 2:
            # Yeterli onset yoksa ilk birkaç saniyeyi al
            pattern_samples = min(len(y), int(sr * 3.0))  # İlk 3 saniye
            return y[:pattern_samples]
        
        # Frame'leri time'a çevir
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # Pattern uzunluğunu tahmin et
        if len(onset_times) >= 3:
            # İlk 3 onset arasındaki mesafeyi kullan
            intervals = np.diff(onset_times[:3])
            avg_interval = np.mean(intervals)
            
            # Pattern'i 2-4 interval olarak tahmin et
            estimated_pattern_duration = avg_interval * 3
            
            # Çok uzun olmasın
            estimated_pattern_duration = min(estimated_pattern_duration, max_pattern_duration)
            
        else:
            # Sadece 2 onset varsa aralarındaki mesafeyi kullan
            estimated_pattern_duration = min(onset_times[1] - onset_times[0] + 1.0, max_pattern_duration)
        
        # İlk pattern'i çıkar
        pattern_end_sample = int((onset_times[0] + estimated_pattern_duration) * sr)
        pattern_end_sample = min(pattern_end_sample, len(y))
        
        # En az 1 saniye olsun
        min_samples = int(sr * 1.0)
        if pattern_end_sample < min_samples:
            pattern_end_sample = min(min_samples, len(y))
        
        first_pattern = y[:pattern_end_sample]
        
        print(f"🎯 İlk pattern tespit edildi: {len(first_pattern)/sr:.1f}s ({len(onset_times)} onset bulundu)")
        
        return first_pattern
    
    def get_embedding(self, feature_vector_scaled):
        """Özellik vektöründen embedding çıkar"""
        embedding = self.embedding_model.predict(feature_vector_scaled, verbose=0)
        return embedding[0]  # (embedding_dim,) boyutunda
    
    def predict_single_with_embedding(self, audio_file, use_vad=True, use_segmentation=True, use_first_pattern=True):
        """Embedding tabanlı tek dosya sınıflandırma"""
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Çok kısa sesler için standart yöntemi kullan
            if len(y) < sr * 3:  # 3 saniyeden kısa
                return self.predict_single_with_embedding_simple(audio_file)
            
            print(f"🎵 Uzun ses dosyası tespit edildi ({len(y)/sr:.1f}s)")
            
            # İlk pattern tespiti (en önceki işlem)
            if use_first_pattern and len(y) > sr * 8:  # 8 saniyeden uzun için
                y = self.detect_first_pattern(y, sr)
                print(f"🎼 İlk pattern çıkarıldı: {len(y)/sr:.1f}s")
            
            # VAD uygula
            if use_vad:
                y_cleaned = self.remove_silence(y, sr)
                print(f"🔇 Sessizlik temizlendi: {len(y)/sr:.1f}s → {len(y_cleaned)/sr:.1f}s")
            else:
                y_cleaned = y
            
            # Segmentasyon uygula (kalan ses hala uzunsa)
            if use_segmentation and len(y_cleaned) > sr * 10:  # 10 saniyeden uzun
                segments = self.segment_audio(y_cleaned, sr)
                print(f"✂️ {len(segments)} segmente bölündü")
                
                all_embeddings = []
                all_predictions = []
                all_confidences = []
                
                for i, segment in enumerate(segments):
                    # Segment özelliklerini çıkar
                    features = extract_features(segment, sr)
                    if features is None:
                        continue
                        
                    # DataFrame'e çevir ve sırala
                    feature_names = [f"mfcc{i+1:02d}" for i in range(20)] + [
                        "rms_mean", "rms_std", "zcr_mean", "centroid_mean", 
                        "bandwidth_mean", "rolloff_mean", "flatness_mean", "flux_mean"
                    ] + [f"contrast_b{i+1}" for i in range(7)] + [
                        "onset_mean", "onset_std", "onset_max", "onset_sum",
                        "attack_time", "attack_slope", "hpi_ratio"
                    ]
                    
                    feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
                    feature_vector_scaled = self.scaler.transform(feature_vector)
                    
                    # Embedding çıkar
                    embedding = self.get_embedding(feature_vector_scaled)
                    
                    # Tahmin yap
                    prediction = self.model.predict(feature_vector_scaled, verbose=0)
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = self.classes[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx]
                    
                    all_embeddings.append(embedding)
                    all_predictions.append(predicted_class)
                    all_confidences.append(confidence)
                
                if not all_predictions:
                    return None, None, None, None
                
                # En güvenli tahmini seç
                best_idx = np.argmax(all_confidences)
                final_prediction = all_predictions[best_idx]
                final_confidence = all_confidences[best_idx]
                
                # Embedding'lerin ortalamasını al
                final_embedding = np.mean(all_embeddings, axis=0)
                
                print(f"🎯 En iyi segment: {final_prediction} (güven: {final_confidence:.3f})")
                
                return final_prediction, final_confidence, feature_vector_scaled[0], final_embedding
            
            else:
                # Tek parça olarak işle (temizlenmiş ses ile)
                features = extract_features(y_cleaned, sr)
                if features is None:
                    return None, None, None, None
                    
                # DataFrame'e çevir ve sırala
                feature_names = [f"mfcc{i+1:02d}" for i in range(20)] + [
                    "rms_mean", "rms_std", "zcr_mean", "centroid_mean", 
                    "bandwidth_mean", "rolloff_mean", "flatness_mean", "flux_mean"
                ] + [f"contrast_b{i+1}" for i in range(7)] + [
                    "onset_mean", "onset_std", "onset_max", "onset_sum",
                    "attack_time", "attack_slope", "hpi_ratio"
                ]
                
                feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Embedding çıkar
                embedding = self.get_embedding(feature_vector_scaled)
                
                # Tahmin yap
                prediction = self.model.predict(feature_vector_scaled, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                predicted_class = self.classes[predicted_class_idx]
                confidence = prediction[0][predicted_class_idx]
                
                print(f"🎯 İlk pattern analizi: {predicted_class} (güven: {confidence:.3f})")
                
                return predicted_class, confidence, feature_vector_scaled[0], embedding
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            # Hata durumunda standart yöntemi dene
            return self.predict_single_with_embedding_simple(audio_file)

    def predict_single_with_embedding_simple(self, audio_file):
        """Basit embedding tabanlı tek dosya sınıflandırma"""
        # Özellik çıkar
        features = extract_from_file(audio_file)
        if features is None:
            return None, None, None, None
            
        # DataFrame'e çevir ve sırala
        feature_names = [f"mfcc{i+1:02d}" for i in range(20)] + [
            "rms_mean", "rms_std", "zcr_mean", "centroid_mean", 
            "bandwidth_mean", "rolloff_mean", "flatness_mean", "flux_mean"
        ] + [f"contrast_b{i+1}" for i in range(7)] + [
            "onset_mean", "onset_std", "onset_max", "onset_sum",
            "attack_time", "attack_slope", "hpi_ratio"
        ]
        
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Normalize et
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Embedding çıkar
        embedding = self.get_embedding(feature_vector_scaled)
        
        # Tahmin yap
        prediction = self.model.predict(feature_vector_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence, feature_vector_scaled[0], embedding
    
    def add_to_database(self, audio_file, predicted_class, features, embedding):
        """Sesi referans veritabanına ekle (embedding ile)"""
        self.reference_database.append({
            'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
            'class': predicted_class,
            'features': features,
            'embedding': embedding
        })
    
    def find_similar_sounds_embedding(self, target_embedding, target_class, top_k=5):
        """Toplu matris tabanlı embedding benzerlik analizi"""
        if len(self.reference_database) == 0:
            return []
        
        # Aynı sınıftan tüm embedding'leri topla
        same_class_refs = [ref for ref in self.reference_database if ref['class'] == target_class]
        
        if len(same_class_refs) == 0:
            return []
        
        # Toplu matris oluştur
        # Target: (1, 64) -> (1, 64)
        # References: (N, 64) -> (N, 64) matrisi
        target_matrix = np.array([target_embedding])  # (1, 64)
        ref_matrix = np.array([ref['embedding'] for ref in same_class_refs])  # (N, 64)
        
        print(f"🎯 Toplu matris analizi:")
        print(f"   Target boyutu: {target_matrix.shape}")
        print(f"   Referans matrisi boyutu: {ref_matrix.shape}")
        
        # Toplu cosine similarity hesapla
        # cosine_similarity(target_matrix, ref_matrix) -> (1, N) matrisi
        cos_similarities = cosine_similarity(target_matrix, ref_matrix)[0]  # (N,) vektörü
        
        # Toplu euclidean distance hesapla
        # euclidean_distances(target_matrix, ref_matrix) -> (1, N) matrisi
        euc_distances = euclidean_distances(target_matrix, ref_matrix)[0]  # (N,) vektörü
        
        # Sonuçları birleştir
        similarities = []
        for i, ref in enumerate(same_class_refs):
            similarities.append({
                'filename': ref['filename'],
                'class': ref['class'],
                'cosine_similarity': cos_similarities[i],
                'euclidean_distance': euc_distances[i],
                'embedding': ref['embedding']
            })
        
        # Cosine similarity'ye göre sırala
        similarities = sorted(similarities, key=lambda x: x['cosine_similarity'], reverse=True)
        
        print(f"   ✅ Toplu analiz tamamlandı: {len(similarities)} benzerlik hesaplandı")
        
        return similarities[:top_k]
    
    def find_similar_sounds_features(self, target_features, target_class, top_k=5):
        """Toplu matris tabanlı özellik benzerlik analizi"""
        if len(self.reference_database) == 0:
            return []
        
        # Aynı sınıftan tüm özellikleri topla
        same_class_refs = [ref for ref in self.reference_database if ref['class'] == target_class]
        
        if len(same_class_refs) == 0:
            return []
        
        # Toplu matris oluştur
        # Target: (1, 42) -> (1, 42) - 42 özellik
        # References: (N, 42) -> (N, 42) matrisi
        target_matrix = np.array([target_features])  # (1, 42)
        ref_matrix = np.array([ref['features'] for ref in same_class_refs])  # (N, 42)
        
        print(f"🔧 Toplu özellik matris analizi:")
        print(f"   Target boyutu: {target_matrix.shape}")
        print(f"   Referans matrisi boyutu: {ref_matrix.shape}")
        
        # Toplu cosine similarity hesapla
        cos_similarities = cosine_similarity(target_matrix, ref_matrix)[0]  # (N,) vektörü
        
        # Toplu euclidean distance hesapla
        euc_distances = euclidean_distances(target_matrix, ref_matrix)[0]  # (N,) vektörü
        
        # Sonuçları birleştir
        similarities = []
        for i, ref in enumerate(same_class_refs):
            similarities.append({
                'filename': ref['filename'],
                'class': ref['class'],
                'cosine_similarity': cos_similarities[i],
                'euclidean_distance': euc_distances[i],
                'features': ref['features']
            })
        
        # Cosine similarity'ye göre sırala
        similarities = sorted(similarities, key=lambda x: x['cosine_similarity'], reverse=True)
        
        print(f"   ✅ Toplu özellik analizi tamamlandı: {len(similarities)} benzerlik hesaplandı")
        
        return similarities[:top_k]
    
    def get_pca_visualization_data_embedding(self, target_embedding, target_class):
        """Embedding PCA ile 2D görselleştirme verisi hazırla"""
        if len(self.reference_database) < 2:
            return None, None, None, None
            
        # Aynı sınıftan sesleri al
        same_class_embeddings = []
        same_class_names = []
        
        for ref in self.reference_database:
            if ref['class'] == target_class:
                same_class_embeddings.append(ref['embedding'])
                same_class_names.append(ref['filename'])
        
        if len(same_class_embeddings) < 2:
            return None, None, None, None
            
        # Target ses ile birleştir
        all_embeddings = same_class_embeddings + [target_embedding]
        all_names = same_class_names + ['Yüklenen Ses']
        
        # PCA uygula
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(all_embeddings)
        
        return pca_embeddings, all_names, pca.explained_variance_ratio_, pca
    
    def classify_multiple_files(self, audio_files):
        """Birden fazla ses dosyasını sınıflandır (embedding ile)"""
        results = []
        
        for audio_file in audio_files:
            result = self.predict_single_with_embedding_simple(audio_file)
            if result[0] is not None:
                predicted_class, confidence, features, embedding = result
                
                # Veritabanına ekle
                self.add_to_database(audio_file, predicted_class, features, embedding)
                
                results.append({
                    'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'features': features,
                    'embedding': embedding
                })
        
        return results
    
    def get_database_summary(self):
        """Veritabanı özetini döndür"""
        if not self.reference_database:
            return {}
            
        df = pd.DataFrame(self.reference_database)
        summary = df['class'].value_counts().to_dict()
        return summary
    
    def compare_similarity_methods(self, target_features, target_embedding, target_class, top_k=5):
        """Özellik ve embedding tabanlı benzerlik yöntemlerini karşılaştır"""
        feature_similarities = self.find_similar_sounds_features(target_features, target_class, top_k)
        embedding_similarities = self.find_similar_sounds_embedding(target_embedding, target_class, top_k)
        
        comparison = {
            'feature_based': feature_similarities,
            'embedding_based': embedding_similarities
        }
        
        return comparison 