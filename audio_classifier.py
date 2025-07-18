import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import librosa
from feature_extractor import extract_from_file, extract_features

class AudioClassifier:
    def __init__(self, model_path="my_enhanced_audio_model.h5", 
                 scaler_path="scaler.pkl", 
                 label_encoder_path="label_encoder.pkl"):
        """Ses sınıflandırıcı ve benzerlik analizi sınıfı"""
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
        
        # Referans ses veritabanı
        self.reference_database = []
        
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
    
    def predict_single_enhanced(self, audio_file, use_vad=True, use_segmentation=True, use_first_pattern=True):
        """Gelişmiş tek dosya sınıflandırma (VAD + Segmentasyon + İlk Pattern)"""
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Çok kısa sesler için standart yöntemi kullan
            if len(y) < sr * 3:  # 3 saniyeden kısa
                return self.predict_single(audio_file)
            
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
                
                all_features = []
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
                    
                    # Tahmin yap
                    prediction = self.model.predict(feature_vector_scaled, verbose=0)
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = self.classes[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx]
                    
                    all_features.append(feature_vector_scaled[0])
                    all_predictions.append(predicted_class)
                    all_confidences.append(confidence)
                
                if not all_predictions:
                    return None, None, None
                
                # En güvenli tahmini seç
                best_idx = np.argmax(all_confidences)
                final_prediction = all_predictions[best_idx]
                final_confidence = all_confidences[best_idx]
                
                # Özelliklerin ortalamasını al
                final_features = np.mean(all_features, axis=0)
                
                print(f"🎯 En iyi segment: {final_prediction} (güven: {final_confidence:.3f})")
                
                return final_prediction, final_confidence, final_features
            
            else:
                # Tek parça olarak işle (temizlenmiş ses ile)
                features = extract_features(y_cleaned, sr)
                if features is None:
                    return None, None, None
                    
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
                
                # Tahmin yap
                prediction = self.model.predict(feature_vector_scaled, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                predicted_class = self.classes[predicted_class_idx]
                confidence = prediction[0][predicted_class_idx]
                
                print(f"🎯 İlk pattern analizi: {predicted_class} (güven: {confidence:.3f})")
                
                return predicted_class, confidence, feature_vector_scaled[0]
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            # Hata durumunda standart yöntemi dene
            return self.predict_single(audio_file)

    def predict_single(self, audio_file):
        """Orijinal tek dosya sınıflandırma"""
        # Özellik çıkar
        features = extract_from_file(audio_file)
        if features is None:
            return None, None, None
            
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
        
        # Tahmin yap
        prediction = self.model.predict(feature_vector_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence, feature_vector_scaled[0]
    
    def add_to_database(self, audio_file, predicted_class, features):
        """Sesi referans veritabanına ekle"""
        self.reference_database.append({
            'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
            'class': predicted_class,
            'features': features
        })
    
    def find_similar_sounds(self, target_features, target_class, top_k=5):
        """Benzer sesleri bul"""
        if len(self.reference_database) == 0:
            return []
            
        similarities = []
        
        for ref in self.reference_database:
            # Aynı sınıftan olanları öncelendir
            if ref['class'] == target_class:
                # Cosine similarity hesapla
                cos_sim = cosine_similarity([target_features], [ref['features']])[0][0]
                
                # Euclidean distance hesapla  
                euc_dist = euclidean_distances([target_features], [ref['features']])[0][0]
                
                similarities.append({
                    'filename': ref['filename'],
                    'class': ref['class'],
                    'cosine_similarity': cos_sim,
                    'euclidean_distance': euc_dist,
                    'features': ref['features']
                })
        
        # Cosine similarity'ye göre sırala (yüksekten düşüğe)
        similarities = sorted(similarities, key=lambda x: x['cosine_similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_pca_visualization_data(self, target_features, target_class):
        """PCA ile 2D görselleştirme verisi hazırla"""
        if len(self.reference_database) < 2:
            return None, None, None, None
            
        # Aynı sınıftan sesleri al
        same_class_features = []
        same_class_names = []
        
        for ref in self.reference_database:
            if ref['class'] == target_class:
                same_class_features.append(ref['features'])
                same_class_names.append(ref['filename'])
        
        if len(same_class_features) < 2:
            return None, None, None, None
            
        # Target ses ile birleştir
        all_features = same_class_features + [target_features]
        all_names = same_class_names + ['Yüklenen Ses']
        
        # PCA uygula
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(all_features)
        
        return pca_features, all_names, pca.explained_variance_ratio_, pca
    
    def classify_multiple_files(self, audio_files):
        """Birden fazla ses dosyasını sınıflandır"""
        results = []
        
        for audio_file in audio_files:
            result = self.predict_single(audio_file)
            if result[0] is not None:
                predicted_class, confidence, features = result
                
                # Veritabanına ekle
                self.add_to_database(audio_file, predicted_class, features)
                
                results.append({
                    'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'features': features
                })
        
        return results
    
    def get_database_summary(self):
        """Veritabanı özetini döndür"""
        if not self.reference_database:
            return {}
            
        df = pd.DataFrame(self.reference_database)
        summary = df['class'].value_counts().to_dict()
        return summary 