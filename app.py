import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from audio_classifier import AudioClassifier
import tempfile
import os
import warnings

warnings.filterwarnings("ignore")

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎵 Ses Benzerlik Analizi",
    page_icon="🎵",
    layout="wide"
)

# CSS stilini ekle
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .similarity-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Sınıflandırıcıyı yükle (cache'lendi)"""
    try:
        classifier = AudioClassifier()
        return classifier
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

# Session state'i başlat
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'reference_database' not in st.session_state:
    st.session_state.reference_database = []
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

def main():
    st.markdown('<div class="main-header">🎵 Ses Benzerlik Analizi</div>', 
                unsafe_allow_html=True)
    
    # Sınıflandırıcıyı yükle
    classifier = load_classifier()
    if classifier is None:
        st.stop()
    
    # Cache'deki veritabanını classifier'a yükle
    classifier.reference_database = st.session_state.reference_database
    
    st.markdown(f"""
    **Desteklenen Sınıflar:** {', '.join(classifier.classes)}
    
    **Kullanım:** Ses dosyalarınızı yükleyin ve sistem otomatik olarak sınıflandıracak.
    """)
    
    # Tab'lar oluştur
    tab1, tab2 = st.tabs(["📁 Toplu Yükleme", "🎯 Tek Dosya Analizi"])
    
    with tab1:
        st.subheader("📁 Çoklu Ses Dosyası Yükleme")
        st.write("Birden fazla ses dosyasını yükleyip toplu analiz yapın.")
        
        # Gelişmiş işleme seçenekleri
        with st.expander("⚙️ Gelişmiş İşleme Ayarları"):
            col1, col2, col3 = st.columns(3)
            with col1:
                use_vad = st.checkbox("🔇 Sessizlik Temizleme (VAD)", value=True, 
                                    help="Uzun ses dosyalarından sessiz bölümleri temizler")
            with col2:
                use_segmentation = st.checkbox("✂️ Ses Segmentasyonu", value=True, 
                                             help="Çok uzun sesleri parçalara böler")
            with col3:
                use_first_pattern = st.checkbox("🎼 İlk Pattern Tespiti", value=True, 
                                               help="Tekrar eden seslerden sadece ilk karakteristik bölümü alır")
        
        # Toplu dosya yükleme
        uploaded_files = st.file_uploader(
            "🎧 Ses dosyalarınızı yükleyin",
            type=['wav'],
            accept_multiple_files=True,
            help="WAV formatında ses dosyaları yükleyebilirsiniz",
            key="bulk_upload"
        )
        
        if uploaded_files:
            st.markdown("---")
            
            # Yeni dosyaları kontrol et (cache için)
            new_files = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['filename'] for f in st.session_state.processed_files]:
                    new_files.append(uploaded_file)
            
            if new_files:
                st.info(f"{len(new_files)} yeni dosya bulundu. İşleniyor...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Sadece yeni dosyaları işle
                for i, uploaded_file in enumerate(new_files):
                    status_text.text(f"İşleniyor: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / len(new_files))
                    
                    # Geçici dosya oluştur
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        file_content = uploaded_file.read()
                        tmp_file.write(file_content)
                        tmp_file.flush()
                        
                        # Gelişmiş sınıflandırma (VAD + Segmentasyon + İlk Pattern)
                        predicted_class, confidence, features = classifier.predict_single_enhanced(
                            tmp_file.name, use_vad=use_vad, use_segmentation=use_segmentation, 
                            use_first_pattern=use_first_pattern
                        )
                        
                        if predicted_class is not None:
                            # Veritabanına ekle
                            classifier.add_to_database(uploaded_file, predicted_class, features)
                            
                            # Session state'e ekle
                            st.session_state.processed_files.append({
                                'filename': uploaded_file.name,
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'features': features
                            })
                            
                            # Ses dosyasını cache'le
                            st.session_state.audio_cache[uploaded_file.name] = file_content
                        
                        # Geçici dosyayı sil
                        os.unlink(tmp_file.name)
                
                # Session state'i güncelle
                st.session_state.reference_database = classifier.reference_database.copy()
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"{len(new_files)} dosya başarıyla işlendi!")
            
            # Tüm işlenmiş dosyaları göster
            results = st.session_state.processed_files
            
            if results:
                # Sonuçları göster
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 Sınıflandırma Sonuçları")
                    
                    # Temizle butonu
                    if st.button("🗑️ Tümünü Temizle", help="Tüm yüklenmiş dosyaları temizle"):
                        st.session_state.processed_files = []
                        st.session_state.reference_database = []
                        st.session_state.audio_cache = {}
                        classifier.reference_database = []
                        st.rerun()
                    
                    # Sonuç tablosu
                    df_results = pd.DataFrame([
                        {
                            'Dosya': r['filename'],
                            'Tahmin': r['predicted_class'],
                            'Güven': f"{r['confidence']:.3f}"
                        } for r in results
                    ])
                    
                    st.dataframe(df_results, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Sınıf Dağılımı")
                    
                    # Sınıf sayıları
                    class_counts = pd.Series([r['predicted_class'] for r in results]).value_counts()
                    
                    # Pie chart
                    fig_pie = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Sınıf Dağılımı"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Ses çalma bölümü
                st.subheader("🎧 Ses Çalar")
                selected_audio = st.selectbox(
                    "Çalmak istediğiniz sesi seçin:",
                    options=[r['filename'] for r in results],
                    key="audio_player_bulk"
                )
                
                # Seçilen ses dosyasını çal
                if selected_audio and selected_audio in st.session_state.audio_cache:
                    st.audio(st.session_state.audio_cache[selected_audio], format='audio/wav')
                
                # Güven skoru histogramı
                st.subheader("📊 Güven Skoru Dağılımı")
                confidences = [r['confidence'] for r in results]
                fig_hist = px.histogram(
                    x=confidences,
                    nbins=20,
                    title="Güven Skorları",
                    labels={'x': 'Güven Skoru', 'y': 'Frekans'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.markdown("---")
                
                # Benzerlik analizi için ses seçimi
                st.subheader("🔍 Benzerlik Analizi")
                
                selected_file = st.selectbox(
                    "Benzerlik analizi için bir ses seçin:",
                    options=[r['filename'] for r in results],
                    help="Seçilen ses ile aynı sınıftaki diğer sesler arasında benzerlik analizi yapılacak"
                )
                
                if selected_file:
                    # Seçilen sesi bul
                    selected_result = next(r for r in results if r['filename'] == selected_file)
                    target_class = selected_result['predicted_class']
                    target_features = selected_result['features']
                    
                    st.markdown(f"""
                    **Seçilen Ses:** {selected_file}  
                    **Sınıf:** {target_class}  
                    **Güven:** {selected_result['confidence']:.3f}
                    """)
                    
                    # Seçilen sesi çal
                    if selected_file in st.session_state.audio_cache:
                        st.audio(st.session_state.audio_cache[selected_file], format='audio/wav')
                    
                    # Benzer sesleri bul
                    similar_sounds = classifier.find_similar_sounds(
                        target_features, target_class, top_k=10
                    )
                    
                    if similar_sounds:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader(f"🎯 En Benzer Sesler ({target_class})")
                            
                            for i, sim in enumerate(similar_sounds):
                                if sim['filename'] != selected_file:  # Kendisini gösterme
                                    st.markdown(f"""
                                    <div class="similarity-card">
                                        <strong>{i+1}. {sim['filename']}</strong><br>
                                        <small>
                                        🎯 Cosine Similarity: {sim['cosine_similarity']:.3f}<br>
                                        📏 Euclidean Distance: {sim['euclidean_distance']:.3f}
                                        </small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Benzer ses çalma
                            st.markdown("### 🎵 Benzer Ses Çalar")
                            similar_options = [sim['filename'] for sim in similar_sounds if sim['filename'] != selected_file]
                            if similar_options:
                                similar_audio = st.selectbox(
                                    "Benzer sesleri dinleyin:",
                                    options=similar_options,
                                    key="similar_audio_player"
                                )
                                
                                if similar_audio and similar_audio in st.session_state.audio_cache:
                                    st.audio(st.session_state.audio_cache[similar_audio], format='audio/wav')
                        
                        with col2:
                            # PCA görselleştirmesi
                            pca_data = classifier.get_pca_visualization_data(
                                target_features, target_class
                            )
                            
                            if pca_data[0] is not None:
                                pca_features, pca_names, variance_ratio, pca_obj = pca_data
                                
                                st.subheader(f"📍 PCA Haritası ({target_class})")
                                
                                fig_pca = go.Figure()
                                
                                # Referans sesleri
                                ref_indices = [i for i, name in enumerate(pca_names) if name != 'Yüklenen Ses']
                                if ref_indices:
                                    fig_pca.add_trace(go.Scatter(
                                        x=pca_features[ref_indices, 0],
                                        y=pca_features[ref_indices, 1],
                                        mode='markers+text',
                                        text=[pca_names[i] for i in ref_indices],
                                        textposition="top center",
                                        marker=dict(color='lightblue', size=10),
                                        name='Referans Sesler'
                                    ))
                                
                                # Hedef ses
                                target_idx = pca_names.index('Yüklenen Ses')
                                fig_pca.add_trace(go.Scatter(
                                    x=[pca_features[target_idx, 0]],
                                    y=[pca_features[target_idx, 1]],
                                    mode='markers+text',
                                    text=['🎯 Seçilen Ses'],
                                    textposition="top center",
                                    marker=dict(color='red', size=15, symbol='star'),
                                    name='Seçilen Ses'
                                ))
                                
                                fig_pca.update_layout(
                                    title=f"PCA Görselleştirmesi<br>Açıklanan Varyans: PC1={variance_ratio[0]:.2%}, PC2={variance_ratio[1]:.2%}",
                                    xaxis_title="1. Ana Bileşen",
                                    yaxis_title="2. Ana Bileşen",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_pca, use_container_width=True)
                            else:
                                st.info(f"PCA görselleştirmesi için {target_class} sınıfından en az 2 ses gerekli.")
                    else:
                        st.info(f"Henüz {target_class} sınıfından başka ses yok.")
            else:
                st.info("Henüz hiç ses dosyası yüklenmemiş.")
    
    with tab2:
        st.subheader("🎯 Tek Dosya Benzerlik Analizi")
        st.write("Tek bir ses dosyası yükleyip mevcut veritabanındaki seslerle karşılaştırın.")
        
        # Tek dosya için gelişmiş işleme seçenekleri
        with st.expander("⚙️ Gelişmiş İşleme Ayarları"):
            col1, col2, col3 = st.columns(3)
            with col1:
                use_vad_single = st.checkbox("🔇 Sessizlik Temizleme (VAD)", value=True, 
                                           help="Uzun ses dosyalarından sessiz bölümleri temizler", key="vad_single")
            with col2:
                use_segmentation_single = st.checkbox("✂️ Ses Segmentasyonu", value=True, 
                                                    help="Çok uzun sesleri parçalara böler", key="seg_single")
            with col3:
                use_first_pattern_single = st.checkbox("🎼 İlk Pattern Tespiti", value=True, 
                                                      help="Tekrar eden seslerden sadece ilk karakteristik bölümü alır", key="pattern_single")
        
        # Tek dosya yükleme
        single_file = st.file_uploader(
            "🎧 Analiz edilecek ses dosyasını yükleyin",
            type=['wav'],
            help="Bu ses, mevcut veritabanındaki seslerle karşılaştırılacak",
            key="single_upload"
        )
        
        if single_file:
            if len(st.session_state.processed_files) == 0:
                st.warning("Önce 'Toplu Yükleme' sekmesinden referans sesler yüklemeniz gerekiyor!")
            else:
                with st.spinner("Ses analiz ediliyor..."):
                    # Geçici dosya oluştur
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(single_file.read())
                        tmp_file.flush()
                        
                        # Gelişmiş sınıflandırma (VAD + Segmentasyon + İlk Pattern)
                        predicted_class, confidence, features = classifier.predict_single_enhanced(
                            tmp_file.name, use_vad=use_vad_single, use_segmentation=use_segmentation_single,
                            use_first_pattern=use_first_pattern_single
                        )
                        
                        # Geçici dosyayı sil
                        os.unlink(tmp_file.name)
                
                if predicted_class is not None:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Sınıflandırma Sonucu")
                        st.markdown(f"""
                        **Dosya:** {single_file.name}  
                        **Tahmin:** {predicted_class}  
                        **Güven:** {confidence:.3f}
                        """)
                        
                        # Yüklenen tek dosyayı çal
                        st.markdown("### 🎧 Yüklenen Ses")
                        st.audio(single_file.getvalue(), format='audio/wav')
                        
                        # Benzer sesleri bul
                        similar_sounds = classifier.find_similar_sounds(
                            features, predicted_class, top_k=10
                        )
                        
                        if similar_sounds:
                            st.subheader(f"🎯 En Benzer Sesler ({predicted_class})")
                            
                            for i, sim in enumerate(similar_sounds):
                                st.markdown(f"""
                                <div class="similarity-card">
                                    <strong>{i+1}. {sim['filename']}</strong><br>
                                    <small>
                                    🎯 Cosine Similarity: {sim['cosine_similarity']:.3f}<br>
                                    📏 Euclidean Distance: {sim['euclidean_distance']:.3f}
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Benzer ses çalma
                            st.markdown("### 🎵 Benzer Ses Çalar")
                            similar_audio_single = st.selectbox(
                                "Benzer sesleri dinleyin:",
                                options=[sim['filename'] for sim in similar_sounds],
                                key="similar_audio_player_single"
                            )
                            
                            if similar_audio_single and similar_audio_single in st.session_state.audio_cache:
                                st.audio(st.session_state.audio_cache[similar_audio_single], format='audio/wav')
                        else:
                            st.info(f"Veritabanında {predicted_class} sınıfından başka ses bulunamadı.")
                    
                    with col2:
                        # PCA görselleştirmesi
                        pca_data = classifier.get_pca_visualization_data(
                            features, predicted_class
                        )
                        
                        if pca_data[0] is not None:
                            pca_features, pca_names, variance_ratio, pca_obj = pca_data
                            
                            st.subheader(f"📍 PCA Haritası ({predicted_class})")
                            
                            fig_pca = go.Figure()
                            
                            # Referans sesleri
                            ref_indices = [i for i, name in enumerate(pca_names) if name != 'Yüklenen Ses']
                            if ref_indices:
                                fig_pca.add_trace(go.Scatter(
                                    x=pca_features[ref_indices, 0],
                                    y=pca_features[ref_indices, 1],
                                    mode='markers+text',
                                    text=[pca_names[i] for i in ref_indices],
                                    textposition="top center",
                                    marker=dict(color='lightblue', size=10),
                                    name='Referans Sesler'
                                ))
                            
                            # Hedef ses
                            target_idx = pca_names.index('Yüklenen Ses')
                            fig_pca.add_trace(go.Scatter(
                                x=[pca_features[target_idx, 0]],
                                y=[pca_features[target_idx, 1]],
                                mode='markers+text',
                                text=['🎯 Analiz Edilen Ses'],
                                textposition="top center",
                                marker=dict(color='red', size=15, symbol='star'),
                                name='Analiz Edilen Ses'
                            ))
                            
                            fig_pca.update_layout(
                                title=f"PCA Görselleştirmesi<br>Açıklanan Varyans: PC1={variance_ratio[0]:.2%}, PC2={variance_ratio[1]:.2%}",
                                xaxis_title="1. Ana Bileşen",
                                yaxis_title="2. Ana Bileşen",
                                height=500
                            )
                            
                            st.plotly_chart(fig_pca, use_container_width=True)
                        else:
                            st.info(f"PCA görselleştirmesi için {predicted_class} sınıfından en az 2 ses gerekli.")
                else:
                    st.error("Ses dosyası işlenemedi.")
    
    # Sidebar bilgi
    with st.sidebar:
        st.markdown("### 📋 Bilgiler")
        st.markdown("""
        **Özellikler:**
        - 🎵 7 sınıf ses tanıma
        - 🎧 Ses çalma özelliği
        - 🔍 Cosine similarity analizi
        - 📏 Euclidean distance hesaplama
        - 📊 PCA görselleştirmesi
        - 📈 İstatistiksel analizler
        - 🔇 **YENİ!** Sessizlik temizleme (VAD)
        - ✂️ **YENİ!** Akıllı ses segmentasyonu
        - 🎼 **YENİ!** İlk pattern tespiti
        
        **Teknik Detaylar:**
        - 42 ses özelliği
        - MFCC, RMS, ZCR, Spektral özellikler
        - TensorFlow/Keras modeli
        - Scikit-learn ön işleme
        """)
        
        # Model istatistikleri
        if len(st.session_state.processed_files) > 0:
            st.markdown("### 📊 Yüklenen Sesler")
            class_counts = {}
            for file_info in st.session_state.processed_files:
                class_name = file_info['predicted_class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                st.markdown(f"**{class_name}:** {count} ses")
            
            st.markdown(f"**Toplam:** {len(st.session_state.processed_files)} ses")

if __name__ == "__main__":
    main() 