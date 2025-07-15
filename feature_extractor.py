import numpy as np
import librosa
from scipy.stats import skew, kurtosis
import warnings

# Genel ayarlar
SR = 22_050
N_FFT = 1024
HOP_LENGTH = 128
N_MEL = 128
N_MFCC = 20
ROLL_PERCENT = 0.85
EPS = 1e-10

def log_attack_features(env, sr, hop):
    if env.max() < EPS:
        return 0.0, 0.0
    peak_idx = np.argmax(env)
    env_norm = env / env.max() + EPS
    try:
        t10 = np.where(env_norm >= 0.10)[0][0]
        t90 = np.where(env_norm >= 0.90)[0][0]
    except IndexError:
        return (peak_idx * hop) / sr, 0.0
    attack_time = (t90 - t10) * hop / sr
    if attack_time < EPS:
        return attack_time, 0.0
    env_db = librosa.amplitude_to_db(env_norm, ref=1.0)
    slope = (env_db[t90] - env_db[t10]) / attack_time
    return attack_time, slope

def hpss_energy_ratio(y):
    y_h, y_p = librosa.effects.hpss(y)
    return np.sum(y_p**2) / (np.sum(y_h**2) + EPS)

def extract_features(signal, sr=SR):
    """42 özellik çıkaran fonksiyon"""
    warnings.filterwarnings("ignore")
    
    if len(signal) < N_FFT:
        signal = np.pad(signal, (0, N_FFT - len(signal)))
    
    S = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MEL)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)

    # RMS
    rms = librosa.feature.rms(
        S=S,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, 
                                               roll_percent=ROLL_PERCENT)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    flux = librosa.onset.onset_strength(S=S, sr=sr, hop_length=HOP_LENGTH)
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr,
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH)
    atk_time, atk_slope = log_attack_features(onset_env, sr, HOP_LENGTH)

    feats = {
        **{f"mfcc{i+1:02d}": mfcc[i].mean() for i in range(N_MFCC)},
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr),
        "centroid_mean": np.mean(centroid),
        "bandwidth_mean": np.mean(bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "flatness_mean": np.mean(flatness),
        "flux_mean": np.mean(flux),
        **{f"contrast_b{b+1}": contrast[b].mean() for b in range(contrast.shape[0])},
        "onset_mean": onset_env.mean(),
        "onset_std": onset_env.std(),
        "onset_max": onset_env.max(),
        "onset_sum": onset_env.sum(),
        "attack_time": atk_time,
        "attack_slope": atk_slope,
        "hpi_ratio": hpss_energy_ratio(signal),
    }
    return feats

def extract_from_file(file_path):
    """Dosyadan özellik çıkarma"""
    try:
        signal, _ = librosa.load(file_path, sr=SR)
        return extract_features(signal)
    except Exception as e:
        print(f"Hata: {file_path}: {e}")
        return None 