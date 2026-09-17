"""Extração de vetores acústicos sem qualquer uso de rótulos."""
from __future__ import annotations

from typing import Iterable

import librosa
import numpy as np
import pywt
import soundfile as sf


def _statistics(values: np.ndarray, names: Iterable[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=float).ravel()
    result: dict[str, float] = {}
    for name in names:
        if name == "mean": result[name] = float(np.mean(values))
        elif name == "std": result[name] = float(np.std(values))
        elif name == "median": result[name] = float(np.median(values))
        elif name == "min": result[name] = float(np.min(values))
        elif name == "max": result[name] = float(np.max(values))
        elif name.startswith("p") and name[1:].isdigit(): result[name] = float(np.percentile(values, int(name[1:])))
        else: raise ValueError(f"Estatística temporal não reconhecida: {name}")
    return result


def _wavelet_features(signal: np.ndarray, duration_seconds: float, config: dict, statistics: list[str]) -> dict[str, float]:
    """Resumo multirresolução dos coeficientes DWT, um vetor por áudio."""
    wavelet, level = config["wavelet"], int(config["level"])
    maximum = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if maximum < 1:
        raise ValueError("áudio curto demais para DWT")
    coefficients = pywt.wavedec(signal, wavelet=wavelet, level=min(level, maximum))
    total_energy = sum(float(np.sum(np.square(coef))) for coef in coefficients) or 1.0
    result: dict[str, float] = {"duration_seconds": duration_seconds}
    for index, coefficient in enumerate(coefficients):
        band = f"approx_L{level}" if index == 0 else f"detail_L{level - index + 1}"
        energy = float(np.sum(np.square(coefficient)))
        result[f"{band}_relative_energy"] = energy / total_energy
        probabilities = np.square(coefficient) / (energy or 1.0)
        probabilities = probabilities[probabilities > 0]
        result[f"{band}_shannon_entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
        result.update({f"{band}_{stat}": value for stat, value in _statistics(coefficient, statistics).items()})
    return result


def extract_features(path: str, audio_config: dict, feature_config: dict, representation_config: dict | None = None) -> dict[str, float]:
    """Retorna um vetor por áudio, via STFT ou DWT, sem usar rótulos."""
    target_sr = int(audio_config["target_sample_rate"])
    # soundfile evita o overhead de librosa.load quando o arquivo já está na
    # taxa-alvo (caso predominante no Al Emadi: mono, 16 kHz).
    signal, source_sr = sf.read(path, dtype="float32", always_2d=True)
    if audio_config.get("mono", True):
        signal = signal.mean(axis=1)
    else:
        signal = signal[:, 0]
    if source_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=source_sr, target_sr=target_sr)
    sr = target_sr
    duration = audio_config.get("clip_duration_seconds")
    if duration:
        target_size = round(float(duration) * sr)
        signal = signal[:target_size]
        if len(signal) < target_size:
            signal = np.pad(signal, (0, target_size - len(signal)))
    if signal.size == 0:
        raise ValueError("arquivo de áudio vazio")

    representation_config = representation_config or {"type": "stft"}
    if representation_config.get("type", "stft") == "dwt":
        return _wavelet_features(signal, float(len(signal) / sr), representation_config, feature_config["temporal_statistics"])

    frame = max(32, round(float(audio_config["frame_length_ms"]) * sr / 1000))
    hop = max(1, round(float(audio_config["hop_length_ms"]) * sr / 1000))
    window = audio_config["window"]
    stats = feature_config["temporal_statistics"]
    stft = librosa.stft(signal, n_fft=frame, hop_length=hop, win_length=frame, window=window, center=True)
    magnitude = np.abs(stft)
    power = magnitude ** 2
    descriptors: dict[str, np.ndarray] = {
        "mfcc": librosa.feature.mfcc(S=librosa.power_to_db(power), sr=sr, n_mfcc=int(feature_config["n_mfcc"])),
    }
    if feature_config.get("include_delta", False):
        descriptors["delta_mfcc"] = librosa.feature.delta(descriptors["mfcc"])
    available = {
        # Ao passar o espectro pronto, librosa precisa receber o tamanho real
        # do frame; seu valor-padrão (2048) diverge do n_fft configurado.
        "rms": librosa.feature.rms(S=magnitude, frame_length=frame),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(signal, frame_length=frame, hop_length=hop),
        "spectral_centroid": librosa.feature.spectral_centroid(S=magnitude, sr=sr),
        "spectral_rolloff": librosa.feature.spectral_rolloff(S=magnitude, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
    }
    for name in feature_config.get("descriptors", []):
        descriptors[name] = available[name]

    result: dict[str, float] = {"duration_seconds": float(len(signal) / sr)}
    for name, matrix in descriptors.items():
        matrix = np.atleast_2d(matrix)
        for index, series in enumerate(matrix):
            prefix = f"{name}_{index:02d}" if matrix.shape[0] > 1 else name
            result.update({f"{prefix}_{stat}": value for stat, value in _statistics(series, stats).items()})
    return result
