#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import tempfile
import librosa
import sys

# Constants
MIN_CONFIDENCE = 0.85
VALID_EMOTIONS = {'angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral'}
MODEL_PATH = 'saved_models/emotion_classifier_v6_enhanced_final.keras'
ENCODER_PATH = 'saved_models/emotion_classifier_v6_enhanced_encoder.pkl'

# Feature extraction
def get_audio_features(audio_file, target_sr=16000, duration=3.0):
    try:
        sound_data, sample_rate = librosa.load(audio_file, sr=target_sr, duration=duration)
        if len(sound_data) == 0:
            return np.zeros((32, 94, 2))

        required_samples = int(duration * target_sr)
        if len(sound_data) < required_samples:
            sound_data = np.pad(sound_data, (0, required_samples - len(sound_data)), mode='constant')

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=sound_data, sr=sample_rate, n_mels=32)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = resize_array(mel_db, (32, 94))
        mel_db = normalize_features(mel_db)

        # MFCC + Chroma + Spectral Contrast
        mfcc = librosa.feature.mfcc(y=sound_data, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=sound_data, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=sound_data, sr=sample_rate)
        features = np.vstack([mfcc, chroma, contrast])
        features = resize_array(features, (32, 94))
        features = normalize_features(features)

        return np.stack([mel_db, features], axis=-1)

    except:
        return np.zeros((32, 94, 2))

def resize_array(input_array, target_size):
    output = np.zeros(target_size)
    h, w = min(target_size[0], input_array.shape[0]), min(target_size[1], input_array.shape[1])
    output[:h, :w] = input_array[:h, :w]
    return output

def normalize_features(x):
    mean, std = np.mean(x), np.std(x)
    return (x - mean) / std if std > 0 else x - mean

def validate_audio_file(audio_path):
    return os.path.exists(audio_path) and audio_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))

def predict_emotion(audio_path):
    try:
        if not validate_audio_file(audio_path):
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': 'Invalid audio format or missing file.'
            }

        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': 'Model or encoder file missing.'
            }

        model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)

        features = get_audio_features(audio_path)
        if np.all(features == 0):
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': 'Feature extraction failed.'
            }

        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = float(predictions[0][idx])
        emotion = encoder.inverse_transform([idx])[0]

        if confidence < MIN_CONFIDENCE:
            return {
                'status': 'rejected',
                'emotion': None,
                'confidence': confidence,
                'message': f'Low confidence ({confidence:.2f}) — possibly rejected emotion.'
            }

        if emotion not in VALID_EMOTIONS:
            return {
                'status': 'rejected',
                'emotion': None,
                'confidence': confidence,
                'message': f'Detected emotion "{emotion}" is not among valid targets.'
            }

        return {
            'status': 'success',
            'emotion': emotion,
            'confidence': confidence,
            'message': 'Prediction successful'
        }

    except Exception as e:
        return {
            'status': 'error',
            'emotion': None,
            'confidence': None,
            'message': f'Prediction failed: {str(e)}'
        }

# ==== Streamlit UI ====
st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.title("🎵 Audio Emotion Classifier")
st.markdown("Upload a speech or song `.wav` file to detect the emotion.")

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3', 'ogg', 'flac'])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format='audio/wav')
    st.write("🎧 Audio loaded. Classifying...")

    result = predict_emotion(tmp_path)

    if result['status'] == 'success':
        st.success(f"🎭 Emotion: **{result['emotion']}**\n\n✅ Confidence: **{result['confidence']:.2f}**")
    elif result['status'] == 'rejected':
        st.warning(f"⚠️ Prediction Rejected: {result['message']}\n\nConfidence: {result['confidence']:.2f}")
    else:
        st.error(f"❌ Error: {result['message']}")


# In[ ]:




