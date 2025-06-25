#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import librosa
from typing import Dict, Optional


# In[22]:


MEL_FREQUENCIES = 32
TIME_STEPS = 94
FEATURE_CHANNELS = 2
MODEL_PATH = "saved_models/emotion_classifier_v6_enhanced_final.keras"
ENCODER_PATH = "saved_models/emotion_classifier_v6_enhanced_encoder.pkl"


# In[25]:


MIN_CONFIDENCE = 0.85
#optimal threshold i got  
MIN_CONFIDENCE=0.80 
# (to allow emotion to be classified more )
# Reject predictions below this threshold (tune as needed)


# In[26]:


VALID_EMOTIONS = {
    'angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral'
}


# In[27]:


def validate_audio_file(audio_path: str) -> bool:
    """Check if file exists and is a supported audio format."""
    return (os.path.exists(audio_path) and 
            audio_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')))


# In[28]:


def resize_array(input_array, target_size):
    if input_array.size == 0:
        return np.zeros(target_size)

    current_shape = input_array.shape

    # Adjust time dimension
    if len(current_shape) >= 2:
        if current_shape[1] > target_size[1]:
            input_array = input_array[:, :target_size[1]]
        elif current_shape[1] < target_size[1]:
            pad_width = [(0, 0), (0, target_size[1] - current_shape[1])]
            input_array = np.pad(input_array, pad_width, mode='constant')

    # Adjust frequency dimension
    if current_shape[0] > target_size[0]:
        input_array = input_array[:target_size[0], :]
    elif current_shape[0] < target_size[0]:
        pad_width = [(0, target_size[0] - current_shape[0]), (0, 0)]
        input_array = np.pad(input_array, pad_width, mode='constant')

    return input_array


# In[29]:


def normalize_features(feature_data):
    if feature_data.size == 0:
        return feature_data
    mean_value = np.mean(feature_data)
    std_value = np.std(feature_data)
    if std_value == 0 or np.isnan(std_value):
        return feature_data - mean_value
    return (feature_data - mean_value) / std_value


# In[30]:


def get_audio_features(audio_file, target_sr=16000, duration=3.0):
    try:
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"File not found: {audio_file}")

        sound_data, sample_rate = librosa.load(audio_file, sr=target_sr, duration=duration)

        required_samples = int(duration * target_sr)
        if len(sound_data) > required_samples:
            sound_data = sound_data[:required_samples]
        else:
            sound_data = np.pad(sound_data, (0, max(0, required_samples - len(sound_data))), mode='constant')

        fft_size = 1024
        hop_size = 512

        # Mel spectrogram
        mel_spectrum = librosa.feature.melspectrogram(
            y=sound_data, sr=sample_rate, n_mels=MEL_FREQUENCIES,
            n_fft=fft_size, hop_length=hop_size
        )
        mel_channel = librosa.power_to_db(mel_spectrum, ref=np.max)
        mel_channel = resize_array(mel_channel, (MEL_FREQUENCIES, TIME_STEPS))
        mel_channel = normalize_features(mel_channel)

        # Additional features
        mfcc = librosa.feature.mfcc(y=sound_data, sr=sample_rate, n_mfcc=13, n_fft=fft_size, hop_length=hop_size)
        chroma = librosa.feature.chroma_stft(y=sound_data, sr=sample_rate, hop_length=hop_size)
        spectral = librosa.feature.spectral_contrast(y=sound_data, sr=sample_rate, hop_length=hop_size)
        mfcc_d = librosa.feature.delta(mfcc)
        chroma_d = librosa.feature.delta(chroma)
        spectral_d = librosa.feature.delta(spectral)

        combined = np.vstack([mfcc, mfcc_d, chroma, chroma_d, spectral, spectral_d])
        feature_channel = resize_array(combined, (MEL_FREQUENCIES, TIME_STEPS))
        feature_channel = normalize_features(feature_channel)

        # Stack both channels
        final_feature = np.stack([mel_channel, feature_channel], axis=-1)
        return final_feature

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros((MEL_FREQUENCIES, TIME_STEPS, FEATURE_CHANNELS))


# In[31]:


def predict_emotion(audio_path: str) -> Dict[str, Optional[str]]:
    """
    Predict emotion from audio file with confidence checks.
    Returns:
        {
            'status': 'success'|'rejected'|'error',
            'emotion': str or None,
            'confidence': float or None,
            'message': str
        }
    """
    try:
        # === Input Validation ===
        if not validate_audio_file(audio_path):
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': f'Invalid audio file: {audio_path}'
            }

        # === Model Loading ===
        if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': 'Model files not found'
            }

        model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)

        # === Feature Extraction ===
        features = get_audio_features(audio_path)
        if np.all(features == 0):  # Check if feature extraction failed
            return {
                'status': 'error',
                'emotion': None,
                'confidence': None,
                'message': 'Feature extraction failed'
            }

        # === Prediction ===
        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_emotion = encoder.inverse_transform([predicted_index])[0]

        # === Confidence Check ===
        if confidence < MIN_CONFIDENCE:
            return {
                'status': 'rejected',
                'emotion': None,
                'confidence': confidence,
                'message': f'Low confidence ({confidence:.2f}). May be sad/surprised.'
            }

        # === Final Validation ===
        if predicted_emotion not in VALID_EMOTIONS:
            return {
                'status': 'rejected',
                'emotion': None,
                'confidence': confidence,
                'message': f'Invalid emotion detected: {predicted_emotion}'
            }

        return {
            'status': 'success',
            'emotion': predicted_emotion,
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

# ====== Main Execution ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <audio_file.wav>")
        sys.exit(1)
    
    result = predict_emotion(sys.argv[1])
    
    # Print formatted results
    print("\n" + "="*50)
    print(f"📁 Input File: {sys.argv[1]}")
    print(f"🔄 Status: {result['status'].upper()}")
    
    if result['status'] == 'success':
        print(f"🎭 Predicted Emotion: {result['emotion']}")
        print(f"✅ Confidence: {result['confidence']:.4f}")
    else:
        print(f"❌ Rejection Reason: {result['message']}")
        if result['confidence']:
            print(f"⚠️ Confidence Score: {result['confidence']:.4f}")
    
    print("="*50 + "\n")


# In[ ]:




