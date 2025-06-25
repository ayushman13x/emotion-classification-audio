# emotion-classification-audio
 🎧 Emotion Classification from Audio using Deep Learning

This project is focused on classifying human emotions from short speech and song audio clips using deep learning. It uses Mel spectrograms and handcrafted features from audio signals, and is trained using a CNN model. The solution includes a Streamlit web app to upload audio and receive the classified emotion.



 📁 Project Structure
├──ipynb file containing all the code
├── test_model.py # Script to test model using audio file
├── saved_models/ # Contains trained model and encoder
│ ├── emotion_classifier_v6_enhanced_final.keras
│ └── encoder_v6.pkl
├── demo video # Sample audio files for demo
└── README.md # Project documentation

 🔁 Workflow

1. Data Loading: Load speech and song audio files with emotion labels.
2. Filtering: Remove `sad` and `surprised` emotion classes.
3. Feature Extraction: Use MFCCs or Mel spectrograms for every audio file.
4. Train-Validation Split**: Stratified split for balanced class distribution.
5. Label Encoding: Convert categorical emotions to numeric format.
6. Model Building: Use a CNN-based architecture with Conv2D, pooling, and dropout.
7. Training: Include class weights and callbacks (e.g., EarlyStopping, ModelCheckpoint).
8. Evaluation: Use accuracy, F1, F2, precision, recall, and confusion matrix.
9. Saving: Save best model, label encoder, class weights, and visualizations.


📂 Dataset Description

The dataset is composed of two sources:

- Speech data: Located in `Audio_Speech_Actors_01-24`
- Song data: Located in `Audio_Song_Actors_01-24`

 ⚙️ Preprocessing & Feature Engineering

The project uses a well-defined pipeline to transform raw audio files into clean, informative features ready for emotion classification. Below are the core steps involved:

---

 1. 🧹 Audio Cleaning & Normalization

- Resampling: All audio clips are resampled to a standard 16 kHz.
- Duration Control: Each audio is trimmed or zero-padded to exactly 3 seconds.
- Mono Conversion: Stereo signals are converted to mono.

These are handled inside the `get_audio_features()` function for consistency and reliability.

---

 2. 🎼 Feature Extraction

Two complementary feature channels are extracted:

 🔹 Mel Spectrogram (Channel 1)
- Computed with `librosa` using 32 mel filters, FFT size of 1024, and hop length of 512.
- Converted to dB scale using `power_to_db`.
- Resized to shape `(32, 94)` using a custom `resize_array()` function.
- Normalized using mean-std normalization (`normalize_features()`).

 🔹 Composite Audio Features (Channel 2)
- Combines:
  - MFCC (13 coefficients) and their first-order delta
  - Chroma Features and delta
  - Spectral Contrast and delta
- Vertically stacked and normalized to match the shape of the first channel.

These are stacked along the last axis to get a final shape of `(32, 94, 2)`.


 3. 🛠 Feature Normalization
    
- `normalize_features(array)`: Normalizes features to have 0 mean and unit variance for stability during training.


 5. 📊 Dataset Loading

The `load_audio_data()` function:
- Loads audio files from speech and song folders.
- Extracts emotion labels based on filename encoding (e.g., `03` → `happy`).
  

 6. ⚖️ Class Imbalance Handling

To address class imbalance in emotional datasets, the pipeline uses:

 ✅ Data Augmentation: `augment_dataset()`
- Generates new samples using:
  - Noise injection
  - Frequency masking
  - Time masking
  - Amplitude scaling
- Increases under-represented class size to the dataset’s median count.

 ✅ Class Weights: `compute_class_weights()`
- Computes per-class weights inversely proportional to frequency.
- Adjusts loss contribution of rare classes for better generalization.

 🧠 Model Architecture & Training Pipeline

The emotion classification model is built using a deep CNN-based architecture with regularization and pooling layers. It processes 3-second audio segments as input in the shape `(32, 94, 2)` — representing mel spectrogram and engineered features.

---

 📐 Model Structure: `build_classifier()`

The model follows a three-block ConvNet pipeline:

| Layer Type           | Parameters                      | Purpose                            |
|----------------------|----------------------------------|------------------------------------|
| `Conv2D`             | 32 filters, 3x3 kernel, ReLU     | Feature extraction                 |
| `BatchNormalization` |                                  | Speed up and stabilize training    |
| `MaxPooling2D`       | 2x2                              | Downsample spatial dimensions      |
| `Conv2D`             | 64 filters, 3x3 kernel, ReLU     | Deeper feature extraction          |
| `Dropout`            | 0.3                              | Prevent overfitting                |
| `Conv2D`             | 128 filters, 3x3 kernel, ReLU    | High-level representation          |
| `Dropout`            | 0.4                              | Additional regularization          |
| `GlobalAvgPooling2D` |                                  | Aggregate features                 |
| `Dense`              | 128 units, ReLU                  | Fully connected intermediate layer |
| `Dropout`            | 0.5                              | Strong regularization              |
| `Dense`              | Softmax, `num_classes` output    | Final emotion prediction           |

---

 🏋️ Training Setup

- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam` (LR = 0.001)
- Metrics: Accuracy
- Class Weights: Calculated dynamically to penalize underrepresented classes using `compute_class_weights()`.
- Callbacks:
  - `EarlyStopping` on `val_accuracy` (patience = 10)
  - `ReduceLROnPlateau` on `val_loss`
  - `ModelCheckpoint` to save best model (`*_best.keras`)

---

 ⚖️ Data Splits

| Dataset        | Count    | Notes                         |
|----------------|----------|-------------------------------|
| Training       | 80%      | Includes augmentation         |
| Validation     | 20%      | Unseen, for evaluation        |

Stratified split is used to preserve emotion class balance across train and val sets.

---

### 📈 Final Performance (on Validation Set)
Train Accuracy: 0.9859
Validation Accuracy: 0.7862

Accuracy: 0.7862
F1 (weighted): 0.7836
F1 (macro): 0.7880
F2 (weighted): 0.7846
F2 (macro): 0.7907
Precision: 0.7862
Recall: 0.7862


Also included:
- 📊 Per-class accuracy via confusion matrix.
- CLASS ACCURACY:
angry: 0.8667
calm: 0.8533
disgust: 0.9231
fearful: 0.7600
happy: 0.7867
neutral: 0.8684
sad: 0.6000
surprised: 0.6923
- 🧾 Detailed classification report with precision, recall, and F1 for each emotion.
 precision    recall  f1-score   support

       angry       0.86      0.87      0.86        75
        calm       0.80      0.85      0.83        75
     disgust       0.78      0.92      0.85        39
     fearful       0.74      0.76      0.75        75
       happy       0.80      0.79      0.79        75
     neutral       0.75      0.87      0.80        38
         sad       0.71      0.60      0.65        75
   surprised       0.87      0.69      0.77        39

    accuracy                           0.79       491
   macro avg       0.79      0.79      0.79       491
weighted avg       0.79      0.79      0.78       491
---
📉 Class Reduction Experiments

During training, it was observed that certain emotion classes — particularly `sad` and `surprised` — showed consistently low prediction confidence and poor per-class accuracy.

so first i drop "sad and then saw the metrics :-

RESULTS FOR emotion_classifier_v6_enhanced:
Train Accuracy: 0.9820
Validation Accuracy: 0.8341

SUMMARY:
Accuracy: 0.8341
F1 (weighted): 0.8326
F1 (macro): 0.8316
F2 (weighted): 0.8328
F2 (macro): 0.8349
Precision: 0.8368
Recall: 0.8341

CLASS ACCURACY:
angry: 0.8933
calm: 0.8667
disgust: 0.8718
fearful: 0.8800
happy: 0.6667
neutral: 0.9211
surprised: 0.7692

DETAILED REPORT:
              precision    recall  f1-score   support

       angry       0.84      0.89      0.86        75
        calm       0.93      0.87      0.90        75
     disgust       0.83      0.87      0.85        39
     fearful       0.81      0.88      0.85        75
       happy       0.81      0.67      0.73        75
     neutral       0.74      0.92      0.82        38
   surprised       0.86      0.77      0.81        39

    accuracy                           0.83       416
   macro avg       0.83      0.84      0.83       416
weighted avg       0.84      0.83      0.83       416

What I found astonishing during experimentation was that dropping the sad class actually decreased the accuracy of the happy class.

This behavior makes sense — sad and happy are acoustically contrasting emotions, and their presence helps the model learn stronger boundaries between positive and negative affective states. Removing sad likely blurred this distinction, making it harder for the model to correctly classify happy.

Then i drop surprised one time and saw the metrics :-

==================================================
RESULTS FOR emotion_classifier_v6_enhanced:
Train Accuracy: 0.9967
Validation Accuracy: 0.8075
==================================================

15/15 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step 
SUMMARY:
Accuracy: 0.8075
F1 (weighted): 0.8043
F1 (macro): 0.8025
F2 (weighted): 0.8054
F2 (macro): 0.8013
Precision: 0.8083
Recall: 0.8075

CLASS ACCURACY:
angry: 0.9467
calm: 0.8400
disgust: 0.6923
fearful: 0.7867
happy: 0.8800
neutral: 0.8421
sad: 0.6267

DETAILED REPORT:
              precision    recall  f1-score   support

       angry       0.83      0.95      0.88        75
        calm       0.83      0.84      0.83        75
     disgust       0.90      0.69      0.78        39
     fearful       0.80      0.79      0.79        75
       happy       0.80      0.88      0.84        75
     neutral       0.76      0.84      0.80        38
         sad       0.77      0.63      0.69        75

    accuracy                           0.81       452
   macro avg       0.81      0.80      0.80       452
weighted avg       0.81      0.81      0.80       452


Dropping the surprised class led to a noticeable drop in the accuracy of the disgust class, which was unexpected at first. However, on deeper inspection, this likely occurred because surprised and disgust share overlapping spectral patterns and intensity variations, especially in expressive vocal cues. Removing surprised reduced the diversity of such patterns during training, making it harder for the model to correctly differentiate disgust.

Now after dropping both the emotion i got result this result :-

final Evaluation Result 

RESULTS FOR emotion_classifier_v6_enhanced:
Train Accuracy: 1.0000
Validation Accuracy: 0.8621
==================================================

12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 
SUMMARY:
Accuracy: 0.8621
F1 (weighted): 0.8622
F1 (macro): 0.8565
F2 (weighted): 0.8614
F2 (macro): 0.8623
Precision: 0.8683
Recall: 0.8621

CLASS ACCURACY:
angry: 0.9200
calm: 0.8800
disgust: 0.8718
fearful: 0.8533
happy: 0.7600
neutral: 0.9211

DETAILED REPORT:
              precision    recall  f1-score   support

       angry       0.87      0.92      0.90        75
        calm       0.87      0.88      0.87        75
     disgust       0.81      0.87      0.84        39
     fearful       0.93      0.85      0.89        75
       happy       0.90      0.76      0.83        75
     neutral       0.73      0.92      0.81        38

    accuracy                           0.86       377
   macro avg       0.85      0.87      0.86       377
weighted avg       0.87      0.86      0.86       377


Confusion Matrix (Table Format):
               Pred: angry  Pred: calm  Pred: disgust  Pred: fearful  \
True: angry             66           0              3              3   
True: calm               0          70              0              0   
True: disgust            4           2             31              0   
True: fearful            1           2              1             67   
True: happy              5           5              1              9   
True: neutral            0           2              0              1   

               Pred: happy  Pred: neutral  
True: angry              3              0  
True: calm               1              4  
True: disgust            2              0  
True: fearful            3              1  
True: happy             52              3  
True: neutral            0             35  


i got my best model after dropping both the classes with the rest classes with accuracy more than around 85 to 90 except happy class and also f1 score of each class around greater than 80.


Saved Outputs

MODEL_PATH = "saved_models/emotion_classifier_v6_enhanced_final.keras"
ENCODER_PATH = "saved_models/emotion_classifier_v6_enhanced_encoder.pkl"

to use it we can use it like this :- 
model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)


 Handling Unknown Emotions (Sad & Surprised)
 
Since the model was trained on only 6 emotions (excluding sad and surprised), test samples containing these unknown classes may be wrongly classified with high confidence.

To prevent this, a minimum confidence threshold is applied during prediction. If the model's confidence is below this threshold, the prediction is rejected:

if confidence < MIN_CONFIDENCE:
    return {
        'status': 'rejected',
        'emotion': None,
        'confidence': confidence,
        'message': f'Low confidence ({confidence:.2f}). May be sad/surprised.'
    }

Calculation of Optimum Threshold

To avoid accepting overconfident wrong predictions on unseen emotions, we compute an optimal confidence threshold using the training data.

Process:

Predict all samples and group them into:

 Known classes (used in training)
 Unknown classes (sad, surprised)

From the unknown group, calculate the 95th percentile confidence and add a margin (+0.05).
Clip the final threshold between 0.60 and 0.85.
If no unknown samples exist, use a default threshold of 0.70.
This ensures only confident predictions are accepted and reduces misclassification on emotions the model wasn't trained on.

Optimum threshold =0.85

but i choose threshold=0.80 as I slightly lowered the threshold to 0.80 to detect more valid emotion classes, especially to better capture 'happy' instances, as their prediction confidence was often slightly below

Personal Reflections & Learnings

Gained hands-on experience with audio preprocessing and feature extraction (MFCCs, spectrograms).

Learned how to handle class imbalance through augmentation and class weights.

Understood the impact of dropping specific emotion classes on model generalization.like dropping sad emotion and how it affect happy emotion same goes with the case of surprised and disgust

Developed skill in model interpretability using confusion matrices and per-class accuracy analysis.

Explored confidence thresholding to handle unknown or excluded classes more reliably.


