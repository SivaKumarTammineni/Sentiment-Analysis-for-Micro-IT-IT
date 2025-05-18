import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 1. Load dataset
df = pd.read_csv("/kaggle/input/sentiment/Equal.csv", encoding="latin1")
df.dropna(subset=["Review", "Sentiment"], inplace=True)
df.drop_duplicates(inplace=True)

# 2. Label encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("✅ Label Mapping:", label_map)

# 3. Tokenization
texts = df['Review'].values
labels = df['label'].values
num_classes = len(np.unique(labels))

tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = 100
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 4. Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_per_fold = []

# Directory to save best models
os.makedirs("fold_models", exist_ok=True)

# 5. Training Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(padded_sequences, labels), start=1):
    print(f"\n🔵 Training Fold {fold}...")

    X_train, X_val = padded_sequences[train_idx], padded_sequences[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # 6. Model Architecture


    model = Sequential([
    Embedding(input_dim=30000, output_dim=200, input_length=max_len),

    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
    LayerNormalization(),

    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Callbacks
    checkpoint_path = f"fold_models/bilstm_fold_{fold}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

    # 7. Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # 8. Evaluation
    val_preds = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, val_preds)
    acc_per_fold.append(acc)

    print(f"✅ Fold {fold} Accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds, target_names=label_encoder.classes_))

# 9. Summary
print(f"\n✅ All folds completed.")
print("📊 Fold Accuracies:", [f"{a:.4f}" for a in acc_per_fold])
print(f"🏁 Average Accuracy: {np.mean(acc_per_fold):.4f}")

# 10. Save tokenizer and label map
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Tokenizer and label map saved as pickle.")
