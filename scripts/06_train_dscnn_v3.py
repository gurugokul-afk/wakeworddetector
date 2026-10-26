"""
06_train_dscnn_v3.py
Trains an improved Depthwise Separable CNN (DS-CNN v3) for 3-class KWS:
  0: BANKAI (Wake Word)
  1: UNKNOWN (Non-keyword Speech)
  2: NOISE (Silence, Fan, HVAC, Ambience)

Features:
- Input: 40 x 101 Log-Mel spectrogram
- Residual skip connections in depthwise-separable blocks
- SpecAugment (time & frequency masking) on training data
- L2 weight regularization + Dropout to prevent overfitting
- Balanced class weights with label smoothing
- Model checkpointing & early stopping
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "split"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["BANKAI", "UNKNOWN", "NOISE"]
CLASS_TO_ID = {"bankai": 0, "unknown": 1, "noise": 2}
NUM_CLASSES = 3
INPUT_SHAPE = (40, 101, 1)

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
SEED = 20260909

np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_split_data(split_name):
    X = []
    y = []
    print(f"\nLoading {split_name.upper()} split...")
    for class_name, class_id in CLASS_TO_ID.items():
        folder = SPLIT_DIR / split_name / class_name
        files = sorted(folder.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No .npy files in {folder}")
        print(f"  {class_name.upper():<8}: {len(files)} files")
        for f in files:
            feat = np.load(f).astype(np.float32)
            if feat.shape != (40, 101):
                raise ValueError(f"Unexpected shape {feat.shape} in {f}")
            X.append(feat)
            y.append(class_id)
            
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.int32)
    return X, y

def spec_augment(x, max_time_mask=10, max_freq_mask=5):
    """Applies random time and frequency masking on Mel spectrogram."""
    x = x.copy()
    # Freq mask (axis 0: 40 bins)
    f = np.random.randint(0, max_freq_mask)
    f0 = np.random.randint(0, 40 - f)
    x[f0:f0+f, :, :] = 0.0
    # Time mask (axis 1: 101 frames)
    t = np.random.randint(0, max_time_mask)
    t0 = np.random.randint(0, 101 - t)
    x[:, t0:t0+t, :] = 0.0
    return x

def make_dataset(X, y, is_training=True):
    def augment_fn(feat, label):
        # 50% chance of SpecAugment
        if tf.random.uniform(()) > 0.5:
            # Frequency masking
            f = tf.random.uniform((), 0, 5, dtype=tf.int32)
            f0 = tf.random.uniform((), 0, 40 - 5, dtype=tf.int32)
            mask_f = tf.concat([
                tf.ones([f0, 101, 1]),
                tf.zeros([f, 101, 1]),
                tf.ones([40 - f0 - f, 101, 1])
            ], axis=0)
            feat = feat * mask_f
            
            # Time masking
            t = tf.random.uniform((), 0, 10, dtype=tf.int32)
            t0 = tf.random.uniform((), 0, 101 - 10, dtype=tf.int32)
            mask_t = tf.concat([
                tf.ones([40, t0, 1]),
                tf.zeros([40, t, 1]),
                tf.ones([40, 101 - t0 - t, 1])
            ], axis=1)
            feat = feat * mask_t
        return feat, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_dscnn_v3():
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="logmel_input")
    
    # 1. Standard Conv layer (stride 2)
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias=False, kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 2. Depthwise Separable Block 1 (stride 1 with residual skip)
    res = x
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(24, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Add()([x, res])
    
    # 3. Depthwise Separable Block 2 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(36, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 4. Depthwise Separable Block 3 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 5. Depthwise Separable Block 4 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Global pooling & classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BANKAI_DS_CNN_V3")
    return model

def train_model():
    X_train, y_train = load_split_data("train")
    X_val, y_val = load_split_data("val")
    X_test, y_test = load_split_data("test")
    
    print("\n" + "="*50)
    print("TRAINING DATASET DISTRIBUTION")
    print("="*50)
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<8}: {np.sum(y_train == cid)}")
        
    # Anti-false-activation weights: Penalize false alarms on UNKNOWN and NOISE strictly
    weights = {
        0: 0.90,  # BANKAI (wake word)
        1: 1.15,  # UNKNOWN (penalize false alarms on speech)
        2: 1.15   # NOISE (penalize false alarms on background)
    }
    print("\nAnti-false-alarm class weights applied:")
    for cid, w in weights.items():
        print(f"  {CLASS_NAMES[cid]:<8}: {w:.4f}")

        
    model = build_dscnn_v3()
    model.summary()
    print("Trainable parameters:", model.count_params())
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    best_model_path = MODEL_DIR / "bankai_dscnn_v3_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    train_ds = make_dataset(X_train, y_train, is_training=True)
    val_ds = make_dataset(X_val, y_val, is_training=False)
    
    print("\nStarting model training...", flush=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1
    )

    
    # Save training history
    hist_path = RESULT_DIR / "training_history_v3.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                f"{history.history['loss'][i]:.6f}",
                f"{history.history['accuracy'][i]:.6f}",
                f"{history.history['val_loss'][i]:.6f}",
                f"{history.history['val_accuracy'][i]:.6f}"
            ])
            
    # Evaluate best model on test split
    print("\nLoading best model for test evaluation...")
    best_model = tf.keras.models.load_model(str(best_model_path))
    probs = best_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    
    print("\n" + "="*50)
    print("DS-CNN V3 FINAL TEST EVALUATION")
    print("="*50)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nConfusion Matrix:")
    print("              BANKAI  UNKNOWN  NOISE")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<12} {row[0]:>6} {row[1]:>8} {row[2]:>6}")
    print("\nClassification Report:\n", report)
    
    # Save test results
    report_path = RESULT_DIR / "classification_report_v3.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ISRO_GEMINI DS-CNN V3 TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    cm_path = RESULT_DIR / "confusion_matrix_v3.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "pred_BANKAI", "pred_UNKNOWN", "pred_NOISE"])
        for i, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[i], row[0], row[1], row[2]])
            
    print(f"\nSaved best model: {best_model_path}")
    print(f"Saved report: {report_path}")

if __name__ == "__main__":
    train_model()
