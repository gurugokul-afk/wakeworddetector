"""
06_train_dscnn_v4.py
Trains the improved Depthwise Separable CNN v4 on the full krishnaDataSet:
  0: BANKAI (Clean Multi-Speaker Wake Word)
  1: UNKNOWN (Speech Commands + Conversational Speech)
  2: NOISE (WhatsApp + Speech Commands silence + Synthetics)

Key enhancements:
- SpecAugment (time & frequency masking)
- Anti-false-alarm loss weighting (penalizing false alarms on speech & noise)
- Residual skip connections in depthwise-separable blocks
- Full test evaluation on 1,156 independent test samples
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
    print(f"\nLoading {split_name.upper()} split...", flush=True)
    for class_name, class_id in CLASS_TO_ID.items():
        folder = SPLIT_DIR / split_name / class_name
        files = sorted(folder.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No .npy files in {folder}")
        print(f"  {class_name.upper():<8}: {len(files)} files", flush=True)
        for f in files:
            feat = np.load(f).astype(np.float32)
            X.append(feat)
            y.append(class_id)
            
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.int32)
    return X, y

def make_dataset(X, y, is_training=True):
    def augment_fn(feat, label):
        if tf.random.uniform(()) > 0.5:
            # Frequency masking (axis 0: 40 bins)
            f = tf.random.uniform((), 0, 5, dtype=tf.int32)
            f0 = tf.random.uniform((), 0, 40 - 5, dtype=tf.int32)
            mask_f = tf.concat([
                tf.ones([f0, 101, 1]),
                tf.zeros([f, 101, 1]),
                tf.ones([40 - f0 - f, 101, 1])
            ], axis=0)
            feat = feat * mask_f
            
            # Time masking (axis 1: 101 frames)
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

def build_dscnn_v4():
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="logmel_input")
    
    # Conv 1
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias=False, kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Block 1 (stride 1 with residual skip)
    res = x
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(24, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Add()([x, res])
    
    # Block 2 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(36, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Block 3 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Block 4 (stride 2)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Output Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BANKAI_DS_CNN_V4")
    return model

def train():
    X_train, y_train = load_split_data("train")
    X_val, y_val = load_split_data("val")
    X_test, y_test = load_split_data("test")
    
    print("\n" + "="*50, flush=True)
    print("TRAINING DATASET DISTRIBUTION", flush=True)
    print("="*50, flush=True)
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<8}: {np.sum(y_train == cid)}", flush=True)
        
    # Strict anti-false-alarm weighting:
    # Penalize false alarms on speech & background noise
    weights = {
        0: 0.85,  # BANKAI (wake word)
        1: 1.25,  # UNKNOWN (penalize speech false alarms)
        2: 1.20   # NOISE (penalize background noise false alarms)
    }
    print("\nClass weights applied:", flush=True)
    for cid, w in weights.items():
        print(f"  {CLASS_NAMES[cid]:<8}: {w:.4f}", flush=True)
        
    model = build_dscnn_v4()
    model.summary()
    print("Total parameters:", model.count_params(), flush=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    best_model_path = MODEL_DIR / "bankai_dscnn_v4_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
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
    
    print("\nStarting Model v4 Training...", flush=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    hist_path = RESULT_DIR / "training_history_v4.csv"
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
            
    # Evaluate best model
    print("\nLoading best model v4 for test evaluation...", flush=True)
    best_model = tf.keras.models.load_model(str(best_model_path))
    probs = best_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    
    print("\n" + "="*50, flush=True)
    print("DS-CNN V4 FINAL TEST EVALUATION", flush=True)
    print("="*50, flush=True)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print("\nConfusion Matrix:\n              BANKAI  UNKNOWN  NOISE", flush=True)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<12} {row[0]:>6} {row[1]:>8} {row[2]:>6}", flush=True)
    print("\nClassification Report:\n", report, flush=True)
    
    # Save results
    report_path = RESULT_DIR / "classification_report_v4.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ISRO_GEMINI DS-CNN V4 TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + report)
        
    cm_path = RESULT_DIR / "confusion_matrix_v4.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "pred_BANKAI", "pred_UNKNOWN", "pred_NOISE"])
        for i, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[i], row[0], row[1], row[2]])
            
    print(f"\nSaved best model: {best_model_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)

if __name__ == "__main__":
    train()
