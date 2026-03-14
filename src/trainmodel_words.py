# src/trainword.py
# =============================================================
# WORD MODEL TRAINING  --  stable production version
# =============================================================
# Trains a dedicated LSTM model on word motion sequences.
# Input:  MP_Data_Words/{word}/{seq}/{frame}.npy
#         Each sequence = (30, 126)  two-hand keypoints
# Output: models/word_model.h5
#         models/word_actions.npy
#
# USAGE:
#   python src/trainword.py
#
# REQUIREMENTS:
#   - At least 40 sequences per word (80+ recommended)
#   - Run collect_words.py first to collect data
# =============================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from function import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.metrics          import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from keras.utils     import to_categorical
from keras.models    import Sequential, Model
from keras.layers    import (LSTM, Dense, Dropout, BatchNormalization,
                              Input, Bidirectional, GlobalAveragePooling1D,
                              Multiply, Reshape, Permute, Flatten,
                              Conv1D, MaxPooling1D)
from keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                              ModelCheckpoint)
from keras.regularizers import l2
import tensorflow as tf

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(ROOT, 'MP_Data_Words')
MODELS_PATH  = os.path.join(ROOT, 'models')
RESULTS_PATH = os.path.join(ROOT, 'results')
os.makedirs(MODELS_PATH,  exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

FEATURES = 126   # two hands x 63 each
# SEQ_LEN comes from function.py

# =============================================================
# 1. LOAD DATA
# =============================================================
print("\n" + "="*60)
print("  LOADING WORD SEQUENCES")
print("="*60)

sequences, labels, label_map = [], [], {}
available_words = []

for action in WORD_SIGNS:
    apath = os.path.join(DATA_PATH, action)
    if not os.path.exists(apath):
        print(f"  [SKIP] {action} -- no folder")
        continue

    seqs_for_action = []
    for seq_name in sorted(os.listdir(apath)):
        seq_dir = os.path.join(apath, seq_name)
        if not os.path.isdir(seq_dir): continue

        frames = []
        valid  = True
        for fn in range(SEQ_LEN):
            fpath = os.path.join(seq_dir, f'{fn}.npy')
            if not os.path.exists(fpath):
                valid = False; break
            frames.append(np.load(fpath))

        if not valid or len(frames) != SEQ_LEN:
            continue

        arr = np.array(frames, dtype=np.float32)   # (30, 126)

        # Skip all-zero sequences (no hand detected)
        if np.all(arr == 0):
            continue

        seqs_for_action.append(arr)

    if len(seqs_for_action) < 10:
        print(f"  [SKIP] {action} -- only {len(seqs_for_action)} sequences "
              f"(need >=10)")
        continue

    idx = len(available_words)
    label_map[action] = idx
    available_words.append(action)

    for arr in seqs_for_action:
        sequences.append(arr)
        labels.append(idx)

    print(f"  [OK]   {action:<20} {len(seqs_for_action)} sequences")

if len(available_words) < 2:
    print("\nERROR: Need at least 2 word classes to train.")
    print("Run: python scripts/collect_words.py")
    sys.exit(1)

X     = np.array(sequences,  dtype=np.float32)   # (N, 30, 126)
y_raw = np.array(labels,     dtype=np.int32)

N_CLASSES = len(available_words)
print(f"\n  Classes  : {N_CLASSES}")
print(f"  Total    : {len(X)} sequences")
print(f"  Shape    : {X.shape}")

# =============================================================
# 2. AUGMENTATION  (doubles the dataset for free)
# =============================================================
print("\n  Augmenting data...")

def augment_sequence(seq):
    """
    Returns 3 augmented copies of a (SEQ_LEN, 126) sequence.
    Every returned array is exactly SEQ_LEN frames -- no shape mismatch.
    """
    T = seq.shape[0]   # always SEQ_LEN
    out = []

    # 1. Time jitter via np.roll -- wraps cleanly, always length T
    shift = np.random.randint(-3, 4)
    out.append(np.roll(seq, shift, axis=0).astype(np.float32))

    # 2. Scale jitter: multiply all keypoints by a random factor near 1.0
    scale = np.random.uniform(0.88, 1.12)
    out.append((seq * scale).astype(np.float32))

    # 3. Gaussian noise on keypoints
    noise = np.random.normal(0, 0.006, seq.shape).astype(np.float32)
    out.append(np.clip(seq + noise, -3.0, 3.0).astype(np.float32))

    # Sanity check: every augmented sequence must be (T, 126)
    for i, a in enumerate(out):
        assert a.shape == seq.shape,             f"Augmentation {i} shape {a.shape} != {seq.shape}"

    return out

aug_seqs, aug_labels = [], []
for seq, lbl in zip(X, y_raw):
    for a in augment_sequence(seq):
        aug_seqs.append(a)
        aug_labels.append(lbl)

X_aug     = np.array(aug_seqs,   dtype=np.float32)
y_aug_raw = np.array(aug_labels, dtype=np.int32)

X_all     = np.concatenate([X, X_aug], axis=0)
y_all_raw = np.concatenate([y_raw, y_aug_raw], axis=0)

print(f"  After augment: {len(X_all)} sequences")

# =============================================================
# 3. SPLIT
# =============================================================
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_all, y_all_raw, test_size=0.25,
    random_state=42, stratify=y_all_raw)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50,
    random_state=42, stratify=y_tmp)

y_tr_cat  = to_categorical(y_tr,  N_CLASSES)
y_val_cat = to_categorical(y_val, N_CLASSES)
y_te_cat  = to_categorical(y_te,  N_CLASSES)

# Class weights to handle imbalance
cw = compute_class_weight('balanced',
                           classes=np.arange(N_CLASSES), y=y_tr)
class_weights = dict(enumerate(cw))

print(f"\n  Train : {len(X_tr)}")
print(f"  Val   : {len(X_val)}")
print(f"  Test  : {len(X_te)}")

# =============================================================
# 4. MODEL
# =============================================================
# Architecture: Conv1D feature extraction + Bidirectional LSTM
# Conv1D extracts local motion patterns (bigrams of frames)
# BiLSTM reads the whole sequence in both directions
# Attention weights the most important frames
# =============================================================
print("\n" + "="*60)
print("  BUILDING MODEL")
print("="*60)

def build_model(seq_len, n_features, n_classes):
    inp = Input(shape=(seq_len, n_features), name='input')

    # ── Conv branch: local motion patterns ──
    x = Conv1D(64, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # ── BiLSTM: temporal sequence ──
    x = Bidirectional(LSTM(128, return_sequences=True,
                            kernel_regularizer=l2(0.001)))(x)
    x = Dropout(0.30)(x)
    x = Bidirectional(LSTM(64, return_sequences=True,
                            kernel_regularizer=l2(0.001)))(x)
    x = Dropout(0.30)(x)

    # ── Attention: focus on peak-motion frames ──
    # score each timestep
    attn = Dense(1, activation='tanh')(x)               # (batch, T, 1)
    attn = Flatten()(attn)                               # (batch, T)
    attn = tf.keras.layers.Activation('softmax')(attn)   # normalise
    attn = Reshape((seq_len, 1))(attn)                   # (batch, T, 1)
    x    = Multiply()([x, attn])                         # weight frames
    x    = GlobalAveragePooling1D()(x)                   # (batch, 128)

    # ── Classifier ──
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.40)(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

model = build_model(SEQ_LEN, FEATURES, N_CLASSES)
model.summary()

# =============================================================
# 5. TRAIN
# =============================================================
print("\n" + "="*60)
print("  TRAINING")
print("="*60)

ckpt_path = os.path.join(MODELS_PATH, 'word_model_best.h5')

callbacks = [
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        ckpt_path,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=0,
        mode='max'
    )
]

history = model.fit(
    X_tr, y_tr_cat,
    validation_data=(X_val, y_val_cat),
    epochs=200,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# =============================================================
# 6. EVALUATE
# =============================================================
print("\n" + "="*60)
print("  EVALUATION")
print("="*60)

_, tr_acc  = model.evaluate(X_tr,  y_tr_cat,  verbose=0)
_, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
_, te_acc  = model.evaluate(X_te,  y_te_cat,  verbose=0)

print(f"\n  Train accuracy : {tr_acc*100:.2f}%")
print(f"  Val   accuracy : {val_acc*100:.2f}%")
print(f"  Test  accuracy : {te_acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

print(f"\n{classification_report(y_te, y_pred, target_names=available_words)}")

print(f"\n{'Word':<20} {'Acc':>8}  {'Status'}")
print("-" * 40)
weak = []
for i, word in enumerate(available_words):
    mask = y_te == i
    if mask.sum() == 0: continue
    acc  = (y_pred[mask] == i).mean() * 100
    flag = "OK" if acc >= 85 else "WARN" if acc >= 70 else "WEAK"
    print(f"  {word:<18} {acc:>6.1f}%  {flag}")
    if acc < 70:
        weak.append(word)

if weak:
    print(f"\n  WEAK classes (collect more data): {weak}")

# =============================================================
# 7. PLOTS
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['categorical_accuracy'], label='Train')
axes[0].plot(history.history['val_categorical_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].set_ylim(0,1)
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend()
plt.tight_layout()
curve_path = os.path.join(RESULTS_PATH, 'word_training.png')
plt.savefig(curve_path, dpi=120); plt.close()
print(f"\n  Saved: {curve_path}")

if N_CLASSES <= 60:
    cm  = confusion_matrix(y_te, y_pred)
    fig = plt.figure(figsize=(max(12, N_CLASSES//2), max(10, N_CLASSES//2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=available_words,
                yticklabels=available_words,
                annot_kws={"size": max(5, 9-N_CLASSES//10)})
    plt.title(f'Word Model Confusion Matrix  |  Test {te_acc*100:.2f}%')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_PATH, 'word_confusion.png')
    plt.savefig(cm_path, dpi=100); plt.close()
    print(f"  Saved: {cm_path}")

# =============================================================
# 8. SAVE FINAL MODEL
# =============================================================
out_model = os.path.join(MODELS_PATH, 'word_model.h5')
out_acts  = os.path.join(MODELS_PATH, 'word_actions.npy')

model.save(out_model)
np.save(out_acts, np.array(available_words))

print(f"\n  Saved model   : {out_model}")
print(f"  Saved actions : {out_acts}")
print(f"\n  Classes trained: {available_words}")

print("\n" + "="*60)
print(f"  DONE.  Test accuracy: {te_acc*100:.2f}%")
if te_acc >= 0.90:
    print("  EXCELLENT -- ready to use!")
elif te_acc >= 0.80:
    print("  GOOD -- collect 20 more sequences per weak class")
else:
    print("  Needs more data -- collect_words.py and retrain")
print("  Next: python src/predict.py")
print("="*60)