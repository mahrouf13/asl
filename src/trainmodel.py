# src/trainmodel.py  -- LETTER model
# Letter model -- input_shape=(63,) -- single dominant hand
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from function import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))

os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'models'),  exist_ok=True)

# ======================================================================
# 1. LOAD DATA
# ======================================================================
label_map  = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
zero_count = 0

print("Loading letter data...")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"  [SKIP] {action_path}")
        continue
    for sequence in sorted(os.listdir(action_path)):
        seq_path = os.path.join(action_path, sequence, "0.npy")
        if not os.path.exists(seq_path):
            continue
        res = np.load(seq_path)
        if np.all(res == 0):
            zero_count += 1
            continue
        sequences.append(res)
        labels.append(label_map[action])

print(f"  Total loaded : {len(sequences)}")
print(f"  Zero skipped : {zero_count}")

if len(sequences) == 0:
    print("ERROR: No data found. Run data.py first.")
    exit(1)

X     = np.array(sequences)
y_raw = np.array(labels)

print(f"  Input shape  : {X.shape}  (should be (N, 63))")
if X.shape[1] != SINGLE_HAND_KP:
    print(f"  WARNING: Expected {SINGLE_HAND_KP} features, got {X.shape[1]}")
    print("  Re-run data.py to re-extract with new function.py")
    exit(1)

# ======================================================================
# 2. SPLIT  70 / 15 / 15
# ======================================================================
X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y_raw, test_size=0.30, random_state=42, stratify=y_raw)
X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw)

y_train = to_categorical(y_train_raw, len(actions))
y_val   = to_categorical(y_val_raw,   len(actions))
y_test  = to_categorical(y_test_raw,  len(actions))

print(f"\n{'='*55}")
print(f"  DATASET SPLIT")
print(f"{'='*55}")
print(f"  Total    : {len(X)}")
print(f"  Train    : {len(X_train)}")
print(f"  Val      : {len(X_val)}")
print(f"  Test     : {len(X_test)}")
print(f"  Classes  : {len(actions)}")
print(f"{'='*55}\n")

# ======================================================================
# 3. CLASS WEIGHTS
# ======================================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weights = dict(enumerate(class_weights))

# ======================================================================
# 4. MODEL  -- input_shape=(63,) -- letter model, single hand
# ======================================================================
model = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.001),
          input_shape=(SINGLE_HAND_KP,)),  # 63 -- single hand letter model
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

# ======================================================================
# 5. TRAIN
# ======================================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

# ======================================================================
# 6. EVALUATE
# ======================================================================
train_pred_prob = model.predict(X_train, verbose=0)
val_pred_prob   = model.predict(X_val,   verbose=0)
test_pred_prob  = model.predict(X_test,  verbose=0)

train_pred = np.argmax(train_pred_prob, axis=1)
val_pred   = np.argmax(val_pred_prob,   axis=1)
test_pred  = np.argmax(test_pred_prob,  axis=1)

def compute_all_metrics(y_true, y_pred, y_prob, split_name):
    acc      = accuracy_score(y_true, y_pred) * 100
    f1_mac   = f1_score(y_true, y_pred, average='macro')        * 100
    f1_wt    = f1_score(y_true, y_pred, average='weighted')     * 100
    prec_mac = precision_score(y_true, y_pred, average='macro',    zero_division=0) * 100
    rec_mac  = recall_score(y_true, y_pred, average='macro',       zero_division=0) * 100
    y_true_cat = to_categorical(y_true, len(actions))
    try:
        auc = roc_auc_score(y_true_cat, y_prob, multi_class='ovr', average='macro') * 100
    except Exception:
        auc = float('nan')
    return {'split':split_name,'accuracy':acc,'f1_macro':f1_mac,
            'f1_weighted':f1_wt,'prec_macro':prec_mac,
            'recall_macro':rec_mac,'auc_macro':auc}

train_metrics = compute_all_metrics(y_train_raw, train_pred, train_pred_prob, 'Train')
val_metrics   = compute_all_metrics(y_val_raw,   val_pred,   val_pred_prob,   'Validation')
test_metrics  = compute_all_metrics(y_test_raw,  test_pred,  test_pred_prob,  'Test')

print(f"\n{'='*65}")
print(f"  METRICS REPORT")
print(f"{'='*65}")
print(f"  {'Metric':<22} {'Train':>10}  {'Val':>10}  {'Test':>10}")
print(f"  {'-'*56}")
for key, label in [('accuracy','Accuracy'),('f1_macro','F1 Macro'),
                   ('f1_weighted','F1 Weighted'),('prec_macro','Prec Macro'),
                   ('recall_macro','Recall Macro'),('auc_macro','ROC-AUC')]:
    print(f"  {label:<22} {train_metrics[key]:>9.2f}%  "
          f"{val_metrics[key]:>9.2f}%  {test_metrics[key]:>9.2f}%")
print(f"{'='*65}")

# Per-class
print(f"\n{'='*75}")
print(f"  PER-CLASS METRICS  (TEST SET)")
print(f"{'='*75}")
print(f"  {'Class':<8} {'Acc':>8}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
print(f"  {'-'*55}")

per_class_rows = []
for i, action in enumerate(actions):
    mask    = y_test_raw == i
    total   = int(mask.sum())
    if total == 0:
        continue
    correct = int((test_pred[mask] == i).sum())
    acc_c   = correct / total * 100
    prec = precision_score(y_test_raw, test_pred, labels=[i],
                           average='macro', zero_division=0) * 100
    rec  = recall_score(y_test_raw, test_pred, labels=[i],
                        average='macro', zero_division=0) * 100
    f1   = f1_score(y_test_raw, test_pred, labels=[i],
                    average='macro', zero_division=0) * 100
    per_class_rows.append((action, acc_c, prec, rec, f1, total, correct))
    flag = 'OK' if acc_c >= 90 else 'WN' if acc_c >= 75 else '!!'
    print(f"  [{flag}] {action:<6} {acc_c:>7.1f}%  {prec:>7.1f}%  "
          f"{rec:>7.1f}%  {f1:>7.1f}%  {total:>8}")

print()
print(f"  Bottom 5 classes:")
for r in sorted(per_class_rows, key=lambda x: x[1])[:5]:
    print(f"    {r[0]}: {r[1]:.1f}%")

# ======================================================================
# 7. PLOTS
# ======================================================================
results_path = os.path.join(ROOT, 'results')
os.makedirs(results_path, exist_ok=True)

# -- Plot A:
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Letter Model -- Training History', fontsize=15, fontweight='bold')

axes[0].plot(history.history['categorical_accuracy'],
             color='royalblue', linewidth=2, label='Train')
axes[0].plot(history.history['val_categorical_accuracy'],
             color='orange', linewidth=2, label='Validation')
axes[0].axhline(y=test_metrics['accuracy']/100, color='green',
                linestyle='--', linewidth=1.5,
                label=f"Test ({test_metrics['accuracy']:.2f}%)")
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.05)

axes[1].plot(history.history['loss'],
             color='royalblue', linewidth=2, label='Train')
axes[1].plot(history.history['val_loss'],
             color='orange', linewidth=2, label='Validation')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/training_curves.png")

# -- Plot B:
cm = confusion_matrix(y_test_raw, test_pred)
fig, ax = plt.subplots(figsize=(16, 14))
im = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=actions, yticklabels=actions,
                 linewidths=0.5, linecolor='lightgray',
                 annot_kws={"size": 10}, ax=ax)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title(f"Confusion Matrix  |  Test Accuracy: {test_metrics['accuracy']:.2f}%",
             fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/confusion_matrix.png")

# -- Plot C:
classes_  = [r[0] for r in per_class_rows]
accs_     = [r[1] for r in per_class_rows]
precs_    = [r[2] for r in per_class_rows]
recs_     = [r[3] for r in per_class_rows]
f1s_      = [r[4] for r in per_class_rows]

x     = np.arange(len(classes_))
width = 0.20

fig, ax = plt.subplots(figsize=(22, 6))
b1 = ax.bar(x - 1.5*width, precs_, width, label='Precision',
            color='royalblue', alpha=0.85)
b2 = ax.bar(x - 0.5*width, recs_,  width, label='Recall',
            color='orange',    alpha=0.85)
b3 = ax.bar(x + 0.5*width, f1s_,   width, label='F1',
            color='green',     alpha=0.85)
b4 = ax.bar(x + 1.5*width, accs_,  width, label='Accuracy',
            color='mediumpurple', alpha=0.85)

ax.axhline(y=90, color='red', linestyle='--', alpha=0.5,
           linewidth=1.2, label='90% line')
ax.set_xticks(x)
ax.set_xticklabels(classes_, fontsize=12)
ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Per-Class Metrics -- Test Set', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/per_class_metrics.png")

# -- Plot D:
metric_keys   = ['accuracy', 'f1_macro', 'f1_weighted',
                 'prec_macro', 'recall_macro', 'auc_macro']
metric_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted',
                 'Prec Macro', 'Recall Macro', 'ROC-AUC']

x2     = np.arange(len(metric_keys))
width2 = 0.25
fig, ax2 = plt.subplots(figsize=(14, 6))
ax2.bar(x2 - width2, [train_metrics[k] for k in metric_keys],
        width2, label='Train',      color='royalblue', alpha=0.85)
ax2.bar(x2,          [val_metrics[k]   for k in metric_keys],
        width2, label='Validation', color='orange',    alpha=0.85)
ax2.bar(x2 + width2, [test_metrics[k]  for k in metric_keys],
        width2, label='Test',       color='green',     alpha=0.85)

# value labels on bars
for bars in ax2.containers:
    ax2.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)

ax2.set_xticks(x2)
ax2.set_xticklabels(metric_labels, fontsize=11)
ax2.set_ylim(70, 107)
ax2.set_ylabel('Score (%)', fontsize=12)
ax2.set_title('All Metrics -- Train / Val / Test',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'split_metrics.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/split_metrics.png")

# -- Plot E:
acc_grid = np.array(accs_).reshape(1, -1)
fig, ax3 = plt.subplots(figsize=(20, 2.5))
sns.heatmap(acc_grid, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=classes_, yticklabels=['Acc %'],
            vmin=60, vmax=100, linewidths=0.5, ax=ax3,
            annot_kws={"size": 11})
ax3.set_title('Per-Class Accuracy Heatmap -- Test Set',
              fontsize=13, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'accuracy_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/accuracy_heatmap.png")

# -- Plot F:
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, ax4 = plt.subplots(figsize=(16, 14))
sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=actions, yticklabels=actions,
            linewidths=0.5, linecolor='lightgray',
            vmin=0, vmax=100,
            annot_kws={"size": 9}, ax=ax4)
ax4.set_xlabel('Predicted', fontsize=13)
ax4.set_ylabel('Actual', fontsize=13)
ax4.set_title('Normalised Confusion Matrix (%)',
              fontsize=14, fontweight='bold')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'confusion_matrix_norm.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/confusion_matrix_norm.png")

print()
print("  All 6 plots saved to results/")


# ======================================================================
# 8. SAVE MODEL
# ======================================================================
model.save(os.path.join(ROOT, 'models', 'model.h5'))
with open(os.path.join(ROOT, 'models', 'model.json'), 'w') as f:
    f.write(model.to_json())

print(f"\n{'='*55}")
print("  SAVED: models/model.h5")
print("  SAVED: models/model.json")
print(f"{'='*55}")
print("\n  Next step: python scripts/collect_word_data.py")