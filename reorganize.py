# reorganize.py
# -----------------------------------------------------------------------------
# Run this ONCE from your project root:
#   cd D:\signlanguagetranslation
#   python reorganize.py
#
# What it does:
#   1. Creates all target folders
#   2. Moves every file to the right folder
#   3. Auto-patches path references in all .py files
#   4. Prints a full report
#
# Safe to re-run -- skips files already in correct location
# -----------------------------------------------------------------------------

import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

print("="*62)
print("  REORGANIZE -- Sign Language Translation Project")
print(f"  Root: {BASE}")
print("="*62)

# -----------------------------------------------------------------------------
# TARGET FOLDER STRUCTURE
# -----------------------------------------------------------------------------
FOLDERS = [
    'src',
    'scripts',
    'models',
    'results',
    'data',
    'MP_Data',
    'MP_Data_Words',
    'Logs',
]

# -----------------------------------------------------------------------------
# FILE -> DESTINATION MAP
# -----------------------------------------------------------------------------
FILE_MAP = {
    # Core source files -> src/
    'function.py'           : 'src',
    'data.py'               : 'src',
    'trainmodel.py'         : 'src',
    'predict.py'            : 'src',
    'trainmodel_words.py'   : 'src',   # new (if exists)

    # Utility scripts -> scripts/
    'augment_weak.py'       : 'scripts',
    'checkweak.py'          : 'scripts',
    'collectdata.py'        : 'scripts',
    'collect_words.py'      : 'scripts',
    'extract_wlasl.py'      : 'scripts',
    'debug.py'              : 'scripts',
    'finalfix.py'           : 'scripts',
    'fixweak.py'            : 'scripts',
    'removeaug.py'          : 'scripts',

    # Model files -> models/
    'model.h5'              : 'models',
    'model.json'            : 'models',
    'word_model.h5'         : 'models',   # new (if exists)
    'word_model.json'       : 'models',   # new (if exists)
    'word_actions.npy'      : 'models',   # new (if exists)
    'hand_landmarker.task'  : 'models',

    # Result images -> results/
    'accuracy_heatmap.png'  : 'results',
    'confusion_matrix.png'  : 'results',
    'per_class_metrics.png' : 'results',
    'split_metrics.png'     : 'results',
    'training_curves.png'   : 'results',
    'word_training.png'     : 'results',
    'word_confusion.png'    : 'results',

    # This script stays at root
    'reorganize.py'         : None,
}

# -----------------------------------------------------------------------------
# STEP 1 -- Create folders
# -----------------------------------------------------------------------------
print("\n[1/4] Creating folders...")
for folder in FOLDERS:
    path = os.path.join(BASE, folder)
    os.makedirs(path, exist_ok=True)
    print(f"  [OK] {folder}/")

# -----------------------------------------------------------------------------
# STEP 2 -- Move files
# -----------------------------------------------------------------------------
print("\n[2/4] Moving files...")
moved   = []
skipped = []
missing = []

for fname, dest_folder in FILE_MAP.items():
    if dest_folder is None:
        continue

    src_path  = os.path.join(BASE, fname)
    dest_dir  = os.path.join(BASE, dest_folder)
    dest_path = os.path.join(dest_dir, fname)

    if not os.path.exists(src_path):
        # Already moved or doesn't exist
        if os.path.exists(dest_path):
            skipped.append(f"{fname} -> {dest_folder}/ (already there)")
        else:
            missing.append(fname)
        continue

    shutil.move(src_path, dest_path)
    moved.append(f"{fname} -> {dest_folder}/")
    print(f"  [OK] {fname:<35} -> {dest_folder}/")

if skipped:
    print(f"\n  Already in place ({len(skipped)} files):")
    for s in skipped:
        print(f"     {s}")

if missing:
    print(f"\n  Not found -- skipped ({len(missing)} files):")
    for m in missing:
        print(f"     {m}")

# -----------------------------------------------------------------------------
# STEP 3 -- Auto-patch path references in Python files
# -----------------------------------------------------------------------------
print("\n[3/4] Patching path references in Python files...")

# -- ROOT helper block -- injected at top of every src/ and scripts/ file       
# This finds the project root regardless of where the file lives
ROOT_BLOCK = '''import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))
'''

# -- Path replacements per file                                                 
# Maps old path strings to new ones for string substitution in source files
OLD_TO_NEW = {
    # hand_landmarker.task -- used in function.py
    "'hand_landmarker.task'"  : "os.path.join(ROOT, 'models', 'hand_landmarker.task')",
    '"hand_landmarker.task"'  : "os.path.join(ROOT, 'models', 'hand_landmarker.task')",
    "model_asset_path='hand_landmarker.task'":
        "model_asset_path=os.path.join(ROOT, 'models', 'hand_landmarker.task')",

    # MP_Data -- used in data.py, trainmodel.py
    "DATA_PATH = 'MP_Data'"          : "DATA_PATH = os.path.join(ROOT, 'MP_Data')",
    'DATA_PATH = "MP_Data"'          : "DATA_PATH = os.path.join(ROOT, 'MP_Data')",
    "DATA_PATH  = 'MP_Data'"         : "DATA_PATH  = os.path.join(ROOT, 'MP_Data')",
    'DATA_PATH  = "MP_Data"'         : "DATA_PATH  = os.path.join(ROOT, 'MP_Data')",
    "os.path.join('MP_Data'"         : "os.path.join(ROOT, 'MP_Data'",
    "os.path.join(\"MP_Data\""       : "os.path.join(ROOT, 'MP_Data'",

    # MP_Data_Words
    "WORD_DATA_PATH = 'MP_Data_Words'":
        "WORD_DATA_PATH = os.path.join(ROOT, 'MP_Data_Words')",
    "WORD_DATA_PATH  = 'MP_Data_Words'":
        "WORD_DATA_PATH  = os.path.join(ROOT, 'MP_Data_Words')",
    "os.path.join('MP_Data_Words'"   : "os.path.join(ROOT, 'MP_Data_Words'",

    # model.h5 -- used in predict.py, trainmodel.py
    "load_model('model.h5')"   : "load_model(os.path.join(ROOT, 'models', 'model.h5'))",
    'load_model("model.h5")'   : "load_model(os.path.join(ROOT, 'models', 'model.h5'))",
    "model.save('model.h5')"   : "model.save(os.path.join(ROOT, 'models', 'model.h5'))",
    'model.save("model.h5")'   : "model.save(os.path.join(ROOT, 'models', 'model.h5'))",
    "load_model('word_model.h5')":
        "load_model(os.path.join(ROOT, 'models', 'word_model.h5'))",
    "model.save('word_model.h5')":
        "model.save(os.path.join(ROOT, 'models', 'word_model.h5'))",

    # model.json
    "open('model.json', 'w')"  :
        "open(os.path.join(ROOT, 'models', 'model.json'), 'w')",
    'open("model.json", "w")'  :
        "open(os.path.join(ROOT, 'models', 'model.json'), 'w')",
    "open('word_model.json', 'w')" :
        "open(os.path.join(ROOT, 'models', 'word_model.json'), 'w')",
    "open('word_actions.npy')" :
        "open(os.path.join(ROOT, 'models', 'word_actions.npy'))",
    "np.save('word_actions.npy'":
        "np.save(os.path.join(ROOT, 'models', 'word_actions.npy')",
    "np.load('word_actions.npy'":
        "np.load(os.path.join(ROOT, 'models', 'word_actions.npy')",

    # Result images
    "savefig('confusion_matrix.png'":
        "savefig(os.path.join(ROOT, 'results', 'confusion_matrix.png')",
    "savefig('accuracy_heatmap.png'":
        "savefig(os.path.join(ROOT, 'results', 'accuracy_heatmap.png')",
    "savefig('per_class_metrics.png'":
        "savefig(os.path.join(ROOT, 'results', 'per_class_metrics.png')",
    "savefig('split_metrics.png'":
        "savefig(os.path.join(ROOT, 'results', 'split_metrics.png')",
    "savefig('training_curves.png'":
        "savefig(os.path.join(ROOT, 'results', 'training_curves.png')",
    "savefig('word_training.png'":
        "savefig(os.path.join(ROOT, 'results', 'word_training.png')",
    "savefig('word_confusion.png'":
        "savefig(os.path.join(ROOT, 'results', 'word_confusion.png')",

    # data/ folder (letter images)
    "os.path.join('data'"     : "os.path.join(ROOT, 'data'",
    "os.path.join(\"data\""   : "os.path.join(ROOT, 'data'",

    # Logs/
    "log_dir = 'Logs'"        : "log_dir = os.path.join(ROOT, 'Logs')",
    'log_dir = "Logs"'        : "log_dir = os.path.join(ROOT, 'Logs')",
    "os.path.join('Logs'"     : "os.path.join(ROOT, 'Logs'",
}

# -- ROOT_INJECTION sentinel -- only inject once                                 
SENTINEL = "ROOT  = _os.path.dirname(_HERE)"

def needs_root_block(content):
    return SENTINEL not in content

def inject_root_block(content):
    """Inject ROOT block after the last import line at top of file"""
    lines      = content.splitlines(keepends=True)
    insert_at  = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or
                stripped.startswith('from ')):
            insert_at = i + 1

    if insert_at == 0:
        return ROOT_BLOCK + "\n" + content

    result = (
        ''.join(lines[:insert_at]) +
        "\n" + ROOT_BLOCK + "\n" +
        ''.join(lines[insert_at:])
    )
    return result

def patch_file(filepath):
    """Apply all path patches to a Python file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        original = f.read()

    patched = original

    # Apply string replacements
    for old, new in OLD_TO_NEW.items():
        if old in patched:
            patched = patched.replace(old, new)

    # Inject ROOT block if any ROOT reference now exists and block not present
    if 'ROOT' in patched and needs_root_block(patched):
        patched = inject_root_block(patched)

    if patched != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(patched)
        return True
    return False

# -- Patch all Python files in src/ and scripts/                               
TARGET_DIRS = ['src', 'scripts']
patched_files   = []
unchanged_files = []

for folder in TARGET_DIRS:
    folder_path = os.path.join(BASE, folder)
    if not os.path.exists(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if not fname.endswith('.py'):
            continue
        fpath  = os.path.join(folder_path, fname)
        result = patch_file(fpath)
        if result:
            patched_files.append(f"{folder}/{fname}")
            print(f"  [OK] Patched: {folder}/{fname}")
        else:
            unchanged_files.append(f"{folder}/{fname}")
            print(f"  -- No changes: {folder}/{fname}")

# -----------------------------------------------------------------------------
# STEP 4 -- Create run.py at project root
# -----------------------------------------------------------------------------
print("\n[4/4] Creating run.py at project root...")

RUN_PY = '''# run.py  -- project root entry point
# Run from: D:\\signlanguagetranslation\\
#   python run.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

MENU = """
+==================================================+
|      SIGN LANGUAGE TRANSLATION PROJECT           |
+==================================================+
|  1  Extract letter keypoints   (src/data.py)     |
|  2  Train letter model         (src/trainmodel)  |
|  3  Get word data from WLASL   (scripts/extract) |
|  4  Collect words via webcam   (scripts/collect) |
|  5  Train word model           (src/trainmodel_w)|
|  6  Run translator             (src/predict.py)  |
|  0  Exit                                         |
+==================================================+
"""

print(MENU)
choice = input("  Select (0-6): ").strip()

scripts = os.path.join(ROOT, 'scripts')

if   choice == '1': exec(open(os.path.join(SRC,     'data.py'),         encoding='utf-8').read())
elif choice == '2': exec(open(os.path.join(SRC,     'trainmodel.py'),    encoding='utf-8').read())
elif choice == '3': exec(open(os.path.join(scripts, 'extract_wlasl.py'), encoding='utf-8').read())
elif choice == '4': exec(open(os.path.join(scripts, 'collect_words.py'), encoding='utf-8').read())
elif choice == '5': exec(open(os.path.join(SRC,     'trainmodel_words.py'), encoding='utf-8').read())
elif choice == '6': exec(open(os.path.join(SRC,     'predict.py'),       encoding='utf-8').read())
elif choice == '0': sys.exit(0)
else: print("Invalid choice.")
'''

run_path = os.path.join(BASE, 'run.py')
with open(run_path, 'w', encoding='utf-8') as f:
    f.write(RUN_PY)
print(f"  [OK] run.py created at project root")

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "="*62)
print("  DONE -- Final Structure")
print("="*62)

structure = {
    'src/'        : ['function.py','data.py','trainmodel.py',
                     'predict.py','trainmodel_words.py'],
    'scripts/'    : ['augment_weak.py','checkweak.py','collectdata.py',
                     'collect_words.py','extract_wlasl.py',
                     'debug.py','finalfix.py','fixweak.py','removeaug.py'],
    'models/'     : ['model.h5','model.json','word_model.h5',
                     'word_model.json','hand_landmarker.task'],
    'results/'    : ['accuracy_heatmap.png','confusion_matrix.png',
                     'per_class_metrics.png','split_metrics.png',
                     'training_curves.png'],
    'data/'       : ['(your letter image folders A-Z)'],
    'MP_Data/'    : ['(your extracted keypoints A-Z)'],
    'MP_Data_Words/': ['(word sequences -- after extract_wlasl.py)'],
    'Logs/'       : ['(tensorboard logs)'],
}

for folder, files in structure.items():
    folder_path = os.path.join(BASE, folder.rstrip('/'))
    exists      = os.path.exists(folder_path)
    print(f"\n  {'[OK]' if exists else '[!!]'} {folder}")
    for fname in files:
        fpath  = os.path.join(folder_path, fname)
        marker = '[OK]' if os.path.exists(fpath) else '.'
        print(f"     {marker}  {fname}")

print()
print(f"  Files moved     : {len(moved)}")
print(f"  Files patched   : {len(patched_files)}")
print(f"  Already in place: {len(skipped)}")
print()
print("  HOW TO RUN:")
print("  python run.py                    <- menu")
print("  python src/predict.py            <- direct")
print("  python src/data.py               <- extract letters")
print("  python scripts/extract_wlasl.py  <- get word data")
print("  python src/trainmodel_words.py   <- train words")
print("="*62)