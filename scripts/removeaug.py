# remove_augmented.py
import os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))


# These are the classes fix_final.py augmented
AUGMENTED_CLASSES = ['A', 'I', 'H', 'J', 'E']

print("Removing augmented images...\n")
for action in AUGMENTED_CLASSES:
    folder = os.path.join(ROOT, 'data', action)
    if not os.path.exists(folder):
        continue
    
    all_files = os.listdir(folder)
    removed   = 0

    for fname in all_files:
        if fname.startswith('aug_'):
            os.remove(os.path.join(folder, fname))
            removed += 1

    remaining = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))])
    print(f"  [{action}] Removed {removed} aug files → {remaining} remaining")

print("\nDone! Now run:")
print("  python data.py")
print("  python trainmodel.py")