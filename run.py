# run.py  -- project root entry point
# Run from: D:\signlanguagetranslation\
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
