import os
import sys
import re
import pkg_resources

# Folders to ignore
IGNORE_FOLDERS = {"assets", "images", "videos", "models", "__pycache__"}

# Output requirements file
REQ_FILE = "requirements.txt"

# Root folder
ROOT = os.path.abspath(os.path.dirname(__file__))

# Regex to detect import statements
IMPORT_RE = re.compile(r"^(?:import|from)\s+([a-zA-Z0-9_]+)")

# Set to store packages
packages = set()

def is_external_package(pkg_name):
    """Check if a package is installed in current environment (pip)"""
    try:
        dist = pkg_resources.get_distribution(pkg_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False
    except Exception:
        return False

for subdir, dirs, files in os.walk(ROOT):
    # Skip ignored folders
    dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

    for file in files:
        if file.endswith(".py"):
            path = os.path.join(subdir, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # Skip files that cannot be read
                continue

            for line in lines:
                match = IMPORT_RE.match(line.strip())
                if match:
                    pkg = match.group(1).split('.')[0]  # Take top-level package
                    if is_external_package(pkg):
                        packages.add(pkg)

# Write packages to requirements.txt
with open(os.path.join(ROOT, REQ_FILE), "w") as f:
    for pkg in sorted(packages):
        f.write(pkg + "\n")

print(f"requirements.txt generated with {len(packages)} packages!")