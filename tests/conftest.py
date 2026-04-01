import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTRA_PATHS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "options"),
    os.path.join(REPO_ROOT, "dataloaders"),
    os.path.join(REPO_ROOT, "models"),
    os.path.join(REPO_ROOT, "utils"),
]

for path in EXTRA_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)

