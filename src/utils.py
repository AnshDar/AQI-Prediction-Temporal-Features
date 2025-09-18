import os
import json
from datetime import datetime

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def timestamped_filename(prefix, ext='csv'):
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{now}.{ext}"
