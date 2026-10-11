
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "vncorenlp"
SEGMENTER_DIR = MODEL_DIR / "models" / "wordsegmenter"

SEGMENTER_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "vncorenlp/VnCoreNLP/master/"
)

files = ["vi-vocab", "wordsegmenter.rdr"]

for filename in files:
    url = BASE_URL + "models/wordsegmenter/" + filename
    destination = SEGMENTER_DIR / filename

    print(f"Đang tải {filename}...", flush=True)
    urlretrieve(url, destination)
    print(f"Đã tải: {destination.name}", flush=True)

print("Kiểm tra dung lượng:")
for filename in files:
    path = SEGMENTER_DIR / filename
    print(filename, path.stat().st_size, "bytes")
