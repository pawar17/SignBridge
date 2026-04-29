"""
SignBridge — Dataset Downloader
================================
Downloads the best publicly available sign-language fingerspelling datasets.

Datasets:
  ASL  grassknoted/asl-alphabet      87,000 imgs  29 classes  200x200 real hands
  ISL  (kept from existing data — no better public static dataset)
  BSL  (no usable static dataset — falls back to ASL at runtime)

Usage:
    python scripts/download_datasets.py             # download all
    python scripts/download_datasets.py --lang ASL  # download only ASL

Kaggle API setup (one-time):
    1. Log in to https://www.kaggle.com
    2. Go to Account -> API -> Create New Token
    3. Save the downloaded kaggle.json to:
         Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json
         Mac/Linux: ~/.kaggle/kaggle.json
    4. Run this script again.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "raw"

# ── Dataset catalogue ─────────────────────────────────────────────────────────
DATASETS = {
    "ASL": {
        "kaggle_id":  "grassknoted/asl-alphabet",
        "zip_name":   "asl-alphabet.zip",
        "target_dir": DATA_ROOT / "asl" / "asl_alphabet",
        "description": "ASL Alphabet by Akash (87K images, 29 classes, 3000 imgs/class)",
        # expected inner folder produced by Kaggle zip
        "inner_train": "asl_alphabet_train/asl_alphabet_train",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_kaggle_auth() -> bool:
    try:
        import kaggle                          # noqa: F401 — just check importable
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return True, api
    except Exception as e:
        return False, str(e)


def _print_setup_instructions():
    print("""
╔══════════════════════════════════════════════════════════╗
║           Kaggle API credentials not found               ║
╠══════════════════════════════════════════════════════════╣
║  One-time setup (takes ~2 minutes):                      ║
║                                                          ║
║  1. Open https://www.kaggle.com and log in               ║
║  2. Click your avatar → Settings → API → Create New Token║
║  3. A file called kaggle.json will be downloaded         ║
║  4. Copy it to:                                          ║
║       Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json   ║
║       Mac/Linux: ~/.kaggle/kaggle.json                   ║
║  5. Run this script again                                ║
║                                                          ║
║  Alternatively, accept the dataset T&C on Kaggle first:  ║
║  https://www.kaggle.com/datasets/grassknoted/asl-alphabet║
╚══════════════════════════════════════════════════════════╝
""")


def _download_and_extract(api, dataset_id: str, zip_name: str,
                           extract_to: Path) -> Path:
    """Download Kaggle dataset zip and extract it."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    zip_path = tmp / zip_name

    print(f"  Downloading {dataset_id} …")
    api.dataset_download_files(dataset_id, path=str(tmp), quiet=False, unzip=False)

    # Kaggle names the file <dataset-name>.zip
    downloaded = list(tmp.glob("*.zip"))
    if not downloaded:
        raise FileNotFoundError(f"No zip found in {tmp} after download")
    zip_path = downloaded[0]

    print(f"  Extracting to {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  Extracted: {extract_to}")
    return extract_to


def _normalise_asl_alphabet(base: Path):
    """
    The asl-alphabet Kaggle zip extracts to:
        asl_alphabet_train/asl_alphabet_train/<A>/<A_001.jpg> …
    We want:
        asl/asl_alphabet/<A>/<A_001.jpg> …

    This function moves the inner folder up and removes the wrapper.
    """
    inner = base / "asl_alphabet_train" / "asl_alphabet_train"
    if not inner.exists():
        # already flat or already moved
        print("  Layout already normalised.")
        return

    print("  Normalising folder layout …")
    for cls_dir in inner.iterdir():
        dest = base / cls_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(cls_dir), str(dest))

    # clean up wrapper dirs
    shutil.rmtree(base / "asl_alphabet_train", ignore_errors=True)

    # also pull out the test folder if present
    test_dir = base / "asl_alphabet_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    classes = sorted([d.name for d in base.iterdir() if d.is_dir()])
    total   = sum(len(list(d.iterdir())) for d in base.iterdir() if d.is_dir())
    print(f"  {len(classes)} classes, {total:,} images: {classes}")


# ── per-language download ─────────────────────────────────────────────────────

def download_asl(api):
    info = DATASETS["ASL"]
    target = info["target_dir"]

    # Skip if already downloaded
    if target.exists() and any(target.iterdir()):
        classes = [d for d in target.iterdir() if d.is_dir()]
        if len(classes) >= 26:
            total = sum(len(list(d.iterdir())) for d in classes)
            print(f"  ASL already present: {len(classes)} classes, {total:,} images — skipping.")
            return target

    _download_and_extract(api, info["kaggle_id"], info["zip_name"], target)
    _normalise_asl_alphabet(target)
    return target


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download sign-language datasets")
    parser.add_argument("--lang", choices=["ASL", "ALL"], default="ALL")
    args = parser.parse_args()

    ok, api_or_err = _check_kaggle_auth()
    if not ok:
        _print_setup_instructions()
        print(f"Error detail: {api_or_err}")
        sys.exit(1)

    api = api_or_err
    print(f"Kaggle authenticated as: {api.get_config_value('username')}\n")

    langs = ["ASL"] if args.lang == "ASL" else ["ASL"]   # extend when BSL/ISL sources added

    for lang in langs:
        print(f"{'='*50}")
        print(f"  {lang}: {DATASETS[lang]['description']}")
        print(f"{'='*50}")
        if lang == "ASL":
            download_asl(api)

    print("\nAll downloads complete.")
    print("Next step: python scripts/training/extract_and_train_landmarks.py --lang ALL")


if __name__ == "__main__":
    main()
