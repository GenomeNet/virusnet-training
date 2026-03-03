#!/usr/bin/env python3
"""Download assets for VirusNet training.

Usage:
    python download_model.py                # download all assets
    python download_model.py model          # base BERT model only
    python download_model.py data           # training data only
    python download_model.py --checksum-only          # print SHA-256 of all files
    python download_model.py model --checksum-only    # print SHA-256 of model
    python download_model.py -o /tmp/model.h5 model   # custom output path
"""

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Asset registry ────────────────────────────────────────────────────
ASSETS = {
    "model": {
        "url": "https://research.bifo.helmholtz-hzi.de/downloads/genomenet/llm_1k_bert.h5",
        "filename": "llm_1k_bert.h5",
        "dest_dir": os.path.join(PROJECT_DIR, "models"),
        "description": "Base BERT model (pre-trained with deepG)",
        "sha256": None,  # pin after first download
        "extract": False,
    },
    "data": {
        "url": "https://research.bifo.helmholtz-hzi.de/downloads/genomenet/additional-archaea-merged.tar.gz",
        "filename": "additional-archaea-merged.tar.gz",
        "dest_dir": os.path.join(PROJECT_DIR, "data"),
        "description": "Training data for VirusNet binary classification",
        "sha256": None,  # pin after first download
        "extract": True,
    },
}


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    """Download *url* to *dest* with a progress indicator."""
    tmp = dest + ".part"
    print(f"Downloading {url}")
    print(f"       to   {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "virusnet-downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0

        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    mb = downloaded / 1e6
                    total_mb = total / 1e6
                    print(f"\r  {mb:8.1f} / {total_mb:.1f} MB  ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:8.1f} MB", end="", flush=True)
        print()

    shutil.move(tmp, dest)
    print("Download complete.")


def extract_tarball(path: str) -> None:
    """Extract a .tar.gz archive into the same directory."""
    dest_dir = os.path.dirname(path)
    print(f"Extracting {os.path.basename(path)} to {dest_dir} ...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")


def verify_checksum(path: str, expected: str | None, name: str) -> bool:
    """Verify file checksum. Returns True if OK or no expected hash set."""
    digest = sha256_file(path)
    print(f"SHA-256: {digest}")
    if expected:
        if digest != expected:
            print(f"CHECKSUM MISMATCH for {name}! Expected {expected}", file=sys.stderr)
            return False
        print("Checksum OK.")
    else:
        print(f"No expected checksum configured for '{name}'.")
        print(f"To pin it, set sha256 in ASSETS[\"{name}\"] to:")
        print(f'  "{digest}"')
    return True


def process_asset(name: str, asset: dict, dest_override: str | None, force: bool, checksum_only: bool) -> bool:
    """Download and verify a single asset. Returns True on success."""
    dest = dest_override or os.path.join(asset["dest_dir"], asset["filename"])

    print(f"\n{'=' * 60}")
    print(f"  {name}: {asset['description']}")
    print(f"{'=' * 60}")

    if checksum_only:
        if not os.path.isfile(dest):
            print(f"File not found: {dest}", file=sys.stderr)
            return False
        print(sha256_file(dest))
        return True

    # Already downloaded?
    if os.path.isfile(dest) and not force:
        print(f"Already exists: {dest}")
        ok = verify_checksum(dest, asset["sha256"], name)
        if not ok:
            print("Re-run with --force to re-download.", file=sys.stderr)
        return ok

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    download(asset["url"], dest)

    ok = verify_checksum(dest, asset["sha256"], name)
    if not ok:
        os.remove(dest)
        return False

    if asset["extract"]:
        extract_tarball(dest)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download VirusNet training assets (base model and training data).")
    parser.add_argument("asset", nargs="*", default=list(ASSETS.keys()),
                        choices=[*ASSETS.keys(), []],
                        help="Which asset(s) to download (default: all)")
    parser.add_argument("-o", "--output", default=None,
                        help="Custom output path (only valid when downloading a single asset)")
    parser.add_argument("--checksum-only", action="store_true",
                        help="Print SHA-256 of existing file(s) and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists.")
    args = parser.parse_args()

    if args.output and len(args.asset) > 1:
        parser.error("-o/--output can only be used with a single asset")

    ok = True
    for name in args.asset:
        dest_override = args.output if len(args.asset) == 1 else None
        if not process_asset(name, ASSETS[name], dest_override, args.force, args.checksum_only):
            ok = False

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
