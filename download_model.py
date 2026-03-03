#!/usr/bin/env python3
"""Download the base BERT model used for fine-tuning.

Usage:
    python download_model.py                    # download to models/llm_1k_bert.h5
    python download_model.py -o /tmp/model.h5   # custom output path
    python download_model.py --checksum-only     # print SHA-256 of existing file
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import shutil

MODEL_URL = "https://research.bifo.helmholtz-hzi.de/downloads/genomenet/llm_1k_bert.h5"
MODEL_FILENAME = "llm_1k_bert.h5"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Set to the known SHA-256 after first successful download.
# Run:  python download_model.py --checksum-only
EXPECTED_SHA256 = None  # e.g. "abc123..."


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download base BERT model for virusnet fine-tuning.")
    parser.add_argument("-o", "--output", default=None,
                        help=f"Output file path (default: {DEFAULT_DIR}/{MODEL_FILENAME})")
    parser.add_argument("--checksum-only", action="store_true",
                        help="Print SHA-256 of existing model file and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists.")
    args = parser.parse_args()

    dest = args.output or os.path.join(DEFAULT_DIR, MODEL_FILENAME)

    if args.checksum_only:
        if not os.path.isfile(dest):
            print(f"File not found: {dest}", file=sys.stderr)
            sys.exit(1)
        print(sha256_file(dest))
        sys.exit(0)

    # Check if already downloaded
    if os.path.isfile(dest) and not args.force:
        print(f"Model already exists: {dest}")
        digest = sha256_file(dest)
        print(f"SHA-256: {digest}")
        if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
            print(f"WARNING: checksum mismatch (expected {EXPECTED_SHA256})", file=sys.stderr)
            print("Re-run with --force to re-download.", file=sys.stderr)
            sys.exit(1)
        print("Checksum OK." if EXPECTED_SHA256 else "No expected checksum configured — skipping verification.")
        sys.exit(0)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    download(MODEL_URL, dest)

    # Verify checksum
    digest = sha256_file(dest)
    print(f"SHA-256: {digest}")

    if EXPECTED_SHA256:
        if digest != EXPECTED_SHA256:
            print(f"CHECKSUM MISMATCH! Expected {EXPECTED_SHA256}", file=sys.stderr)
            os.remove(dest)
            sys.exit(1)
        print("Checksum OK.")
    else:
        print("No expected checksum configured yet.")
        print("To pin this checksum, set EXPECTED_SHA256 in download_model.py to:")
        print(f'EXPECTED_SHA256 = "{digest}"')


if __name__ == "__main__":
    main()
