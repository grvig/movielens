"""Fetch and verify the MovieLens 100K archive.

Run this once before anything else::

    python scripts/download_data.py

The archive is downloaded to ``data/raw/ml-100k.zip``, checked against the MD5 recorded in
``configs/default.yaml`` and extracted to ``data/raw/ml-100k/``.  Nothing under ``data/``
is committed, so this script is the only way a fresh clone gets the dataset.

A checksum mismatch is treated as a hard failure.  Silently training on a truncated or
substituted archive would invalidate every number in results/ with no visible symptom.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Running a file inside scripts/ puts scripts/ on the import path, not the repository
# root, so "import src" would fail on a fresh clone unless the package had been installed
# first. Putting the root on the path here means the documented commands work straight
# after a git clone, with pip install -e . left as an optional convenience.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config

CHUNK_SIZE = 1024 * 256
EXPECTED_FILES = ["u.data", "u.item", "u.genre", "u.user"]


def download_archive(url, destination):
    print("downloading " + url)
    with urllib.request.urlopen(url) as response:
        total = response.headers.get("Content-Length")
        if total is None:
            total_bytes = 0
        else:
            total_bytes = int(total)
        downloaded = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded = downloaded + len(chunk)
                report_progress(downloaded, total_bytes)
    print("")
    print("saved " + str(destination) + " (" + str(destination.stat().st_size) + " bytes)")


def report_progress(downloaded, total_bytes):
    if total_bytes <= 0:
        return
    percent = 100.0 * downloaded / total_bytes
    sys.stdout.write("\r  " + format(percent, ".1f") + "%")
    sys.stdout.flush()


def compute_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path, expected):
    observed = compute_md5(path)
    if observed == expected:
        print("checksum ok: " + observed)
        return
    message = [
        "checksum mismatch for " + str(path),
        "  expected: " + str(expected),
        "  observed: " + observed,
        "Delete the file and retry. If the observed hash is stable across retries the",
        "published archive has changed; update download.md5 in configs/default.yaml and",
        "say so in the report, because it means the dataset itself moved.",
    ]
    raise ValueError("\n".join(message))


def extract_archive(archive_path, raw_dir, extract_subdir):
    target = raw_dir / extract_subdir
    if target.exists():
        print("removing previous extraction at " + str(target))
        shutil.rmtree(target)
    print("extracting to " + str(raw_dir))
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(raw_dir)
    if not target.exists():
        raise FileNotFoundError("archive did not contain " + extract_subdir + "/")
    return target


def check_expected_files(extracted_dir):
    missing = []
    for name in EXPECTED_FILES:
        if not (extracted_dir / name).exists():
            missing.append(name)
    if len(missing) > 0:
        raise FileNotFoundError("extraction is missing: " + ", ".join(missing))
    print("found all expected files: " + ", ".join(EXPECTED_FILES))


def main():
    parser = argparse.ArgumentParser(description="Download and verify MovieLens 100K.")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.section("download")
    raw_dir = config.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / settings["archive_name"]

    if archive_path.exists() and not args.force:
        print("archive already present at " + str(archive_path))
    else:
        download_archive(settings["url"], archive_path)

    verify_checksum(archive_path, settings["md5"])
    extracted_dir = extract_archive(archive_path, raw_dir, settings["extract_subdir"])
    check_expected_files(extracted_dir)
    print("done. next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
