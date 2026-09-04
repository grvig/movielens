"""Fetch and verify the MovieLens 100K archive.

Run this once before anything else::

    python scripts/download_data.py

The archive is downloaded to ``data/raw/ml-100k.zip``, checked against the MD5 recorded in
``configs/default.yaml`` and extracted to ``data/raw/ml-100k/``. Nothing under ``data/`` is
committed, so this script is the only way a fresh clone gets the dataset.

A checksum mismatch is treated as a hard failure. Silently training on a truncated or
substituted archive would invalidate every number in results/ with no visible symptom.

Expired certificate
-------------------

GroupLens let the certificate for ``files.grouplens.org`` expire on 28 August 2026, so the
ordinary path fails with ``CERTIFICATE_VERIFY_FAILED``. ``--allow-expired-cert`` works
around it *without* turning verification off.

Turning verification off entirely would accept any certificate at all, including one a
machine-in-the-middle generated a moment ago. Instead this pins the certificate: it
requires the server to present exactly the certificate whose SHA-256 fingerprint is in the
config, and ignores only the expiry date. That trusts no certificate authority, so it is
strictly stronger than ordinary verification - the only way to satisfy it is to hold
GroupLens's private key.

The download is then checked twice: against GroupLens's own published ``.md5`` file, and
against the MD5 pinned in the config. Once GroupLens renews, the flag stops being needed
and the plain path works again.
"""

import argparse
import hashlib
import http.client
import shutil
import ssl
import sys
import urllib.parse
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
CONNECTION_TIMEOUT = 60


def normalise_fingerprint(value):
    return value.replace(":", "").replace(" ", "").strip().lower()


def open_pinned_connection(url, expected_fingerprint):
    """Open a TLS connection and refuse it unless the certificate is the pinned one.

    Hostname and date checks are switched off because the fingerprint pin subsumes both: a
    certificate that hashes to the expected value is the certificate we inspected, whatever
    its dates say and whoever signed it.
    """
    parsed = urllib.parse.urlparse(url)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    connection = http.client.HTTPSConnection(
        parsed.netloc, context=context, timeout=CONNECTION_TIMEOUT
    )
    connection.connect()
    certificate = connection.sock.getpeercert(binary_form=True)
    observed = hashlib.sha256(certificate).hexdigest()
    expected = normalise_fingerprint(expected_fingerprint)

    if observed != expected:
        connection.close()
        message = [
            "certificate fingerprint mismatch for " + parsed.netloc,
            "  expected: " + expected,
            "  observed: " + observed,
            "This is not the certificate this project pinned. It may simply have been",
            "renewed, in which case drop --allow-expired-cert and use the ordinary path.",
            "Do not update the pin to whatever was served without checking why.",
        ]
        raise ssl.SSLError("\n".join(message))

    print("certificate pin ok: " + observed[:16] + "... (expiry deliberately not checked)")
    return connection, parsed


def pinned_get(url, expected_fingerprint):
    """GET a URL over a pinned connection, returning the response body as bytes."""
    connection, parsed = open_pinned_connection(url, expected_fingerprint)
    try:
        path = parsed.path
        if parsed.query:
            path = path + "?" + parsed.query
        connection.request("GET", path, headers={"Host": parsed.hostname})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(
                "unexpected status " + str(response.status) + " for " + url
            )
        return response.read()
    finally:
        connection.close()


def download_archive(url, destination, fingerprint=None):
    print("downloading " + url)
    if fingerprint is not None:
        payload = pinned_get(url, fingerprint)
        with open(destination, "wb") as handle:
            handle.write(payload)
    else:
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


def fetch_published_md5(url, fingerprint):
    """Read GroupLens's own published checksum for the archive.

    Their file is of the form "MD5 (ml-100k.zip) = <hash>", so the hash is the last token.
    """
    if fingerprint is None:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8", errors="replace")
    else:
        text = pinned_get(url, fingerprint).decode("utf-8", errors="replace")
    tokens = text.strip().split()
    if len(tokens) == 0:
        raise ValueError("published checksum file at " + url + " was empty")
    return tokens[-1].strip().lower()


def verify_checksum(path, expected, published=None):
    observed = compute_md5(path)
    if published is not None and published != expected:
        raise ValueError(
            "the checksum published by GroupLens (" + published + ") does not match the "
            "one pinned in configs/default.yaml (" + expected + "). Do not proceed until "
            "you know which is right and why they differ."
        )
    if observed == expected:
        print("checksum ok: " + observed)
        if published is not None:
            print("  confirmed against the checksum published by GroupLens")
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
    parser.add_argument(
        "--allow-expired-cert",
        action="store_true",
        help="accept the pinned certificate even though it has expired",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.section("download")
    raw_dir = config.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / settings["archive_name"]

    fingerprint = None
    if args.allow_expired_cert:
        fingerprint = settings["certificate_sha256"]

    if archive_path.exists() and not args.force:
        print("archive already present at " + str(archive_path))
    else:
        try:
            download_archive(settings["url"], archive_path, fingerprint)
        except ssl.SSLCertVerificationError as error:
            raise SystemExit(
                "TLS verification failed: " + str(error) + "\n"
                "GroupLens's certificate expired on 28 August 2026. If it still has not "
                "been renewed, rerun with --allow-expired-cert, which pins the exact "
                "certificate rather than disabling verification."
            )

    published = None
    if "md5_url" in settings:
        published = fetch_published_md5(settings["md5_url"], fingerprint)

    verify_checksum(archive_path, settings["md5"], published)
    extracted_dir = extract_archive(archive_path, raw_dir, settings["extract_subdir"])
    check_expected_files(extracted_dir)
    print("done. next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
