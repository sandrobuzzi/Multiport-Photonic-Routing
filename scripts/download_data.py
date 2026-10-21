"""Download paper data from Zenodo, or import a ZIP downloaded from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import re
import sys
import tempfile
import urllib.error
import urllib.request
import urllib.parse
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfln_paths import DATA_ROOT, ROOT

RECORD_ID = "23194529"
DATA_EXTENSIONS = {".npz", ".csv", ".json", ".s2p", ".npy", ".h5", ".hdf5"}


def archive_data_path(name: str) -> Path | None:
    """Select data-directory members without allowing archive path traversal."""
    parts = PurePosixPath(name).parts
    if not parts or ".." in parts or name.startswith(("/", "\\")) or "\\" in name:
        raise ValueError(f"Invalid archive member: {name}")
    if parts[0] == "data":
        relative = parts[1:]
    elif len(parts) > 2 and parts[1] == "data":
        relative = parts[2:]
    else:
        return None
    path = Path(*relative)
    if not relative or path.suffix.lower() not in DATA_EXTENSIONS:
        return None
    return path


def extract_data(archive: Path, destination: Path) -> int:
    count = 0
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            relative = archive_data_path(member.filename)
            if relative is None:
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(
                    f"Archive path leaves the data directory: {member.filename}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            count += 1
    if not count:
        raise ValueError("The archive contains no paper data directory.")
    return count


def request_json(url: str) -> dict:
    headers = {"User-Agent": "Multiport-Photonic-Routing"}
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as response:
        return json.load(response)


def review_files(url: str):
    """Use a locally supplied review link without saving it in this repository."""
    if urllib.parse.urlsplit(url).hostname != "zenodo.org":
        raise ValueError("The review link must point to zenodo.org.")
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    with opener.open(url, timeout=60) as response:
        page = response.read().decode("utf-8")
    files = {}
    for row in re.findall(r"<tr\b.*?</tr>", page, flags=re.S):
        match = re.search(r'href="([^\"]+/files/[^\"]+\.zip\?download=1[^\"]*)"', row)
        if not match:
            continue
        link = urllib.parse.urljoin("https://zenodo.org", html.unescape(match.group(1)))
        key = Path(urllib.parse.urlsplit(link).path).name
        checksum = re.search(r"md5:([0-9a-f]{32})", row)
        files[key] = {
            "key": key,
            "links": {"self": link},
            "checksum": f"md5:{checksum.group(1)}" if checksum else None,
        }
    if not files:
        raise ValueError("The authorized Zenodo review page lists no ZIP archive.")
    return list(files.values()), opener


def download(url: str, target: Path, checksum: str | None, opener=None) -> None:
    headers = {"User-Agent": "Multiport-Photonic-Routing"}
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    algorithm, expected = checksum.split(":", 1) if checksum else ("sha256", None)
    digest = hashlib.new(algorithm)
    open_url = opener.open if opener else urllib.request.urlopen
    with open_url(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as incoming, target.open("wb") as outgoing:
        while block := incoming.read(1024 * 1024):
            outgoing.write(block)
            digest.update(block)
    if expected and digest.hexdigest() != expected:
        target.unlink()
        raise ValueError("Downloaded archive checksum does not match Zenodo metadata.")


def report_missing(destination: Path) -> list[str]:
    manifest = json.loads((ROOT / "config/data_manifest.json").read_text())
    return [
        item["path"]
        for item in manifest["files"]
        if not (destination / item["path"]).is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default=RECORD_ID, help="Zenodo record ID.")
    parser.add_argument(
        "--archive", type=Path, help="ZIP already downloaded from Zenodo."
    )
    parser.add_argument("--destination", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report missing inputs without downloading.",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if not args.check:
        destination.mkdir(parents=True, exist_ok=True)
        if args.archive:
            print(f"Imported {extract_data(args.archive, destination)} data files.")
        else:
            review_url = os.environ.get("ZENODO_REVIEW_URL")
            opener = None
            if review_url:
                files, opener = review_files(review_url)
            else:
                try:
                    record = request_json(
                        f"https://zenodo.org/api/records/{args.record}"
                    )
                except urllib.error.HTTPError as error:
                    if error.code in (401, 403, 404):
                        raise SystemExit(
                            "The Zenodo record is not publicly accessible yet. "
                            "Set ZENODO_REVIEW_URL locally, use --archive with an authorized "
                            "Zenodo download, or retry after publication."
                        ) from None
                    raise
                files = [
                    item
                    for item in record.get("files", [])
                    if item["key"].lower().endswith(".zip")
                ]
            if not files:
                raise SystemExit("No ZIP data archive is listed in this Zenodo record.")
            for item in files:
                with tempfile.TemporaryDirectory(prefix="tfln-zenodo-") as temporary:
                    archive = Path(temporary) / "data.zip"
                    download(
                        item["links"]["self"], archive, item.get("checksum"), opener
                    )
                    print(
                        f"Imported {extract_data(archive, destination)} files from {item['key']}."
                    )
    missing = report_missing(destination)
    if missing:
        print("Inputs absent from this local Zenodo download:")
        for path in missing:
            print(f"  {path}")
        if args.check:
            raise SystemExit(1)
    else:
        print("All inputs in the data manifest are present.")


if __name__ == "__main__":
    main()
