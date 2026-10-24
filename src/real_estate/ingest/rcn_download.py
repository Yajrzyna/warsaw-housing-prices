"""Download the Warsaw RCN (Rejestr Cen Nieruchomosci) GML archives.

The download links are scraped from the city's page on every run, so new
yearly files (and refreshed snapshots of the current year) are picked up
automatically. Files are stored untouched in data/raw/rcn_transactions/ and
described in manifest.json (URL, size, sha256, download time).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from real_estate.config import HTTP_HEADERS, RCN_RAW_DIR, RCN_WARSAW_HOST, RCN_WARSAW_PAGE

LINK_RE = re.compile(r'href="(/documents/[^"]*?/(zbior_danych_rc[a-z]*_[0-9_]+\.zip)[^"]*)"')


@dataclass
class RcnFile:
    filename: str
    url: str
    size_bytes: int | None = None
    sha256: str | None = None
    downloaded_at: str | None = None


def list_available_files(page_url: str = RCN_WARSAW_PAGE) -> list[RcnFile]:
    """Scrape the city page and return one entry per published zip archive."""
    html = requests.get(page_url, headers=HTTP_HEADERS, timeout=60).text
    files: dict[str, RcnFile] = {}
    for path, filename in LINK_RE.findall(html):
        path = path.split("?")[0]  # drop cache-busting "?t=" parameter
        files[filename] = RcnFile(filename=filename, url=RCN_WARSAW_HOST + path)
    if not files:
        raise RuntimeError(f"No RCN archives found on {page_url} - page layout changed?")
    return sorted(files.values(), key=lambda f: f.filename)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with tmp.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def download_all(out_dir: Path = RCN_RAW_DIR, force: bool = False) -> list[RcnFile]:
    """Download every archive listed on the page; skip files already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = {f["filename"]: f for f in json.loads(manifest_path.read_text())["files"]}

    results = []
    for f in list_available_files():
        dest = out_dir / f.filename
        if dest.exists() and not force and f.filename in manifest:
            results.append(RcnFile(**manifest[f.filename]))
            continue
        _download(f.url, dest)
        f.size_bytes = dest.stat().st_size
        f.sha256 = _sha256(dest)
        f.downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results.append(f)

    manifest_path.write_text(
        json.dumps({"source": RCN_WARSAW_PAGE, "files": [asdict(f) for f in results]}, indent=2)
    )
    return results


if __name__ == "__main__":
    for f in download_all():
        print(f"{f.filename:45s} {f.size_bytes / 1e6:8.1f} MB")
