from __future__ import annotations

import argparse
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

STUDY_ID = "ST003390"
STUDY_URL = (
    "https://metabolomicsworkbench.org/data/DRCCMetadata.php?"
    "DataMode=CollectionData&Mode=Study&ResultType=1&StudyID=ST003390&StudyType=MS"
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attr_map = dict(attrs)
            self._current_href = attr_map.get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href:
            self.links.append((self._current_href, " ".join(self._text)))
            self._current_href = None
            self._text = []


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "download"


def extract_links(html: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(html)
    links = []
    for raw_href, raw_text in parser.links:
        href = urljoin(base_url, raw_href)
        text = " ".join(raw_text.split()).lower()
        if any(token in href.lower() for token in [".rar", ".wiff", ".raw"]):
            continue
        if STUDY_ID.lower() in href.lower() or any(
            token in text for token in ["download", "metabolite", "mztab", "mwtab", "data file", "data matrix"]
        ):
            if re.search(r"(download|mwtab|ST003390|showfile|MSdata|zip|csv|tsv|txt|xlsx?|json)", href, re.I):
                links.append(href)
    return links


def filename_from_url(url: str, content_type: str = "") -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    for key in ["DF", "FILE", "filename"]:
        if key in params and params[key]:
            return slugify(params[key][0])
    name = parsed.path.split("/")[-1] or slugify(url)
    if "." not in name:
        suffix = ".html" if "html" in content_type else ".dat"
        name = f"{slugify(name)}{suffix}"
    return slugify(name)


def download_study_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Download public ST003390 files linked from the Workbench study page.

    Workbench exposes this large study through page links/FTP downloads rather than
    a normal mwTab matrix. This helper records metadata and downloads linked ZIP,
    TSV, CSV, TXT, XLS, XLSX, and JSON files it can discover from the study page.
    """
    ensure_dirs()
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(STUDY_URL, timeout=30)
    response.raise_for_status()
    (raw_dir / "ST003390_study_page.html").write_text(response.text, encoding="utf-8")

    candidates = extract_links(response.text, STUDY_URL)

    downloaded: list[Path] = []
    seen = set()
    queue = list(sorted(set(candidates)))
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            size = int(head.headers.get("content-length") or 0)
            if size > 100_000_000:
                continue
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            continue
        content_type = r.headers.get("content-type", "")
        name = filename_from_url(url, content_type)
        out = raw_dir / name
        out.write_bytes(r.content)
        downloaded.append(out)
        if "html" in content_type or out.suffix.lower() in {".php", ".html", ".dat"}:
            try:
                queue.extend(link for link in extract_links(r.text, url) if link not in seen)
            except UnicodeDecodeError:
                pass
        if zipfile.is_zipfile(out):
            with zipfile.ZipFile(out) as zf:
                zf.extractall(raw_dir / out.stem)
    manifest = {"study_id": STUDY_ID, "study_url": STUDY_URL, "links": sorted(seen)}
    (raw_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return downloaded


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_candidate_tables(raw_dir: Path = RAW_DIR) -> list[Path]:
    patterns = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls", "*.json"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(raw_dir.rglob(pattern))
    return [p for p in paths if p.stat().st_size > 0]


def detect_label_column(columns: Iterable[str]) -> str | None:
    candidates = ["Phenotype", "phenotype", "Group", "group", "Disease", "disease", "Class", "label", "Label"]
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for ST003390 project")
    parser.add_argument("command", choices=["download"], help="Utility command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        files = download_study_files()
        print(f"Downloaded/discovered {len(files)} files into {RAW_DIR}")


if __name__ == "__main__":
    main()
