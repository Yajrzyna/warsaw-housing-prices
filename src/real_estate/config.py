"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"

RCN_RAW_DIR = RAW_DIR / "rcn_transactions"
RCN_INTERIM_DIR = INTERIM_DIR / "rcn"

# Warsaw's Geodesy & Cadastre Office publishes the full RCN register as GML
# (open data since the 26.09.2025 amendment of Prawo geodezyjne i kartograficzne).
RCN_WARSAW_PAGE = "https://architektura.um.warszawa.pl/udostepniane-dane-rcn1"
RCN_WARSAW_HOST = "https://architektura.um.warszawa.pl"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research; Warsaw apartment price analysis)"}
