"""Portable locations for downloaded data and locally generated outputs."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    Path(os.environ.get("TFLN_ROUTER_DATA", ROOT / "data")).expanduser().resolve()
)
OUTPUT_ROOT = (
    Path(os.environ.get("TFLN_ROUTER_OUTPUT", ROOT / "outputs")).expanduser().resolve()
)


def data_path(relative: str) -> Path:
    path = DATA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Zenodo input: {path}\n"
            "Run python scripts/download_data.py, or set TFLN_ROUTER_DATA "
            "to the downloaded data directory."
        )
    return path


def output_path(relative: str) -> Path:
    path = OUTPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
