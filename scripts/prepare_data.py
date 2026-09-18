"""Download and prepare Bundestag speech records for LoRA training."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERIODS = (20, 21)
REQUIRED_PERSON_COLUMNS = {"id", "first_name", "last_name"}
REQUIRED_SPEECH_COLUMNS = {
    "id",
    "text",
    "date",
    "protocol_id",
    "speaker_id",
    "speaker_first_name",
    "speaker_last_name",
    "extraction_method",
    "extraction_status",
}
STAGE_DIRECTION_PATTERN = re.compile(
    r"\([^)]*(?:Beifall|Zuruf|Heiterkeit|Lachen|Widerspruch|Unruhe)[^)]*\)",
    flags=re.IGNORECASE,
)
SHORT_PARENTHETICAL_PATTERN = re.compile(r"\(\s*[A-ZÄÖÜ][^)]{0,80}\)")


@dataclass(frozen=True)
class PreparationConfig:
    download_dir: Path
    raw_output: Path
    processed_output: Path
    manifest_output: Path
    periods: tuple[int, ...]
    first_name: str
    last_name: str


def clean_speech_text(text: str) -> str:
    """Remove parliamentary stage directions and normalize whitespace."""
    cleaned = STAGE_DIRECTION_PATTERN.sub("", str(text))
    cleaned = SHORT_PARENTHETICAL_PATTERN.sub(
        lambda match: match.group(0) if len(match.group(0)) > 60 else "",
        cleaned,
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned.strip()


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def normalize_ids(values: pd.Series) -> set[str]:
    return set(values.dropna().astype(str).str.replace(r"\.0$", "", regex=True))


def load_period_speeches(config: PreparationConfig, period: int) -> pd.DataFrame:
    persons_path = config.download_dir / f"bundestag_wp{period}_persons.csv"
    speeches_path = config.download_dir / f"bundestag_wp{period}_speeches.csv"
    if not persons_path.is_file() or not speeches_path.is_file():
        raise FileNotFoundError(
            f"Missing period {period} exports. Expected {persons_path} and {speeches_path}."
        )

    persons = pd.read_csv(persons_path)
    speeches = pd.read_csv(speeches_path, low_memory=False)
    require_columns(persons, REQUIRED_PERSON_COLUMNS, persons_path)
    require_columns(speeches, REQUIRED_SPEECH_COLUMNS, speeches_path)

    name_match = (
        persons["first_name"].fillna("").str.strip().str.casefold()
        == config.first_name.casefold()
    ) & (
        persons["last_name"].fillna("").str.strip().str.casefold()
        == config.last_name.casefold()
    )
    speaker_ids = normalize_ids(persons.loc[name_match, "id"])
    if not speaker_ids:
        raise ValueError(
            f"No exact person match for {config.first_name} {config.last_name} "
            f"in {persons_path}."
        )

    speech_ids = speeches["speaker_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    selected = speeches.loc[speech_ids.isin(speaker_ids)].copy()
    selected = selected.loc[
        selected["extraction_method"].eq("xml")
        & selected["extraction_status"].eq("complete")
        & selected["text"].notna()
    ]
    selected["legislative_period"] = period
    return selected


def prepare_records(config: PreparationConfig) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    period_frames = [load_period_speeches(config, period) for period in config.periods]
    speeches = pd.concat(period_frames, ignore_index=True)
    speeches["text"] = speeches["text"].map(clean_speech_text)
    speeches = speeches.loc[speeches["text"].str.len().gt(0)]
    speeches = speeches.sort_values(
        ["date", "protocol_id", "id", "legislative_period"], kind="stable"
    ).drop_duplicates(subset=["text"], keep="first")
    if speeches.empty:
        raise ValueError("No complete XML speech records matched the requested speaker.")

    training_records = [{"text": text} for text in speeches["text"].tolist()]
    return speeches.reset_index(drop=True), training_records


def write_outputs(
    config: PreparationConfig,
    speeches: pd.DataFrame,
    training_records: Sequence[dict[str, str]],
) -> None:
    for output_path in (
        config.raw_output,
        config.processed_output,
        config.manifest_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    speeches.to_csv(config.raw_output, index=False)
    with config.processed_output.open("w", encoding="utf-8") as output_file:
        for record in training_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "record_count": len(training_records),
        "speaker": f"{config.first_name} {config.last_name}",
        "legislative_periods": list(config.periods),
        "date_min": str(speeches["date"].min()),
        "date_max": str(speeches["date"].max()),
        "source": "German Bundestag DIP plenary protocols",
        "quality_filter": {"extraction_method": "xml", "extraction_status": "complete"},
    }
    config.manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def download_exports(download_dir: Path, periods: Sequence[int], api_key: str | None) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    for period in periods:
        command = [
            "bpe",
            "--period",
            str(period),
            "--output-dir",
            str(download_dir),
            "--disable-xml-cache",
        ]
        if api_key:
            command.extend(["--api-key", api_key])
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare complete XML Bundestag speeches for LoRA training."
    )
    parser.add_argument("--download", action="store_true", help="Download DIP exports first.")
    parser.add_argument("--periods", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--first-name", default="Friedrich")
    parser.add_argument("--last-name", default="Merz")
    parser.add_argument("--download-dir", type=Path, default=REPOSITORY_ROOT / "data/download")
    parser.add_argument("--raw-output", type=Path, default=REPOSITORY_ROOT / "data/raw/speeches.csv")
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/training_data.jsonl",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    periods = tuple(dict.fromkeys(arguments.periods))
    if arguments.download:
        download_exports(
            arguments.download_dir,
            periods,
            os.getenv("BUNDESTAG_API_KEY"),
        )

    config = PreparationConfig(
        download_dir=arguments.download_dir,
        raw_output=arguments.raw_output,
        processed_output=arguments.processed_output,
        manifest_output=arguments.manifest_output,
        periods=periods,
        first_name=arguments.first_name.strip(),
        last_name=arguments.last_name.strip(),
    )
    speeches, training_records = prepare_records(config)
    write_outputs(config, speeches, training_records)
    print(f"Prepared {len(training_records)} records at {config.processed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())