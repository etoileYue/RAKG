from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install project requirements first."
    ) from exc


DEFAULT_REPO_IDS = ("zai-org/LongBench", "THUDM/LongBench")
DEFAULT_SUBSETS = ("dureader", "multifieldqa_zh")
ARCHIVE_NAME = "data.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected LongBench subsets into local JSONL files."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_IDS[0],
        help=f"Preferred dataset repo id. Fallbacks: {', '.join(DEFAULT_REPO_IDS[1:])}",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="LongBench subset names to extract.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Logical split name used in the output filename. LongBench currently ships test only.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/longbench",
        help="Directory for extracted JSONL files.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=2,
        help="How many rows to print per subset after extraction.",
    )
    return parser.parse_args()


def resolve_repo_candidates(preferred_repo_id: str) -> list[str]:
    candidates = [preferred_repo_id]
    for repo_id in DEFAULT_REPO_IDS:
        if repo_id not in candidates:
            candidates.append(repo_id)
    return candidates


def download_archive(preferred_repo_id: str) -> tuple[str, Path]:
    last_error: Exception | None = None
    for repo_id in resolve_repo_candidates(preferred_repo_id):
        try:
            archive_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=ARCHIVE_NAME,
            )
            return repo_id, Path(archive_path)
        except Exception as exc:  # pragma: no cover - depends on remote state
            last_error = exc

    if last_error is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to resolve a LongBench archive path.")
    raise RuntimeError(
        "Unable to download LongBench archive from any known repo id."
    ) from last_error


def extract_subset(
    archive_path: Path,
    subset: str,
    split: str,
    output_root: Path,
) -> Path:
    member_name = f"data/{subset}.jsonl"
    destination = output_root / subset / f"{split}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        if member_name not in archive.namelist():
            raise FileNotFoundError(
                f"Subset '{subset}' not found in archive. Expected member: {member_name}"
            )
        with archive.open(member_name) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)

    return destination


def load_preview_rows(path: Path, count: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(count), handle):
            rows.append(json.loads(line))
    return rows


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    repo_id, archive_path = download_archive(args.repo_id)

    print(f"Using LongBench archive from: {repo_id}")
    print(f"Archive path: {archive_path}")

    for subset in args.subsets:
        output_path = extract_subset(
            archive_path=archive_path,
            subset=subset,
            split=args.split,
            output_root=output_root,
        )
        total_rows = count_lines(output_path)
        preview_rows = load_preview_rows(output_path, args.preview_count)
        columns = list(preview_rows[0].keys()) if preview_rows else []

        print()
        print(f"[{subset}]")
        print(f"saved_to={output_path}")
        print(f"rows={total_rows}")
        print(f"columns={columns}")
        for index, row in enumerate(preview_rows, start=1):
            print(f"preview_{index}={json.dumps(row, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
