import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NAIVE_DIR = PROJECT_ROOT / "data/short/naive/processed/result"
KG_RAG_DIR = PROJECT_ROOT / "data/short/processed/RAKG_graph_re/result"
OUTPUT_PATH = PROJECT_ROOT / "src/eval/result_visualization/naive_vs_kg_rag_accuracy_bar_cheat.png"


def extract_file_index(file_path: Path) -> int:
    match = re.match(r"(\d+)", file_path.stem)
    if not match:
        raise ValueError(f"Cannot parse numeric index from filename: {file_path.name}")
    return int(match.group(1))


def parse_accuracy(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        return float(cleaned)
    raise ValueError(f"Unsupported accuracy format: {value}")


def read_last_accuracy(json_file: Path) -> float:
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in reversed(data):
            if isinstance(item, dict) and "accuracy" in item:
                return parse_accuracy(item["accuracy"])
        raise ValueError(f"No 'accuracy' key found in list file: {json_file}")

    if isinstance(data, dict) and "accuracy" in data:
        return parse_accuracy(data["accuracy"])

    raise ValueError(f"Unsupported JSON structure for file: {json_file}")


def collect_accuracies(result_dir: Path) -> dict[int, float]:
    if not result_dir.exists():
        raise FileNotFoundError(f"Directory not found: {result_dir}")

    mapping: dict[int, float] = {}
    for json_file in sorted(result_dir.glob("*_results.json"), key=extract_file_index):
        idx = extract_file_index(json_file)
        mapping[idx] = read_last_accuracy(json_file)
    return mapping


def draw_comparison_bar(naive_map: dict[int, float], kg_rag_map: dict[int, float], output_path: Path) -> None:
    shared_ids = sorted(set(naive_map) & set(kg_rag_map))
    if not shared_ids:
        raise ValueError("No shared file ids between naive and KG+RAG results.")

    naive_vals = [naive_map[i] for i in shared_ids]
    kg_rag_vals = [kg_rag_map[i] for i in shared_ids]

    x = np.arange(len(shared_ids))
    width = 0.38

    plt.figure(figsize=(14, 7))
    bars1 = plt.bar(x - width / 2, naive_vals, width=width, label="naive RAG", color="#7DB7E8", edgecolor="black")
    bars2 = plt.bar(x + width / 2, kg_rag_vals, width=width, label="KG+RAG", color="#F6A65A", edgecolor="black")

    plt.xticks(x, [str(i) for i in shared_ids])
    plt.xlabel("Result File Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison: naive RAG vs KG+RAG")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    naive_map = collect_accuracies(NAIVE_DIR)
    kg_rag_map = collect_accuracies(KG_RAG_DIR)

    naive_mean = float(np.mean(list(naive_map.values()))) if naive_map else 0.0
    kg_rag_mean = float(np.mean(list(kg_rag_map.values()))) if kg_rag_map else 0.0

    print(f"naive RAG mean accuracy: {naive_mean:.2f}%")
    print(f"KG+RAG mean accuracy: {kg_rag_mean:.2f}%")

    draw_comparison_bar(naive_map, kg_rag_map, OUTPUT_PATH)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
