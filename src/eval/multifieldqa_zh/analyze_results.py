"""Analyze MultiFieldQA-ZH scored results for RAKG and NaiveRAG."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Path("/tmp/rakg-matplotlib").mkdir(parents=True, exist_ok=True)
Path("/tmp/rakg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rakg-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rakg-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAKG_RESULTS = PROJECT_ROOT / "data/eval/multifieldqa_zh/result/scored_results.jsonl"
DEFAULT_NAIVE_RESULTS = PROJECT_ROOT / "data/eval/naiveRAG/result/scored_results.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/eval/analysis_figures"

METHOD_NAMES = {
    "rakg": "本项目架构",
    "naive": "NaiveRAG",
}

PLOT_METHOD_NAMES = METHOD_NAMES.copy()
EN_PLOT_METHOD_NAMES = {
    "rakg": "RAKG",
    "naive": "NaiveRAG",
}

METRICS = {
    "official_f1": "官方 F1",
    "answer_judge": "答案正确性 Judge",
    "retrieval_judge": "检索覆盖性 Judge",
}

PLOT_METRICS = METRICS.copy()
EN_PLOT_METRICS = {
    "official_f1": "Official F1",
    "answer_judge": "Answer Judge",
    "retrieval_judge": "Retrieval Judge",
}

PLOT_TEXT = {
    "sample_index": "原始文本编号",
    "metric_mean_suffix": "平均值",
    "line_title_suffix": "逐文本 QA 平均值对比",
    "overview_title": "共同原始文本的三项指标平均值对比",
    "overview_ylabel": "文本级均值的平均值",
    "win_loss_title": "本项目架构相对 NaiveRAG 的逐文本平均官方 F1 差值",
    "win_loss_ylabel": "平均 F1 差值",
    "heatmap_title": "答案 Judge 与检索 Judge 逐文本平均热力图",
    "heatmap_rows": {
        "rakg_answer": "本项目架构 答案",
        "rakg_retrieval": "本项目架构 检索",
        "naive_answer": "NaiveRAG 答案",
        "naive_retrieval": "NaiveRAG 检索",
    },
    "missing": "缺失",
    "heatmap_colorbar": "文本内 QA 通过比例",
}

EN_PLOT_TEXT = {
    "sample_index": "Source sample index",
    "metric_mean_suffix": "mean",
    "line_title_suffix": "per-source QA mean comparison",
    "overview_title": "Metric means over shared source samples",
    "overview_ylabel": "Mean of per-source means",
    "win_loss_title": "Per-source official F1 difference: RAKG minus NaiveRAG",
    "win_loss_ylabel": "Mean F1 difference",
    "heatmap_title": "Answer and retrieval judge pass-rate heatmap",
    "heatmap_rows": {
        "rakg_answer": "RAKG answer",
        "rakg_retrieval": "RAKG retrieval",
        "naive_answer": "NaiveRAG answer",
        "naive_retrieval": "NaiveRAG retrieval",
    },
    "missing": "N/A",
    "heatmap_colorbar": "QA pass rate within source",
}

PREFERRED_CJK_FONTS = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
]

PREFERRED_CJK_FONT_PATHS = [
    Path("/tmp/noto-cjk-fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
]


@dataclass(frozen=True)
class LoadedResults:
    records_by_text: dict[int, list[dict[str, Any]]]
    qa_count_by_text: dict[int, int]
    metric_means_by_text: dict[int, dict[str, float]]
    skipped_non_success: int
    skipped_missing_text_index: int
    duplicate_qa_overwrites: int
    ignored_legacy_records: int


def configure_matplotlib(font_path: Path | None = None) -> bool:
    """Use an installed CJK-capable font when one is available."""

    candidate_font_paths = [font_path] if font_path else []
    candidate_font_paths.extend(PREFERRED_CJK_FONT_PATHS)
    for candidate in candidate_font_paths:
        if not candidate:
            continue
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            font_name = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["figure.dpi"] = 140
            return True

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in PREFERRED_CJK_FONTS:
        if font_name in installed:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["figure.dpi"] = 140
            return True
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("警告：未找到可用中文字体，图表将使用英文标签。可通过 --font-path 指定字体文件。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    return False


def use_english_plot_labels() -> None:
    PLOT_METHOD_NAMES.update(EN_PLOT_METHOD_NAMES)
    PLOT_METRICS.update(EN_PLOT_METRICS)
    PLOT_TEXT.update(EN_PLOT_TEXT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 MultiFieldQA-ZH 前若干条评分结果并绘图。")
    parser.add_argument(
        "--rakg-results",
        type=Path,
        default=DEFAULT_RAKG_RESULTS,
        help=f"本项目架构 scored_results.jsonl 路径，默认 {DEFAULT_RAKG_RESULTS}",
    )
    parser.add_argument(
        "--naive-results",
        type=Path,
        default=DEFAULT_NAIVE_RESULTS,
        help=f"NaiveRAG scored_results.jsonl 路径，默认 {DEFAULT_NAIVE_RESULTS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"图表和摘要输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="从 source_sample_index=0 开始分析的原始文本数量，默认 50。",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="可选中文字体文件路径，例如 NotoSansCJK-Regular.ttc。",
    )
    return parser.parse_args()


def parse_text_index(record: dict[str, Any]) -> int | None:
    raw_index = record.get("source_sample_index", record.get("sample_index"))
    if raw_index is None:
        return None
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path, sample_count: int) -> LoadedResults:
    expanded_by_text: dict[int, dict[str, dict[str, Any]]] = {}
    legacy_by_text: dict[int, dict[str, Any]] = {}
    legacy_count_by_text: dict[int, int] = {}
    skipped_non_success = 0
    skipped_missing_text_index = 0
    duplicate_qa_overwrites = 0
    ignored_legacy_records = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text_index = parse_text_index(record)
            if text_index is None:
                skipped_missing_text_index += 1
                continue
            if text_index < 0 or text_index >= sample_count:
                continue
            if record.get("status") != "success":
                skipped_non_success += 1
                continue

            qa_id = str(record.get("qa_id") or "").strip()
            if qa_id:
                records_for_text = expanded_by_text.setdefault(text_index, {})
                if qa_id in records_for_text:
                    duplicate_qa_overwrites += 1
                records_for_text[qa_id] = record
            else:
                legacy_by_text[text_index] = record
                legacy_count_by_text[text_index] = legacy_count_by_text.get(text_index, 0) + 1

    records_by_text: dict[int, list[dict[str, Any]]] = {}
    for text_index in sorted(set(expanded_by_text) | set(legacy_by_text)):
        qa_records = expanded_by_text.get(text_index)
        if qa_records:
            if text_index in legacy_by_text:
                ignored_legacy_records += legacy_count_by_text[text_index]
            records_by_text[text_index] = [
                qa_records[qa_id]
                for qa_id in sorted(qa_records)
            ]
        else:
            records_by_text[text_index] = [legacy_by_text[text_index]]

    qa_count_by_text = {text_index: len(records) for text_index, records in records_by_text.items()}
    metric_means_by_text = {
        text_index: {
            metric: mean_or_nan(np.array([metric_value(record, metric) for record in records], dtype=float))
            for metric in METRICS
        }
        for text_index, records in records_by_text.items()
    }

    return LoadedResults(
        records_by_text=records_by_text,
        qa_count_by_text=qa_count_by_text,
        metric_means_by_text=metric_means_by_text,
        skipped_non_success=skipped_non_success,
        skipped_missing_text_index=skipped_missing_text_index,
        duplicate_qa_overwrites=duplicate_qa_overwrites,
        ignored_legacy_records=ignored_legacy_records,
    )


def metric_value(record: dict[str, Any], metric: str) -> float:
    value = record.get(metric)
    if value is None:
        return float("nan")
    return float(value)


def collect_metric(
    loaded: LoadedResults,
    sample_indices: list[int],
    metric: str,
) -> np.ndarray:
    return np.array([loaded.metric_means_by_text[index].get(metric, float("nan")) for index in sample_indices], dtype=float)


def mean_or_nan(values: np.ndarray) -> float:
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def setup_axes(ax: plt.Axes, title: str, ylabel: str, sample_indices: list[int]) -> None:
    ax.set_title(title)
    ax.set_xlabel(PLOT_TEXT["sample_index"])
    ax.set_ylabel(ylabel)
    ax.set_xticks(sample_indices)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_line(
    output_dir: Path,
    filename: str,
    metric: str,
    sample_indices: list[int],
    rakg_values: np.ndarray,
    naive_values: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.plot(sample_indices, rakg_values, marker="o", linewidth=2, label=PLOT_METHOD_NAMES["rakg"])
    ax.plot(sample_indices, naive_values, marker="s", linewidth=2, label=PLOT_METHOD_NAMES["naive"])
    setup_axes(
        ax,
        f"{PLOT_METRICS[metric]} {PLOT_TEXT['line_title_suffix']}",
        f"{PLOT_METRICS[metric]} {PLOT_TEXT['metric_mean_suffix']}",
        sample_indices,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save_figure(fig, output_dir / filename)


def plot_overview_bar(
    output_dir: Path,
    rakg_means: dict[str, float],
    naive_means: dict[str, float],
) -> None:
    labels = [PLOT_METRICS[metric] for metric in METRICS]
    rakg_values = [rakg_means[metric] for metric in METRICS]
    naive_values = [naive_means[metric] for metric in METRICS]
    positions = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(positions - width / 2, rakg_values, width, label=PLOT_METHOD_NAMES["rakg"])
    ax.bar(positions + width / 2, naive_values, width, label=PLOT_METHOD_NAMES["naive"])
    ax.set_title(PLOT_TEXT["overview_title"])
    ax.set_ylabel(PLOT_TEXT["overview_ylabel"])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    save_figure(fig, output_dir / "metrics_overview_bar.png")


def plot_win_loss(
    output_dir: Path,
    sample_indices: list[int],
    f1_diff: np.ndarray,
) -> None:
    colors = ["#2E7D32" if value > 0 else "#C62828" if value < 0 else "#6B7280" for value in f1_diff]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.bar(sample_indices, f1_diff, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.9)
    setup_axes(ax, PLOT_TEXT["win_loss_title"], PLOT_TEXT["win_loss_ylabel"], sample_indices)
    save_figure(fig, output_dir / "win_loss_by_sample.png")


def plot_judge_heatmap(
    output_dir: Path,
    sample_indices: list[int],
    values: dict[str, dict[str, np.ndarray]],
) -> None:
    rows = [
        (PLOT_TEXT["heatmap_rows"]["rakg_answer"], values["rakg"]["answer_judge"]),
        (PLOT_TEXT["heatmap_rows"]["rakg_retrieval"], values["rakg"]["retrieval_judge"]),
        (PLOT_TEXT["heatmap_rows"]["naive_answer"], values["naive"]["answer_judge"]),
        (PLOT_TEXT["heatmap_rows"]["naive_retrieval"], values["naive"]["retrieval_judge"]),
    ]
    matrix = np.vstack([row_values for _, row_values in rows])

    fig, ax = plt.subplots(figsize=(13, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title(PLOT_TEXT["heatmap_title"])
    ax.set_xlabel(PLOT_TEXT["sample_index"])
    ax.set_xticks(np.arange(len(sample_indices)))
    ax.set_xticklabels(sample_indices)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                text = PLOT_TEXT["missing"]
                text_color = "#111827"
            else:
                text = str(int(value)) if value in (0, 1) else f"{value:.2f}"
                text_color = "#FFFFFF" if value >= 0.5 else "#111827"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(PLOT_TEXT["heatmap_colorbar"])
    save_figure(fig, output_dir / "judge_heatmap.png")


def build_summary(
    sample_indices: list[int],
    values: dict[str, dict[str, np.ndarray]],
    loaded_rakg: LoadedResults,
    loaded_naive: LoadedResults,
) -> dict[str, Any]:
    rakg_means = {metric: mean_or_nan(values["rakg"][metric]) for metric in METRICS}
    naive_means = {metric: mean_or_nan(values["naive"][metric]) for metric in METRICS}
    differences = {metric: rakg_means[metric] - naive_means[metric] for metric in METRICS}
    f1_diff = values["rakg"]["official_f1"] - values["naive"]["official_f1"]

    improved = [index for index, diff in zip(sample_indices, f1_diff) if diff > 0]
    regressed = [index for index, diff in zip(sample_indices, f1_diff) if diff < 0]
    tied = [index for index, diff in zip(sample_indices, f1_diff) if diff == 0]

    return {
        "sample_count": len(sample_indices),
        "sample_indices": sample_indices,
        "qa_count_by_text": {
            "rakg": {str(index): loaded_rakg.qa_count_by_text[index] for index in sample_indices},
            "naive": {str(index): loaded_naive.qa_count_by_text[index] for index in sample_indices},
        },
        "metrics": {
            "rakg": rakg_means,
            "naive": naive_means,
            "difference": differences,
        },
        "f1_win_loss": {
            "win_count": len(improved),
            "loss_count": len(regressed),
            "tie_count": len(tied),
            "win_sample_indices": improved,
            "loss_sample_indices": regressed,
            "tie_sample_indices": tied,
        },
        "skipped": {
            "rakg_non_success": loaded_rakg.skipped_non_success,
            "naive_non_success": loaded_naive.skipped_non_success,
            "rakg_missing_text_index": loaded_rakg.skipped_missing_text_index,
            "naive_missing_text_index": loaded_naive.skipped_missing_text_index,
            "rakg_duplicate_qa_overwrites": loaded_rakg.duplicate_qa_overwrites,
            "naive_duplicate_qa_overwrites": loaded_naive.duplicate_qa_overwrites,
            "rakg_ignored_legacy_records": loaded_rakg.ignored_legacy_records,
            "naive_ignored_legacy_records": loaded_naive.ignored_legacy_records,
        },
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_path = output_dir / "analysis_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    metrics = summary["metrics"]
    win_loss = summary["f1_win_loss"]
    skipped = summary["skipped"]
    print(f"分析原始文本数：{summary['sample_count']}，文本编号：{summary['sample_indices']}")
    print(
        "每文本 QA 平均后的官方 F1："
        f"{METHOD_NAMES['rakg']}={metrics['rakg']['official_f1']:.4f}，"
        f"{METHOD_NAMES['naive']}={metrics['naive']['official_f1']:.4f}，"
        f"差值={metrics['difference']['official_f1']:.4f}"
    )
    print(
        "每文本 QA 平均后的答案 Judge："
        f"{METHOD_NAMES['rakg']}={metrics['rakg']['answer_judge']:.4f}，"
        f"{METHOD_NAMES['naive']}={metrics['naive']['answer_judge']:.4f}；"
        "检索 Judge："
        f"{METHOD_NAMES['rakg']}={metrics['rakg']['retrieval_judge']:.4f}，"
        f"{METHOD_NAMES['naive']}={metrics['naive']['retrieval_judge']:.4f}"
    )
    print(
        "按文本级平均官方 F1 比较："
        f"{METHOD_NAMES['rakg']} 胜 {win_loss['win_count']} 条，"
        f"负 {win_loss['loss_count']} 条，平 {win_loss['tie_count']} 条。"
    )
    print(
        "跳过记录："
        f"{METHOD_NAMES['rakg']} 非成功 {skipped['rakg_non_success']} 条，"
        f"{METHOD_NAMES['naive']} 非成功 {skipped['naive_non_success']} 条；"
        f"重复 qa_id 覆盖 {METHOD_NAMES['rakg']}={skipped['rakg_duplicate_qa_overwrites']}、"
        f"{METHOD_NAMES['naive']}={skipped['naive_duplicate_qa_overwrites']}。"
    )
    print(f"图表与摘要已输出到：{output_dir}")


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count 必须大于 0。")

    if not configure_matplotlib(args.font_path):
        use_english_plot_labels()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_rakg = load_jsonl(args.rakg_results, args.sample_count)
    loaded_naive = load_jsonl(args.naive_results, args.sample_count)
    sample_indices = sorted(set(loaded_rakg.records_by_text) & set(loaded_naive.records_by_text))
    if not sample_indices:
        raise RuntimeError("两套结果没有共同存在的成功原始文本，无法分析。")

    if len(sample_indices) < args.sample_count:
        print(f"警告：共同成功原始文本数为 {len(sample_indices)}，少于请求的 {args.sample_count} 条。")

    values = {
        "rakg": {
            metric: collect_metric(loaded_rakg, sample_indices, metric)
            for metric in METRICS
        },
        "naive": {
            metric: collect_metric(loaded_naive, sample_indices, metric)
            for metric in METRICS
        },
    }

    plot_line(
        output_dir,
        "official_f1_line.png",
        "official_f1",
        sample_indices,
        values["rakg"]["official_f1"],
        values["naive"]["official_f1"],
    )
    plot_line(
        output_dir,
        "answer_judge_line.png",
        "answer_judge",
        sample_indices,
        values["rakg"]["answer_judge"],
        values["naive"]["answer_judge"],
    )
    plot_line(
        output_dir,
        "retrieval_judge_line.png",
        "retrieval_judge",
        sample_indices,
        values["rakg"]["retrieval_judge"],
        values["naive"]["retrieval_judge"],
    )

    summary = build_summary(sample_indices, values, loaded_rakg, loaded_naive)
    plot_overview_bar(output_dir, summary["metrics"]["rakg"], summary["metrics"]["naive"])
    plot_win_loss(
        output_dir,
        sample_indices,
        values["rakg"]["official_f1"] - values["naive"]["official_f1"],
    )
    plot_judge_heatmap(output_dir, sample_indices, values)
    write_summary(output_dir, summary)
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
