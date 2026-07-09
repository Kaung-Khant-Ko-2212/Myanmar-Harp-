from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def normalize_component_status(status: str | None) -> str:
    return "strike" if status == "strike" else "touch_only"


def compute_metrics(tp: int, fn: int, fp: int, tn: int) -> dict[str, float]:
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
    }


def load_summary(input_dir: Path) -> dict:
    files = sorted(input_dir.glob("*_right_av_decision_events.json"))
    if not files:
        raise FileNotFoundError(f"No *_right_av_decision_events.json files found in {input_dir}")

    fusion_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    audio_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    beat_counts: Counter[str] = Counter()
    finger_counts: Counter[str] = Counter()
    cm_video: Counter[tuple[str, str]] = Counter()
    cm_audio: Counter[tuple[str, str]] = Counter()

    total_events = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for event in data.get("events", []):
            fusion_status = normalize_component_status(event.get("fusion", {}).get("status"))
            video_status = normalize_component_status(event.get("video", {}).get("status"))
            audio_status = normalize_component_status(event.get("audio", {}).get("status"))

            fusion_counts[fusion_status] += 1
            video_counts[video_status] += 1
            audio_counts[audio_status] += 1
            strategy_counts[str(event.get("fusion", {}).get("strategy", "unknown"))] += 1
            beat_counts[str(event.get("fusion", {}).get("beat_label", "unknown"))] += 1
            finger_counts[str(event.get("touch", {}).get("finger_type", "unknown"))] += 1
            cm_video[(video_status, fusion_status)] += 1
            cm_audio[(audio_status, fusion_status)] += 1
            total_events += 1

    video_tp = cm_video[("strike", "strike")]
    video_fn = cm_video[("touch_only", "strike")]
    video_fp = cm_video[("strike", "touch_only")]
    video_tn = cm_video[("touch_only", "touch_only")]

    audio_tp = cm_audio[("strike", "strike")]
    audio_fn = cm_audio[("touch_only", "strike")]
    audio_fp = cm_audio[("strike", "touch_only")]
    audio_tn = cm_audio[("touch_only", "touch_only")]

    return {
        "clips_analyzed": len(files),
        "events_analyzed": total_events,
        "fusion_status_counts": dict(fusion_counts),
        "video_status_counts": dict(video_counts),
        "audio_status_counts": dict(audio_counts),
        "fusion_strategy_counts": dict(strategy_counts),
        "beat_label_counts": dict(beat_counts),
        "finger_type_counts": dict(finger_counts),
        "video_confusion_matrix": {
            "tp": video_tp,
            "fn": video_fn,
            "fp": video_fp,
            "tn": video_tn,
        },
        "audio_confusion_matrix": {
            "tp": audio_tp,
            "fn": audio_fn,
            "fp": audio_fp,
            "tn": audio_tn,
        },
        "video_metrics": compute_metrics(video_tp, video_fn, video_fp, video_tn),
        "audio_metrics": compute_metrics(audio_tp, audio_fn, audio_fp, audio_tn),
        "evaluation_note": (
            "Proxy evaluation only: AV fusion status is treated as the target label because the "
            "repository does not contain external human-annotated strike ground truth."
        ),
    }


def write_chart(summary: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "specificity"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    video_values = [summary["video_metrics"][key] * 100 for key in metric_keys]
    audio_values = [summary["audio_metrics"][key] * 100 for key in metric_keys]

    fusion_counts = summary["fusion_status_counts"]
    strike_count = fusion_counts.get("strike", 0)
    touch_only_count = fusion_counts.get("touch_only", 0)

    x = np.arange(len(metric_keys))
    width = 0.34

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    ax = axes[0]
    bars_video = ax.bar(x - width / 2, video_values, width, label="Video rule", color="#355C7D")
    bars_audio = ax.bar(x + width / 2, audio_values, width, label="Audio rule", color="#2A9D8F")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Percent")
    ax.set_title(
        f"Proxy Performance vs AV Fusion Label\n"
        f"{summary['clips_analyzed']} clips, {summary['events_analyzed']:,} events"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars_video, bars_audio):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax2 = axes[1]
    labels = ["Strike", "Touch only"]
    values = [strike_count, touch_only_count]
    colors = ["#E76F51", "#A8DADC"]
    bars = ax2.bar(labels, values, color=colors, width=0.6)
    ax2.set_ylabel("Events")
    ax2.set_title("AV Fusion Output Distribution")
    ax2.grid(axis="y", alpha=0.25)

    for bar in bars:
        height = int(bar.get_height())
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(10, summary["events_analyzed"] * 0.002),
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute proxy performance for the Myanmar harp AV pipeline and draw a summary graph."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("backend/touch_events/best"),
        help="Directory containing *_right_av_decision_events.json files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/performance_summary.json"),
        help="Where to save the computed summary JSON.",
    )
    parser.add_argument(
        "--chart-png",
        type=Path,
        default=Path("outputs/performance_summary.png"),
        help="Where to save the generated chart.",
    )
    args = parser.parse_args()

    summary = load_summary(args.input_dir)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_chart(summary, args.chart_png)

    print(f"clips_analyzed={summary['clips_analyzed']}")
    print(f"events_analyzed={summary['events_analyzed']}")
    print(f"summary_json={args.summary_json}")
    print(f"chart_png={args.chart_png}")


if __name__ == "__main__":
    main()
