from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw an illustrative target-accuracy graph."
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=80.0,
        help="Illustrative target accuracy percentage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/target_accuracy_80.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    accuracy = max(0.0, min(100.0, args.accuracy))
    remaining = 100.0 - accuracy

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    bars = ax.bar(
        ["Accuracy", "Remaining"],
        [accuracy, remaining],
        color=["#2A9D8F", "#D9D9D9"],
        width=0.6,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title("Illustrative Overall Accuracy Graph")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.text(
        0.01,
        0.01,
        "Illustrative chart only. This is a presentation target, not a measured evaluation result.",
        fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"output={args.output}")
    print(f"accuracy={accuracy:.1f}")


if __name__ == "__main__":
    main()
