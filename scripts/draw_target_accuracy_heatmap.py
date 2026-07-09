from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a target overall accuracy heatmap."
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=80.0,
        help="Target overall accuracy percentage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/target_overall_accuracy_heatmap_80.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    accuracy = max(0.0, min(100.0, args.accuracy))
    data = np.array([[accuracy]])

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

    ax.set_title(f"Target Overall Accuracy Heatmap ({accuracy:.0f}%)")
    ax.set_xticks([0], ["Overall Accuracy"])
    ax.set_yticks([])

    ax.text(
        0,
        0,
        f"{accuracy:.1f}%",
        ha="center",
        va="center",
        color="black",
        fontsize=18,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Percent")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"output={args.output}")
    print(f"accuracy={accuracy:.1f}")


if __name__ == "__main__":
    main()
