"""Turn results.json into the tables and figures that go in the README.

The headline this report exists to deliver is the gap between two protocols run
on identical data: a random split, where 40% of test images have a near-duplicate
twin in training, and a group-aware split where none do.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (light surface); see dataviz palette reference.
# Slots are assigned in fixed order and never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

HONEST = SERIES[0]   # group-aware
LEAKY = SERIES[1]    # random split


def _style(ax):
    """Recessive axes: the data is the ink, the frame is not."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _fig(width=7.5, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, _style(ax)


def build_comparison_table(grouped: dict, random_: dict | None) -> list[dict]:
    """One row per model: honest accuracy, leaky accuracy, and the inflation between."""
    rows = []
    random_holdout = (random_ or {}).get("holdout", {})
    for name, metrics in grouped.get("holdout", {}).items():
        leaky = random_holdout.get(name)
        g_acc = metrics.get("accuracy")
        r_acc = leaky.get("accuracy") if leaky else None
        rows.append(
            {
                "model": name,
                "params": metrics.get("params"),
                "grouped_accuracy": g_acc,
                "grouped_f1": metrics.get("f1_macro"),
                "random_accuracy": r_acc,
                "inflation": (r_acc - g_acc) if (r_acc is not None and g_acc is not None) else None,
            }
        )
    return sorted(rows, key=lambda r: r["grouped_accuracy"] or 0, reverse=True)


def markdown_table(rows: list[dict]) -> str:
    """Render the comparison as a GitHub table. Missing arms render as an em dash."""
    def pct(v):
        return "—" if v is None else f"{100 * v:.1f}"

    def params(v):
        return "—" if v is None else (f"{v / 1e6:.2f} M" if v >= 1e6 else f"{v / 1e3:.0f} K")

    def f1(v):
        return "—" if v is None else f"{v:.3f}"

    def inflation(v):
        return "—" if v is None else f"+{100 * v:.1f}"

    lines = [
        "| Model | Params | Group-aware acc. (%) | Group-aware F1 | Random-split acc. (%) | Inflation (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {params(r['params'])} | {pct(r['grouped_accuracy'])} "
            f"| {f1(r['grouped_f1'])} | {pct(r['random_accuracy'])} | {inflation(r['inflation'])} |"
        )
    return "\n".join(lines) + "\n"


def summarise(grouped: dict, random_: dict | None, leaked_fraction: float) -> str:
    """A short prose statement of the finding, with the numbers filled in."""
    rows = build_comparison_table(grouped, random_)
    lx = next((r for r in rows if r["model"] == "LXNet"), None)
    best = rows[0] if rows else None
    parts = [
        f"Under a random split {100 * leaked_fraction:.1f}% of test images have a "
        f"near-duplicate in the training set; under the group-aware split, none do."
    ]
    if lx and lx["grouped_accuracy"] is not None:
        parts.append(
            f"LXNet scores {100 * lx['grouped_accuracy']:.1f}% on the group-aware test set"
            + (
                f", against {100 * lx['random_accuracy']:.1f}% on the random split "
                f"(+{100 * lx['inflation']:.1f} pp of leakage)."
                if lx["random_accuracy"] is not None
                else "."
            )
        )
    if best and best["model"] != "LXNet":
        parts.append(f"The strongest model overall is {best['model']} at {100 * best['grouped_accuracy']:.1f}%.")
    return " ".join(parts)


def _bar_leakage(rows: list[dict], out: Path) -> Path | None:
    paired = [r for r in rows if r["random_accuracy"] is not None]
    if not paired:
        return None
    fig, ax = _fig(7.5, 4.0)
    x = np.arange(len(paired))
    w = 0.36
    g = [100 * r["grouped_accuracy"] for r in paired]
    rnd = [100 * r["random_accuracy"] for r in paired]
    # 2px surface gap between adjacent fills
    ax.bar(x - w / 2, g, w, label="Group-aware split (honest)", color=HONEST, edgecolor=SURFACE, linewidth=2)
    ax.bar(x + w / 2, rnd, w, label="Random split (paper protocol)", color=LEAKY, edgecolor=SURFACE, linewidth=2)
    # direct labels: the palette's contrast WARN obliges visible values
    for xi, v in zip(x - w / 2, g):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2, rnd):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [r["model"] for r in paired], color=INK)
    ax.set_ylabel("Test accuracy (%)", color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_title("Near-duplicate leakage inflates every model", color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _bar_models(rows: list[dict], out: Path) -> Path:
    fig, ax = _fig(7.5, 4.0)
    labels = [r["model"] for r in rows]
    vals = [100 * (r["grouped_accuracy"] or 0) for r in rows]
    ax.barh(np.arange(len(rows)), vals, 0.62, color=HONEST, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(rows)), labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Test accuracy (%), group-aware split", color=INK_MUTED, fontsize=9)
    ax.set_title("Honest accuracy by model", color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _cv_folds(grouped: dict, out: Path) -> Path | None:
    cv = grouped.get("cross_validation") or {}
    if not cv:
        return None
    fig, ax = _fig(6.4, 4.0)
    names = list(cv)
    for i, name in enumerate(names):
        accs = [100 * f["accuracy"] for f in cv[name]["folds"]]
        ax.scatter([i] * len(accs), accs, s=64, color=SERIES[i % len(SERIES)],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        m = float(np.mean(accs))
        ax.plot([i - 0.22, i + 0.22], [m, m], color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.28, m, f"mean {m:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), names, color=INK)
    ax.set_xlim(-0.5, len(names) - 0.1)
    ax.set_ylabel("Fold accuracy (%)", color=INK_MUTED, fontsize=9)
    ax.set_title("Cross-validation folds, group-aware", color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_figures(grouped: dict, random_: dict | None, out_dir: Path) -> list[Path]:
    """Write every figure the README references. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_comparison_table(grouped, random_)
    written = [_bar_models(rows, out_dir / "fig_accuracy_by_model.png")]
    for maybe in (
        _bar_leakage(rows, out_dir / "fig_leakage.png"),
        _cv_folds(grouped, out_dir / "fig_cv_folds.png"),
    ):
        if maybe is not None:
            written.append(maybe)

    from .data import CLASS_LABELS

    names = list(CLASS_LABELS.values())
    for model, metrics in grouped.get("holdout", {}).items():
        cm = metrics.get("confusion_matrix")
        if cm:
            written.append(
                confusion_figure(
                    np.array(cm),
                    names[: len(cm)],
                    out_dir / f"fig_confusion_{model}.png",
                    f"{model} — group-aware test set (row %)",
                )
            )
    return written


def confusion_figure(matrix: np.ndarray, class_names: list[str], out: Path, title: str) -> Path:
    """Sequential single-hue heatmap; row-normalised so classes are comparable."""
    matrix = np.asarray(matrix, dtype=float)
    norm = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8, color=INK_MUTED)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8, color=INK_MUTED)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("True", color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of true class")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def leakage_fraction(data_root: str | Path = "dataset", seed: int = 42) -> float:
    """Share of random-split test images having a near-duplicate in training."""
    from .data import dedupe, group_samples, index_dataset, stratified_split

    samples, _ = dedupe(index_dataset(data_root))
    group_of = {s.path: g for g, members in group_samples(samples).items() for s in members}
    parts = stratified_split(samples, (0.70, 0.15, 0.15), seed=seed)
    train_groups = {group_of[s.path] for s in parts["train"]}
    leaked = sum(1 for s in parts["test"] if group_of[s.path] in train_groups)
    return leaked / len(parts["test"])


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build figures and the results table.")
    parser.add_argument("--grouped", type=Path, default=Path("runs/grouped/results.json"))
    parser.add_argument("--random", type=Path, default=Path("runs/random/results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    args = parser.parse_args(argv)

    grouped = load(args.grouped)
    random_ = load(args.random) if args.random.exists() else None

    figures = write_figures(grouped, random_, args.out_dir)
    rows = build_comparison_table(grouped, random_)
    leaked = leakage_fraction(args.data_root)

    body = [
        "# Results",
        "",
        summarise(grouped, random_, leaked),
        "",
        "## Hold-out comparison",
        "",
        markdown_table(rows),
        "## Dataset after cleaning",
        "",
        f"- images indexed: {grouped['dedupe']['kept'] + grouped['dedupe']['exact_duplicates']}",
        f"- byte-identical duplicates removed: {grouped['dedupe']['exact_duplicates']}",
        f"- cross-class label conflicts removed: {grouped['dedupe']['cross_class_conflicts']}",
        f"- split (train/val/test): {grouped['split']['train']}/{grouped['split']['val']}/{grouped['split']['test']}",
        f"- random-split test images with a training twin: {100 * leaked:.1f}%",
        "",
        "## Figures",
        "",
    ]
    body += [f"![{p.stem}]({p.name})" for p in figures]
    (args.out_dir / "RESULTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'RESULTS.md'} and {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
