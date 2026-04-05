#!/usr/bin/env python3
"""
plot_results.py — Generate benchmark result charts from JSON result files.

Usage:
  python3 scripts/plot_results.py benchmarks/YYYY-MM-DD-results.json

The JSON file should contain a list of experiment dicts. Each experiment:
{
  "name": "SGLang piecewise CUDA graph",
  "date": "2026-04-05",
  "model": "Qwen/Qwen3-0.6B",
  "hardware": "spark-01 GB10 SM121",
  "conditions": [
    {
      "label": "Standard CUDA graph",
      "color": "#4A90D9",
      "offline": {"ttft_p50": 4367, "ttft_p99": 5241, "tpot_p50": 39.3, "tpot_p99": 126.0, "throughput": 2217},
      "online":  {"ttft_p50": 29.8,  "ttft_p99": 44.7,  "tpot_p50": 9.50, "tpot_p99": 11.0}
    },
    ...
  ],
  "notes": ["note1", "note2"]
}

Outputs: benchmarks/charts/YYYY-MM-DD-<slug>.png
Also updates the corresponding benchmarks/YYYY-MM-DD-results.md to embed the chart.
"""

import json
import sys
import re
import datetime
import pathlib
import textwrap

REPO_ROOT = pathlib.Path(__file__).parent.parent


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def plot_experiment(exp: dict, out_path: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    conditions = exp["conditions"]
    date = exp.get("date", datetime.date.today().isoformat())
    model = exp.get("model", "")
    hardware = exp.get("hardware", "")
    notes = exp.get("notes", [])

    colors = [c.get("color", f"C{i}") for i, c in enumerate(conditions)]
    labels = [c["label"] for c in conditions]

    fig = plt.figure(figsize=(14, 10), facecolor="#0F1117")
    title = f"{exp['name']}\n{model}  ·  {hardware}  ·  {date}"
    fig.suptitle(title, fontsize=12, color="white", fontweight="bold", y=0.98)

    def style_ax(ax, title_str):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA")
        ax.set_title(title_str, color="#DDDDDD", fontsize=9.5, pad=7)
        ax.spines[:].set_color("#2A2D3A")
        ax.yaxis.label.set_color("#AAAAAA")
        ax.grid(axis="y", color="#2A2D3A", linewidth=0.7, zorder=0)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    x = np.arange(2)  # p50, p99
    w = 0.8 / len(conditions)
    offsets = np.linspace(-(len(conditions) - 1) * w / 2,
                          (len(conditions) - 1) * w / 2, len(conditions))

    def bar_group(ax, metric_key, ylabel, panel_title, scale=1.0):
        for i, (cond, color, offset) in enumerate(zip(conditions, colors, offsets)):
            vals = [cond["offline"].get(f"{metric_key}_p50", 0) * scale,
                    cond["offline"].get(f"{metric_key}_p99", 0) * scale]
            bars = ax.bar(x + offset, vals, w, color=color, zorder=3, label=cond["label"])
            ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals],
                         color="white", fontsize=7.5, padding=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["p50", "p99"], color="#AAAAAA")
        ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
        style_ax(ax, panel_title)

    # ── Throughput ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    tput_vals = [c["offline"].get("throughput", 0) for c in conditions]
    bars = ax1.bar(labels, tput_vals, color=colors, width=0.5, zorder=3)
    ax1.bar_label(bars, labels=[f"{v:,.0f}" for v in tput_vals],
                  color="white", fontsize=10, fontweight="bold", padding=4)
    ax1.set_ylabel("tokens / sec", color="#AAAAAA", fontsize=9)
    ax1.set_ylim(0, max(tput_vals) * 1.25)
    ax1.tick_params(axis="x", colors="#AAAAAA", labelsize=8)
    if len(tput_vals) == 2 and tput_vals[0] > 0:
        delta = (tput_vals[-1] - tput_vals[0]) / tput_vals[0] * 100
        sign = "+" if delta >= 0 else ""
        ax1.annotate(f"{sign}{delta:.0f}%", xy=(len(conditions) - 1, tput_vals[-1]),
                     xytext=(len(conditions) - 0.6, tput_vals[-1] * 1.08),
                     color=colors[-1], fontsize=13, fontweight="bold")
    style_ax(ax1, "Offline Throughput (tok/s)")

    # ── TTFT offline ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 0])
    bar_group(ax2, "ttft", "ms", "Offline TTFT (lower = better)")
    max_ttft = max(c["offline"].get("ttft_p99", 0) for c in conditions)
    ax2.set_ylim(0, max_ttft * 1.2)

    # ── TPOT offline ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bar_group(ax3, "tpot", "ms / token", "Offline TPOT (lower = better)")
    max_tpot = max(c["offline"].get("tpot_p99", 0) for c in conditions)
    ax3.set_ylim(0, max_tpot * 1.2)

    # ── Online TTFT ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (cond, color, offset) in enumerate(zip(conditions, colors, offsets)):
        vals = [cond["online"].get("ttft_p50", 0), cond["online"].get("ttft_p99", 0)]
        bars = ax4.bar(x + offset, vals, w, color=color, zorder=3)
        ax4.bar_label(bars, labels=[f"{v:.1f}" for v in vals],
                      color="white", fontsize=7.5, padding=3)
    ax4.set_xticks(x); ax4.set_xticklabels(["p50", "p99"], color="#AAAAAA")
    ax4.set_ylabel("ms", color="#AAAAAA", fontsize=9)
    max_on_ttft = max(c["online"].get("ttft_p99", 0) for c in conditions)
    ax4.set_ylim(0, max_on_ttft * 1.3)
    style_ax(ax4, "Online TTFT (low load)")

    # ── Online TPOT ──────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (cond, color, offset) in enumerate(zip(conditions, colors, offsets)):
        vals = [cond["online"].get("tpot_p50", 0), cond["online"].get("tpot_p99", 0)]
        bars = ax5.bar(x + offset, vals, w, color=color, zorder=3)
        ax5.bar_label(bars, labels=[f"{v:.2f}" for v in vals],
                      color="white", fontsize=7.5, padding=3)
    ax5.set_xticks(x); ax5.set_xticklabels(["p50", "p99"], color="#AAAAAA")
    ax5.set_ylabel("ms / token", color="#AAAAAA", fontsize=9)
    max_on_tpot = max(c["online"].get("tpot_p99", 0) for c in conditions)
    ax5.set_ylim(0, max_on_tpot * 1.3)
    style_ax(ax5, "Online TPOT (low load)")

    # ── Notes panel ──────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1A1D27")
    ax6.axis("off")
    ax6.set_title("Notes", color="#DDDDDD", fontsize=10, pad=8)
    note_colors = ["white", "#AAAAAA", "#F5A623"]
    for i, note in enumerate(notes[:10]):
        wrapped = textwrap.fill(note, width=38)
        color = note_colors[min(i // 3, len(note_colors) - 1)]
        ax6.text(0.05, 0.95 - i * 0.095, wrapped,
                 transform=ax6.transAxes, color=color,
                 fontsize=7.8, verticalalignment="top", fontfamily="monospace")

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=patches, loc="lower center", ncol=len(conditions),
               fancybox=True, framealpha=0.15, labelcolor="white",
               fontsize=10, bbox_to_anchor=(0.5, 0.005))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  chart -> {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_results.py <results.json>")
        sys.exit(1)

    results_json = pathlib.Path(sys.argv[1])
    experiments = json.loads(results_json.read_text())
    if isinstance(experiments, dict):
        experiments = [experiments]

    charts = []
    for exp in experiments:
        date = exp.get("date", datetime.date.today().isoformat())
        slug = slugify(exp.get("name", "experiment"))
        out_path = REPO_ROOT / "benchmarks" / "charts" / f"{date}-{slug}.png"
        plot_experiment(exp, out_path)
        charts.append(out_path)

    print(f"Generated {len(charts)} chart(s).")
    return charts


if __name__ == "__main__":
    main()
