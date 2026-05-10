import os
import json
import csv
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


def find_experiments(results_root):
    if not os.path.isdir(results_root):
        return []
    entries = []
    for name in os.listdir(results_root):
        path = os.path.join(results_root, name)
        if os.path.isdir(path):
            metrics_path = os.path.join(path, "metrics.json")
            hist_csv = os.path.join(path, "train_history.csv")
            if os.path.exists(metrics_path):
                entries.append((name, path, metrics_path, hist_csv if os.path.exists(hist_csv) else None))
    return entries


def load_metrics(entries):
    rows = []
    for name, path, metrics_path, hist_csv in entries:
        try:
            with open(metrics_path, "r") as fh:
                metrics = json.load(fh)
        except Exception:
            continue
        m = metrics.copy()
        m["exp_dir"] = name
        # infer model type and n
        m["model"] = "transformer" if "transformer" in name.lower() else ("pixelcnn" if "pixelcnn" in name.lower() else "other")
        m_n = re.search(r"n(\d+)", name)
        m["n"] = int(m_n.group(1)) if m_n else None
        # read last epoch losses if available
        m["train_loss_last"] = None
        m["val_loss_last"] = None
        if hist_csv and os.path.exists(hist_csv):
            try:
                with open(hist_csv, "r") as fh:
                    reader = csv.DictReader(fh)
                    rows_hist = list(reader)
                    if rows_hist:
                        last = rows_hist[-1]
                        m["train_loss_last"] = float(last.get("train_loss", "nan"))
                        m["val_loss_last"] = float(last.get("val_loss", "nan"))
            except Exception:
                pass
        rows.append(m)
    return rows


def write_combined_csv(rows, out_csv):
    if not rows:
        return
    keys = [
        "exp_dir",
        "model",
        "n",
        "sampler_score",
        "total_err",
        "energy_ref",
        "energy_gen",
        "energy_err",
        "mag_ref",
        "mag_gen",
        "mag_err",
        "chi_ref",
        "chi_gen",
        "corr_err",
        "chi_err",
        "hist_energy_err",
        "hist_mag_err",
        "train_loss_last",
        "val_loss_last",
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([r.get(k) for k in keys])


def plot_comparison(rows, out_png):
    if not rows:
        return
    labels = [f"{r['model']}_n{r.get('n', '?')}" for r in rows]
    scores = [r.get("sampler_score", 0.0) for r in rows]

    # stacked breakdown of error components
    comps = [
        "energy_err",
        "mag_err",
        "corr_err",
        "chi_err",
        "hist_energy_err",
        "hist_mag_err",
    ]
    comp_vals = {c: [r.get(c, 0.0) for r in rows] for c in comps}

    ind = np.arange(len(rows))
    width = 0.7
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.6), 5))
    fig.subplots_adjust(left=0.12, right=0.72, top=0.92, bottom=0.2)
    bottom = np.zeros(len(rows))
    cmap = plt.get_cmap("RdYlBu")
    colors = [cmap(i / (len(comps) + 1)) for i in range(len(comps))]
    for i, c in enumerate(comps):
        vals = np.array(comp_vals[c], dtype=float)
        ax.bar(ind, vals, width, bottom=bottom, label=c, color=colors[i])
        bottom += vals

    # overlay sampler_score as a line
    ax2 = ax.twinx()
    ax2.plot(ind, scores, color="k", marker="o", linestyle="-", label="sampler_score")
    ax2.set_ylabel("sampler_score")

    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("error component (stacked)")
    ax.set_title("Per-component error breakdown and sampler_score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.10, 1))
    ax2.legend(loc="upper left", bbox_to_anchor=(1.10, 0.6))
    fig.savefig(out_png)
    plt.close(fig)


def regenerate_loss_plots(entries):
    for name, path, metrics_path, hist_csv in entries:
        if hist_csv and os.path.exists(hist_csv):
            try:
                import pandas as pd

                df = pd.read_csv(hist_csv)
                xs = df["epoch"].values
                ys_train = df["train_loss"].values
                ys_val = df["val_loss"].values
                plt.figure()
                cmap = plt.get_cmap("RdYlBu")
                ctrain = cmap(0.15)
                cval = cmap(0.85)
                plt.plot(xs, ys_train, label="train", color=ctrain, marker="o")
                plt.plot(xs, ys_val, label="val", color=cval, marker="o")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                ymin, ymax = min(ys_train.min(), ys_val.min()), max(ys_train.max(), ys_val.max())
                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5
                margin = (ymax - ymin) * 0.1
                plt.ylim(ymin - margin, ymax + margin)
                plt.tight_layout()
                out_png = os.path.join(path, "loss_curve.png")
                plt.savefig(out_png)
                plt.close()
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=os.path.join("ising", "ising_results"))
    parser.add_argument("--out-csv", default=os.path.join("ising", "ising_results", "comparison.csv"))
    parser.add_argument("--out-png", default=os.path.join("ising", "ising_results", "comparison_plot.png"))
    args = parser.parse_args()

    entries = find_experiments(args.results_root)
    rows = load_metrics(entries)
    # sort by model then n
    rows = sorted(rows, key=lambda r: (r.get("model", ""), r.get("n") or 0))
    os.makedirs(args.results_root, exist_ok=True)
    write_combined_csv(rows, args.out_csv)
    # regenerate per-experiment loss plots to ensure visibility
    regenerate_loss_plots(entries)
    plot_comparison(rows, args.out_png)
    print(f"Wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
