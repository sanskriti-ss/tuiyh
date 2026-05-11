import argparse
import math
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ising.ising_transformer import (
    IsingTransformer,
    generate_ising_samples,
    spins_to_tokens,
    energy_per_spin,
    magnetization,
    correlation_function,
    batch_iter,
)


def train_for_temperature(cfg, temperature, device):
    beta = 1.0 / temperature
    train = generate_ising_samples(cfg.num_train, cfg.n, beta, cfg.burn_in, cfg.thin, device)
    val = generate_ising_samples(cfg.num_val, cfg.n, beta, cfg.burn_in, cfg.thin, device)
    train_tokens = spins_to_tokens(train).view(cfg.num_train, cfg.n * cfg.n)
    val_tokens = spins_to_tokens(val).view(cfg.num_val, cfg.n * cfg.n)

    model = IsingTransformer(cfg.n, cfg.d_model, cfg.nhead, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start_token = torch.full((1,), 2, device=device, dtype=torch.long)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in batch_iter(train_tokens, cfg.batch_size, shuffle=True):
            x_in = torch.cat([start_token.expand(batch.shape[0], 1), batch[:, :-1]], dim=1)
            logits = model(x_in)
            loss = F.cross_entropy(logits.reshape(-1, 2), batch.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in batch_iter(val_tokens, cfg.batch_size, shuffle=False):
                x_in = torch.cat([start_token.expand(batch.shape[0], 1), batch[:, :-1]], dim=1)
                logits = model(x_in)
                loss = F.cross_entropy(logits.reshape(-1, 2), batch.reshape(-1))
                val_loss += loss.item()

        print(f"T={temperature:.3f} epoch={epoch+1} train_loss={total_loss:.4f} val_loss={val_loss:.4f}")

    return model


def sample_model(model, num_samples, n, device):
    model.eval()
    seq_len = n * n
    tokens = torch.full((num_samples, seq_len), 2, device=device, dtype=torch.long)
    with torch.no_grad():
        for idx in range(seq_len):
            logits = model(tokens)
            probs = torch.softmax(logits[:, idx, :], dim=-1)
            tokens[:, idx] = torch.multinomial(probs, num_samples=1).squeeze(1)
    spins = (tokens.float() * 2.0 - 1.0).view(num_samples, n, n)
    return spins


def plot_curve(temps, ref_vals, gen_vals, ylabel, outpath):
    plt.figure()
    plt.plot(temps, ref_vals, marker="o", label="ref")
    plt.plot(temps, gen_vals, marker="o", label="gen")
    plt.xlabel("temperature")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_corr_curves(temps, ref_corr, gen_corr, outpath):
    # ref_corr/gen_corr shape: (num_t, max_r)
    plt.figure()
    for r in range(ref_corr.shape[1]):
        plt.plot(temps, ref_corr[:, r], marker="o", label=f"ref r={r+1}")
        plt.plot(temps, gen_corr[:, r], marker="x", linestyle="--", label=f"gen r={r+1}")
    plt.xlabel("temperature")
    plt.ylabel("G(r)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--temps", type=str, default="1.5,2.0,2.269,2.5,3.0")
    parser.add_argument("--num-train", type=int, default=400)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--burn-in", type=int, default=80)
    parser.add_argument("--thin", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


class Cfg:
    def __init__(self, args):
        self.n = args.n
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.burn_in = args.burn_in
        self.thin = args.thin
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.layers = args.layers
        self.dropout = args.dropout


def main():
    args = parse_args()
    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    cfg = Cfg(args)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )

    results_dir = os.path.join("ising", "ising_results", f"temp_sweep_n{cfg.n}_{int(time.time())}")
    os.makedirs(results_dir, exist_ok=True)

    max_r = max(1, cfg.n // 4)
    ref_M, gen_M = [], []
    ref_E, gen_E = [], []
    ref_G, gen_G = [], []

    for t in temps:
        print(f"=== temperature {t:.3f} ===")
        model = train_for_temperature(cfg, t, device)
        ref = generate_ising_samples(cfg.num_val, cfg.n, 1.0 / t, cfg.burn_in, cfg.thin, device)
        gen = sample_model(model, cfg.num_val, cfg.n, device)

        ref_M.append(magnetization(ref).mean().item())
        gen_M.append(magnetization(gen).mean().item())
        ref_E.append(energy_per_spin(ref).mean().item())
        gen_E.append(energy_per_spin(gen).mean().item())
        ref_G.append(correlation_function(ref, max_r).mean(dim=0).cpu().numpy())
        gen_G.append(correlation_function(gen, max_r).mean(dim=0).cpu().numpy())

    ref_G = np.stack(ref_G, axis=0)
    gen_G = np.stack(gen_G, axis=0)

    plot_curve(temps, ref_M, gen_M, "M(T)", os.path.join(results_dir, "main_M_T.png"))
    plot_curve(temps, ref_E, gen_E, "E(T)", os.path.join(results_dir, "main_E_T.png"))
    plot_corr_curves(temps, ref_G, gen_G, os.path.join(results_dir, "main_G_r_T.png"))

    summary = {
        "temps": temps,
        "ref_M": ref_M,
        "gen_M": gen_M,
        "ref_E": ref_E,
        "gen_E": gen_E,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Wrote sweep results to", results_dir)


if __name__ == "__main__":
    main()
