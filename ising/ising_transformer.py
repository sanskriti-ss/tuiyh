import argparse
import math
import time
from dataclasses import dataclass
import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def critical_temperature():
    return 2.0 / math.log(1.0 + math.sqrt(2.0))


def make_checkerboard_mask(n, device):
    coords = torch.arange(n, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    mask_even = ((xx + yy) % 2 == 0).unsqueeze(0)
    mask_odd = ~mask_even
    return mask_even, mask_odd


def metropolis_sweep(spins, beta, mask_even, mask_odd):
    for mask in (mask_even, mask_odd):
        neigh = (
            torch.roll(spins, shifts=1, dims=1)
            + torch.roll(spins, shifts=-1, dims=1)
            + torch.roll(spins, shifts=1, dims=2)
            + torch.roll(spins, shifts=-1, dims=2)
        )
        delta_e = 2.0 * spins * neigh
        accept = (delta_e <= 0) | (torch.rand_like(spins) < torch.exp(-beta * delta_e))
        flips = mask & accept
        spins = torch.where(flips, -spins, spins)
    return spins


def generate_ising_samples(num_samples, n, beta, burn_in, thin, device):
    spins = torch.randint(0, 2, (1, n, n), device=device, dtype=torch.int8) * 2 - 1
    spins = spins.to(torch.float32)
    mask_even, mask_odd = make_checkerboard_mask(n, device)

    for _ in range(burn_in):
        spins = metropolis_sweep(spins, beta, mask_even, mask_odd)

    samples = []
    for _ in range(num_samples):
        for _ in range(thin):
            spins = metropolis_sweep(spins, beta, mask_even, mask_odd)
        samples.append(spins.clone())
    return torch.cat(samples, dim=0)


def spins_to_tokens(spins):
    return ((spins + 1) / 2).to(torch.long)


def tokens_to_spins(tokens):
    return tokens.to(torch.float32) * 2.0 - 1.0


class IsingTransformer(nn.Module):
    def __init__(self, n, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.n = n
        self.seq_len = n * n
        self.vocab = 3
        self.token_emb = nn.Embedding(self.vocab, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens):
        batch, length = tokens.shape
        emb = self.token_emb(tokens) + self.pos_emb[:, :length, :]
        mask = torch.triu(torch.ones(length, length, device=tokens.device), diagonal=1).bool()
        out = self.transformer(emb, mask)
        logits = self.head(out)
        return logits


@dataclass
class TrainConfig:
    n: int = 16
    num_train: int = 5000
    num_val: int = 1000
    burn_in: int = 200
    thin: int = 20
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    dropout: float = 0.1


def make_datasets(cfg, device):
    beta = 1.0 / critical_temperature()
    train = generate_ising_samples(cfg.num_train, cfg.n, beta, cfg.burn_in, cfg.thin, device)
    val = generate_ising_samples(cfg.num_val, cfg.n, beta, cfg.burn_in, cfg.thin, device)
    train_tokens = spins_to_tokens(train).view(cfg.num_train, cfg.n * cfg.n)
    val_tokens = spins_to_tokens(val).view(cfg.num_val, cfg.n * cfg.n)
    return train_tokens, val_tokens


def batch_iter(tokens, batch_size, shuffle=True):
    idx = torch.randperm(tokens.shape[0], device=tokens.device) if shuffle else torch.arange(tokens.shape[0], device=tokens.device)
    for start in range(0, tokens.shape[0], batch_size):
        batch_idx = idx[start : start + batch_size]
        yield tokens[batch_idx]


def train_model(cfg, device):
    train_tokens, val_tokens = make_datasets(cfg, device)
    model = IsingTransformer(cfg.n, cfg.d_model, cfg.nhead, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_token = torch.full((1,), 2, device=device, dtype=torch.long)

    history = {"train_loss": [], "val_loss": []}

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

        history["train_loss"].append(total_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch+1} train_loss={total_loss:.4f} val_loss={val_loss:.4f}")

    return model, history


def sample_model(model, num_samples, n, device):
    model.eval()
    seq_len = n * n
    tokens = torch.full((num_samples, seq_len), 2, device=device, dtype=torch.long)

    with torch.no_grad():
        for idx in range(seq_len):
            logits = model(tokens)
            probs = torch.softmax(logits[:, idx, :], dim=-1)
            tokens[:, idx] = torch.multinomial(probs, num_samples=1).squeeze(1)

    spins = tokens_to_spins(tokens).view(num_samples, n, n)
    return spins


def energy_per_spin(spins):
    right = torch.roll(spins, shifts=-1, dims=2)
    down = torch.roll(spins, shifts=-1, dims=1)
    energy = -(spins * (right + down)).mean(dim=(1, 2))
    return energy


def magnetization(spins):
    return spins.mean(dim=(1, 2))


def magnetization_per_spin(spins):
    return magnetization(spins).abs()


def susceptibility(spins, temperature):
    n = spins.shape[1]
    m = magnetization(spins)
    m_mean = m.mean()
    return (n * n) * (m.pow(2).mean() - m_mean.pow(2)) / temperature


def histogram_distance(a, b, bins, min_val, max_val):
    hist_a = torch.histc(a, bins=bins, min=min_val, max=max_val)
    hist_b = torch.histc(b, bins=bins, min=min_val, max=max_val)
    hist_a = hist_a / (hist_a.sum() + 1e-6)
    hist_b = hist_b / (hist_b.sum() + 1e-6)
    return (hist_a - hist_b).abs().sum()


def wasserstein_1d(a, b):
    # 1D Wasserstein-1 distance via sorted samples
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(0.0, device=a.device)
    n = min(a.numel(), b.numel())
    a_sorted = torch.sort(a.view(-1))[0][:n]
    b_sorted = torch.sort(b.view(-1))[0][:n]
    return (a_sorted - b_sorted).abs().mean()


def correlation_function(spins, max_r):
    corrs = []
    for r in range(1, max_r + 1):
        corr_x = (spins * torch.roll(spins, shifts=r, dims=2)).mean(dim=(1, 2))
        corr_y = (spins * torch.roll(spins, shifts=r, dims=1)).mean(dim=(1, 2))
        corrs.append(0.5 * (corr_x + corr_y))
    return torch.stack(corrs, dim=1)


def evaluate_sampler(model, cfg, device):
    tc = critical_temperature()
    beta = 1.0 / tc
    ref = generate_ising_samples(cfg.num_val, cfg.n, beta, cfg.burn_in, cfg.thin, device)
    gen = sample_model(model, cfg.num_val, cfg.n, device)

    ref_energy = energy_per_spin(ref)
    gen_energy = energy_per_spin(gen)
    ref_mag = magnetization_per_spin(ref)
    gen_mag = magnetization_per_spin(gen)
    ref_chi = susceptibility(ref, tc)
    gen_chi = susceptibility(gen, tc)

    max_r = max(1, cfg.n // 4)
    ref_corr = correlation_function(ref, max_r).mean(dim=0)
    gen_corr = correlation_function(gen, max_r).mean(dim=0)

    energy_err = (gen_energy.mean() - ref_energy.mean()).abs() / (ref_energy.mean().abs() + 1e-6)
    mag_err = (gen_mag.mean() - ref_mag.mean()).abs() / (ref_mag.mean() + 1e-6)
    corr_err = (gen_corr - ref_corr).abs().mean() / (ref_corr.abs().mean() + 1e-6)
    chi_err = (gen_chi - ref_chi).abs() / (ref_chi.abs() + 1e-6)
    hist_energy_err = histogram_distance(ref_energy, gen_energy, 40, -2.0, 2.0)
    hist_mag_err = histogram_distance(ref_mag, gen_mag, 40, 0.0, 1.0)
    w_energy = wasserstein_1d(ref_energy, gen_energy)
    w_mag = wasserstein_1d(ref_mag, gen_mag)

    total_err = energy_err + mag_err + corr_err + chi_err + hist_energy_err + hist_mag_err
    wass_err = w_energy + w_mag + corr_err
    score = 1.0 / (1.0 + wass_err)

    metrics = {
        "energy_ref": ref_energy.mean().item(),
        "energy_gen": gen_energy.mean().item(),
        "energy_err": energy_err.item(),
        "w_energy": w_energy.item(),
        "mag_ref": ref_mag.mean().item(),
        "mag_gen": gen_mag.mean().item(),
        "mag_err": mag_err.item(),
        "w_mag": w_mag.item(),
        "chi_ref": ref_chi.item(),
        "chi_gen": gen_chi.item(),
        "corr_err": corr_err.item(),
        "chi_err": chi_err.item(),
        "hist_energy_err": hist_energy_err.item(),
        "hist_mag_err": hist_mag_err.item(),
        "wass_err": wass_err.item(),
        "legacy_total_err": total_err.item(),
        "total_err": total_err.item(),
        "sampler_score": score.item(),
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        n=args.n,
        num_train=args.num_train,
        num_val=args.num_val,
        burn_in=args.burn_in,
        thin=args.thin,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
    )

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
    print(f"device={device} Tc={critical_temperature():.6f}")

    start = time.time()
    model, history = train_model(cfg, device)

    # prepare results directory (inside ising/ising_results)
    results_dir = os.path.join("ising", "ising_results", f"transformer_n{cfg.n}_{int(time.time())}")
    os.makedirs(results_dir, exist_ok=True)

    # save training history to CSV
    hist_csv = os.path.join(results_dir, "train_history.csv")
    with open(hist_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(history["train_loss"], history["val_loss"])):
            writer.writerow([i + 1, t, v])

    # plot loss curves
    try:
        plt.figure()
        cmap = plt.get_cmap("RdYlBu")
        ctrain = cmap(0.15)
        cval = cmap(0.85)
        xs = np.arange(1, len(history["train_loss"]) + 1)
        ys_train = np.array(history["train_loss"]) if len(history["train_loss"]) > 0 else np.array([])
        ys_val = np.array(history["val_loss"]) if len(history["val_loss"]) > 0 else np.array([])
        if ys_train.size > 0:
            plt.plot(xs, ys_train, label="train", color=ctrain, marker="o")
        if ys_val.size > 0:
            plt.plot(xs, ys_val, label="val", color=cval, marker="o")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        # set sensible y-limits with margin
        all_y = np.concatenate([ys_train, ys_val]) if (ys_train.size + ys_val.size) > 0 else np.array([0.0, 1.0])
        ymin, ymax = all_y.min(), all_y.max()
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5
        margin = (ymax - ymin) * 0.1
        plt.ylim(ymin - margin, ymax + margin)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "loss_curve.png"))
        plt.close()
    except Exception:
        pass

    # evaluate and save metrics + sample grids
    metrics = evaluate_sampler(model, cfg, device)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # save a batch of generated samples
    samples = sample_model(model, min(200, cfg.num_val), cfg.n, device)
    torch.save(samples.cpu(), os.path.join(results_dir, "samples.pt"))

    # save sample grid image
    try:
        grid = samples[:36].cpu().numpy()
        fig, axes = plt.subplots(6, 6, figsize=(6, 6), dpi=100)
        for i, ax in enumerate(axes.flat):
            ax.imshow(grid[i], cmap="RdYlBu", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "samples_grid.png"))
        plt.close()
    except Exception:
        pass

    elapsed = time.time() - start
    print(f"elapsed_sec={elapsed:.1f}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
