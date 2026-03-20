#!/usr/bin/env python3
"""
EquiLearn Final — Research-grade implementation

Features:
 - Real-data ready (loads CSVs from directory)
 - DKT + EPV embeddings (trainable) + EPV integration into prediction
 - Curriculum Negotiation Protocol (CNP) feedback support
 - MORL: mastery + differentiable fairness (variance across EPV clusters)
 - Early stopping, LR scheduler, saving models, logging, plots
Usage example:
 python equilearn_final.py --data_dir ./EquiLearn_Data --ednet ednet_sample.csv --khan khan_metadata.csv --outdir ./EquiLearn_out --seeds 0,1,2 --epochs 5
"""

import os
import argparse
import time
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gini_coefficient(x):
    x = np.array(x, dtype=float).flatten()
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        x = x - np.min(x)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n+1)
    return (np.sum((2*index - n - 1) * x_sorted)) / (n * np.sum(x_sorted))

def cohens_d_paired(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    diffs = x - y
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float('nan')
    return np.mean(diffs) / sd

# ----------------------------
# Data loading
# ----------------------------
def load_ednet(path):
    df = pd.read_csv(path)
    expected = {"user_id","question_id","correct","timestamp"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"ednet CSV missing required columns: {expected}. Found: {set(df.columns)}")
    df = df[["user_id","question_id","correct","timestamp"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["question_id"] = df["question_id"].astype(str)
    df["correct"] = df["correct"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)
    return df

def load_khan_meta(path):
    df = pd.read_csv(path)
    expected = {"question_id","cognitive_style","cultural_framing","expressive_modality","language"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"khan metadata CSV missing required columns: {expected}. Found: {set(df.columns)}")
    df = df[["question_id","cognitive_style","cultural_framing","expressive_modality","language"]].copy()
    df["question_id"] = df["question_id"].astype(str)
    df["cognitive_style"] = df["cognitive_style"].astype(float)
    df["cultural_framing"] = df["cultural_framing"].astype(float)
    df["expressive_modality"] = df["expressive_modality"].astype(float)
    return df

# ----------------------------
# EPV extractor and CNP
# ----------------------------
class EpistemicPositionalityVector:
    def __init__(self, khan_meta_df, q_vocab):
        # initialize EPV per question index (vocab indices)
        self.q_vocab = q_vocab
        self.epv_init = {}
        for _, r in khan_meta_df.iterrows():
            qid = str(r["question_id"])
            if qid in q_vocab:
                idx = q_vocab[qid]
                self.epv_init[idx] = np.array([float(r["cognitive_style"]), float(r["cultural_framing"]), float(r["expressive_modality"])], dtype=np.float32)
        # fill missing with zeros
        self.default = np.array([0.0,0.0,0.0], dtype=np.float32)
    def get_by_idx(self, idx):
        return self.epv_init.get(int(idx), self.default)

class CurriculumNegotiationProtocol:
    def __init__(self):
        # user feedback stored as list of question indices flagged as not_relatable
        self.user_feedback = defaultdict(list)
    def register_feedback(self, user_id, qid_idx):
        self.user_feedback[int(user_id)].append(int(qid_idx))
    def has_feedback(self, user_id):
        return int(user_id) in self.user_feedback and len(self.user_feedback[int(user_id)])>0

# ----------------------------
# Model: DKT with EPV integration
# ----------------------------
class EquiLearnModel(nn.Module):
    def __init__(self, num_questions, q_embed=64, epv_dim=3, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.q_embed = nn.Embedding(num_questions+1, q_embed, padding_idx=0)  # indices start at 1
        self.r_embed = nn.Embedding(2, q_embed)
        self.input_fc = nn.Linear(q_embed*2, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.q_proj = nn.Linear(q_embed, hidden_dim)
        self.epv_proj = nn.Linear(epv_dim, hidden_dim)
        self.out_bias = nn.Parameter(torch.zeros(1))
    def forward(self, q_seq, r_seq):
        # q_seq, r_seq: (B,T)
        q_e = self.q_embed(q_seq)   # (B,T,qe)
        r_e = self.r_embed(r_seq)   # (B,T,qe)
        x = torch.cat([q_e, r_e], dim=-1)
        x = torch.relu(self.input_fc(x))
        x = self.layernorm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        return out
    def predict(self, hidden, next_q_idx, next_epv):
        # hidden: (B,T,H), next_q_idx: (B,T), next_epv: (B,T,epv_dim)
        q_next_e = self.q_embed(next_q_idx)
        qproj = self.q_proj(q_next_e)
        epvproj = self.epv_proj(next_epv)
        composed = qproj + epvproj
        scores = (hidden * composed).sum(dim=-1) + self.out_bias
        return torch.sigmoid(scores)

# ----------------------------
# Data and loader utilities
# ----------------------------
def build_question_vocab(ednet_df, max_questions=None):
    qs = sorted(ednet_df["question_id"].unique())
    if max_questions:
        qs = qs[:max_questions]
    # indexing from 1, reserve 0 for padding
    q_to_idx = {qid: idx for idx, qid in enumerate(qs, start=1)}
    return q_to_idx

def create_dkt_loader(ednet_df, q_vocab, max_seq_len=100, batch_size=64, min_seq=2):
    sequences = []
    grouped = ednet_df.groupby("user_id")
    for user_id, g in grouped:
        g_sorted = g.sort_values("timestamp")
        qids = g_sorted["question_id"].map(lambda x: q_vocab.get(x, None)).dropna().astype(int).values
        corrects = g_sorted["correct"].values
        if len(qids) < min_seq:
            continue
        for i in range(0, len(qids)-1, max_seq_len):
            q_chunk = qids[i:i+max_seq_len+1]
            r_chunk = corrects[i:i+max_seq_len+1]
            if len(q_chunk) < 2:
                continue
            pad = max_seq_len + 1 - len(q_chunk)
            q_pad = np.pad(q_chunk, (0,pad), constant_values=0)
            r_pad = np.pad(r_chunk, (0,pad), constant_values=0)
            sequences.append((int(user_id), q_pad[:-1], r_pad[:-1], q_pad[1:], r_pad[1:]))
    if len(sequences) == 0:
        raise RuntimeError("No sequences generated - check dataset, filters, or increase max_seq_len.")
    users = torch.tensor([s[0] for s in sequences], dtype=torch.long)
    q_in = torch.tensor(np.stack([s[1] for s in sequences]), dtype=torch.long)
    r_in = torch.tensor(np.stack([s[2] for s in sequences]), dtype=torch.long)
    q_next = torch.tensor(np.stack([s[3] for s in sequences]), dtype=torch.long)
    targets = torch.tensor(np.stack([s[4] for s in sequences]), dtype=torch.float32)
    dataset = TensorDataset(users, q_in, r_in, q_next, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(dataset)

# ----------------------------
# Fairness loss: differentiable variance across EPV-cluster means (torch)
# ----------------------------
def build_cluster_keys(khan_meta_df, q_vocab):
    keys = {}
    idx = 0
    for _, r in khan_meta_df.iterrows():
        qid = str(r["question_id"])
        if qid in q_vocab:
            key = tuple(np.round([r["cognitive_style"], r["cultural_framing"], r["expressive_modality"]], 1).astype(int).tolist())
            if key not in keys:
                keys[key] = idx
                idx += 1
    if not keys:
        keys[(0,0,0)] = 0
    return keys

def fairness_loss_tensor(residuals, epv_array, cluster_keys, device):
    # residuals: torch tensor (N,)
    # epv_array: numpy array (N,3)
    residuals = residuals.to(device)
    epv_t = torch.tensor(epv_array, dtype=torch.float32, device=device)
    epv_round = torch.round(epv_t * 10).int()
    cluster_idx = []
    for r in epv_round.cpu().numpy():
        key = tuple(r.tolist())
        cluster_idx.append(cluster_keys.get(key, 0))
    cluster_idx = torch.tensor(cluster_idx, dtype=torch.long, device=device)
    num_clusters = max(cluster_keys.values()) + 1
    sums = torch.zeros(num_clusters, device=device)
    counts = torch.zeros(num_clusters, device=device)
    sums = sums.scatter_add(0, cluster_idx, residuals)
    ones = torch.ones_like(residuals, device=device)
    counts = counts.scatter_add(0, cluster_idx, ones)
    counts = torch.clamp(counts, min=1.0)
    means = sums / counts
    return torch.var(means)

# ----------------------------
# Evaluation: AUC + fairness (1 - Gini over cluster means)
# ----------------------------
def evaluate_model(model, loader, epv_extractor, q_vocab_rev, device):
    model.eval()
    all_true = []
    all_pred = []
    residuals_by_cluster = defaultdict(list)
    with torch.no_grad():
        for users, q_seq, r_seq, q_next, targets in loader:
            q_seq = q_seq.to(device); r_seq = r_seq.to(device); q_next = q_next.to(device)
            hidden = model(q_seq, r_seq)
            # prepare epv array for next positions
            flat_next = q_next.cpu().numpy().flatten()
            epv_list = []
            for qidx in flat_next:
                qid = q_vocab_rev.get(int(qidx), None)
                epv_list.append(epv_extractor.get_by_idx(int(qidx)) if qid is not None else np.array([0.0,0.0,0.0], dtype=np.float32))
            epv_arr = np.stack(epv_list).reshape(q_next.shape[0], q_next.shape[1], 3)
            epv_tensor = torch.tensor(epv_arr, dtype=torch.float32, device=device)
            probs = model.predict(hidden, q_next, epv_tensor)
            preds = probs.cpu().numpy().flatten()
            trues = targets.cpu().numpy().flatten()
            all_pred.extend(preds.tolist()); all_true.extend(trues.tolist())
            residuals = np.abs(trues - preds)
            # aggregate cluster-level residuals
            for qidx, res in zip(flat_next, residuals):
                qid = q_vocab_rev.get(int(qidx), None)
                epv = epv_extractor.get_by_idx(int(qidx)) if qid is not None else np.array([0.0,0.0,0.0], dtype=np.float32)
                cluster_key = tuple(np.round(epv,1))
                residuals_by_cluster[cluster_key].append(float(res))
    try:
        auc = roc_auc_score(all_true, all_pred)
    except Exception:
        auc = float('nan')
    cluster_means = np.array([np.mean(v) for v in residuals_by_cluster.values()]) if residuals_by_cluster else np.array([0.0])
    gini_val = gini_coefficient(cluster_means)
    fairness = 1.0 - gini_val
    return auc, fairness

# ----------------------------
# Training orchestration: baseline & EquiLearn across seeds
# ----------------------------
def run_training(ednet_df, khan_meta_df, outdir, seeds, epochs=5, batch_size=128, max_q=None, max_seq_len=80, lambda_fair=1.0, early_stop_patience=3, device=None):
    ensure_dir(outdir)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    q_vocab = build_question_vocab(ednet_df, max_questions=max_q)
    q_vocab_rev = {v:k for k,v in q_vocab.items()}
    epv_extractor = EpistemicPositionalityVector(khan_meta_df, q_vocab)
    cluster_keys = build_cluster_keys(khan_meta_df, q_vocab)
    results = []
    for seed in seeds:
        seed_everything(seed)
        print(f"\n--- Seed {seed} on device {device} ---")
        loader, nseq = create_dkt_loader(ednet_df, q_vocab, max_seq_len=max_seq_len, batch_size=batch_size)
        # Baseline training (mastery only)
        model_base = EquiLearnModel(num_questions=len(q_vocab)+1, q_embed=64, epv_dim=3, hidden_dim=128).to(device)
        opt_base = optim.Adam(model_base.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.BCELoss(reduction='none')
        best_base_auc = -1.0; no_improve = 0
        for epoch in range(epochs):
            model_base.train()
            pbar = tqdm(loader, desc=f"seed{seed}-base-ep{epoch}", leave=False)
            for users, q_seq, r_seq, q_next, targets in pbar:
                q_seq = q_seq.to(device); r_seq = r_seq.to(device); q_next = q_next.to(device); targets = targets.to(device)
                hidden = model_base(q_seq, r_seq)
                # prepare epv for next
                flat_next = q_next.cpu().numpy().flatten()
                epv_list = [epv_extractor.get_by_idx(int(qidx)) for qidx in flat_next]
                epv_arr = np.stack(epv_list).reshape(q_next.shape[0], q_next.shape[1], 3)
                epv_tensor = torch.tensor(epv_arr, dtype=torch.float32, device=device)
                probs = model_base.predict(hidden, q_next, epv_tensor)
                mask = (q_next != 0).float()
                loss_tensor = crit(probs, targets) * mask
                loss = loss_tensor.sum() / (mask.sum() + 1e-9)
                opt_base.zero_grad(); loss.backward(); opt_base.step()
            # evaluate
            auc_base, fair_base = evaluate_model(model_base, loader, epv_extractor, q_vocab_rev, device)
            print(f"Epoch {epoch} baseline eval -> AUC: {auc_base:.4f}, Fairness: {fair_base:.4f}")
            if auc_base > best_base_auc:
                best_base_auc = auc_base; no_improve = 0
                torch.save(model_base.state_dict(), os.path.join(outdir, f"model_base_seed{seed}.pt"))
            else:
                no_improve += 1
            if no_improve >= early_stop_patience:
                print("Early stopping baseline.")
                break
        # EquiLearn training (mastery + fairness)
        model_eq = EquiLearnModel(num_questions=len(q_vocab)+1, q_embed=64, epv_dim=3, hidden_dim=128).to(device)
        opt_eq = optim.Adam(model_eq.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_eq, mode='max', factor=0.7, patience=1, verbose=False)
        best_metric = -1.0; no_improve = 0
        history = {"epoch":[], "auc":[], "fair":[], "equity":[]}
        for epoch in range(epochs):
            model_eq.train()
            pbar = tqdm(loader, desc=f"seed{seed}-eq-ep{epoch}", leave=False)
            for users, q_seq, r_seq, q_next, targets in pbar:
                q_seq = q_seq.to(device); r_seq = r_seq.to(device); q_next = q_next.to(device); targets = targets.to(device)
                hidden = model_eq(q_seq, r_seq)
                # prepare flattened epv array for next positions (numpy)
                flat_next = q_next.cpu().numpy().flatten()
                epv_list = [epv_extractor.get_by_idx(int(qidx)) for qidx in flat_next]
                epv_arr = np.stack(epv_list).reshape(q_next.shape[0], q_next.shape[1], 3)
                epv_tensor = torch.tensor(epv_arr, dtype=torch.float32, device=device)
                probs = model_eq.predict(hidden, q_next, epv_tensor)
                mask = (q_next != 0).float()
                mastery_loss = (nn.BCELoss(reduction='none')(probs, targets) * mask).sum() / (mask.sum() + 1e-9)
                residuals = torch.abs(targets - probs).view(-1)
                epv_flat = epv_arr.reshape(-1,3)
                fair_loss = fairness_loss_tensor(residuals, epv_flat, cluster_keys, device)
                loss = mastery_loss + lambda_fair * fair_loss
                opt_eq.zero_grad(); loss.backward(); opt_eq.step()
            # evaluate after epoch
            auc_eq, fair_eq = evaluate_model(model_eq, loader, epv_extractor, q_vocab_rev, device)
            equity = 0.5 * (auc_eq if not math.isnan(auc_eq) else 0.0) + 0.5 * fair_eq
            history["epoch"].append(epoch); history["auc"].append(auc_eq); history["fair"].append(fair_eq); history["equity"].append(equity)
            print(f"Epoch {epoch} EquiLearn eval -> AUC: {auc_eq:.4f}, Fairness: {fair_eq:.4f}, Equity: {equity:.4f}")
            scheduler.step(equity)
            # early stopping on equity
            if equity > best_metric:
                best_metric = equity; no_improve = 0
                torch.save(model_eq.state_dict(), os.path.join(outdir, f"model_eq_seed{seed}.pt"))
            else:
                no_improve += 1
            if no_improve >= early_stop_patience:
                print("Early stopping EquiLearn.")
                break
        # final evaluation (load best model)
        model_base.load_state_dict(torch.load(os.path.join(outdir, f"model_base_seed{seed}.pt")))
        model_eq.load_state_dict(torch.load(os.path.join(outdir, f"model_eq_seed{seed}.pt")))
        auc_base, fair_base = evaluate_model(model_base, loader, epv_extractor, q_vocab_rev, device)
        auc_eq, fair_eq = evaluate_model(model_eq, loader, epv_extractor, q_vocab_rev, device)
        results_row = {"seed": seed, "auc_base": float(auc_base), "fair_base": float(fair_base), "auc_eq": float(auc_eq), "fair_eq": float(fair_eq)}
        results.append(results_row)
        # save epoch history plots for this seed
        plt.figure(figsize=(8,4))
        plt.plot(history["epoch"], history["auc"], label="AUC")
        plt.plot(history["epoch"], history["fair"], label="Fairness")
        plt.plot(history["epoch"], history["equity"], label="Equity")
        plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"seed{seed}_training_history.png"))
        plt.close()
    # aggregate results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outdir, "results_per_seed.csv"), index=False)
    # stats
    try:
        auc_stat = wilcoxon(results_df["auc_base"], results_df["auc_eq"])
        fair_stat = wilcoxon(results_df["fair_base"], results_df["fair_eq"])
        p_auc = auc_stat.pvalue; p_fair = fair_stat.pvalue
    except Exception:
        p_auc = float("nan"); p_fair = float("nan")
    d_auc = cohens_d_paired(results_df["auc_eq"], results_df["auc_base"])
    d_fair = cohens_d_paired(results_df["fair_eq"], results_df["fair_base"])
    summary = {"p_auc": p_auc, "p_fair": p_fair, "cohen_d_auc": d_auc, "cohen_d_fair": d_fair}
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(str(summary) + "\n")
    # scatter plot final
    plt.figure(figsize=(6,4))
    plt.scatter(results_df["auc_base"], results_df["fair_base"], label="Baseline", marker='x')
    plt.scatter(results_df["auc_eq"], results_df["fair_eq"], label="EquiLearn", marker='o')
    plt.xlabel("AUC"); plt.ylabel("Fairness (1 - Gini)"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_auc_fair_scatter.png"))
    plt.close()
    return results_df, summary

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./EquiLearn_Data")
    parser.add_argument("--ednet", type=str, default="ednet_sample.csv")
    parser.add_argument("--khan", type=str, default="khan_metadata.csv")
    parser.add_argument("--outdir", type=str, default="./EquiLearn_out")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_q", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=80)
    parser.add_argument("--lambda_fair", type=float, default=1.0)
    parser.add_argument("--early_stop", type=int, default=3)
    args = parser.parse_args()

    ednet_path = os.path.join(args.data_dir, args.ednet)
    khan_path = os.path.join(args.data_dir, args.khan)
    ensure_dir(args.outdir)
    if not os.path.exists(ednet_path):
        raise FileNotFoundError(f"EdNet file not found at {ednet_path}")
    if not os.path.exists(khan_path):
        raise FileNotFoundError(f"Khan metadata file not found at {khan_path}")

    print("Loading data...")
    ednet_df = load_ednet(ednet_path)
    khan_meta_df = load_khan_meta(khan_path)
    print(f"Loaded ednet rows: {len(ednet_df)}, khan meta rows: {len(khan_meta_df)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()!='']

    results_df, summary = run_training(ednet_df, khan_meta_df, args.outdir, seeds, epochs=args.epochs, batch_size=args.batch_size, max_q=args.max_q, max_seq_len=args.max_seq_len, lambda_fair=args.lambda_fair, early_stop_patience=args.early_stop)
    print("Done. Summary:", summary)
    print("Results per seed:\n", results_df)

if __name__ == "__main__":
    main()
