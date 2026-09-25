#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torchvision
import torchvision.transforms as transforms

EARLY_EXIT_THRESHOLD = 0.90  
CHOSEN_THETA = 0.20
DEFAULT_SEEDS_2TIER = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_SEEDS_3TIER = [42, 43, 44, 45, 46, 47, 48, 49]

DATASETS = {
    'CIFAR-100': {
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ]),
    },
    'CIFAR-10': {
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        ]),
    }
}

ROUTER_LABELS = {
    'xgb': 'XGBoost', 'rf': 'Random Forest', 'logreg': 'Logistic Regression',
    'dtree': 'Decision Tree', 'heuristic': 'GA Heuristic',
}

class HeuristicRouter:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        score = X_arr @ self.weights[:3] + self.weights[3]
        prob_escalate = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob_escalate, prob_escalate])

def load_router(router_dir, filename):
    path = os.path.join(router_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing router artifact: {}\n"
            "Did you run train_and_export_routers.py on Colab and copy its "
            "--router_dir output here?".format(path)
        )
    return joblib.load(path)

# --- Dataset and Model Helpers ---

def get_dataset(dataset_name, transform, data_root='./data'):
    if dataset_name == 'CIFAR-100':
        return torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR-10':
        return torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    raise ValueError("Unknown dataset: {}".format(dataset_name))

def load_models_for_dataset(dataset_name, device):
    if dataset_name == 'CIFAR-100':
        model_a = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True).to(device)
        model_b = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True).to(device)
        model_c = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True).to(device)
    elif dataset_name == 'CIFAR-10':
        model_a = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
        model_b = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(device)
        model_c = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).to(device)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
    model_a.eval(); model_b.eval(); model_c.eval()
    return model_a, model_b, model_c

# --- Latency Profiling ---

def measure_single_sample_latency(model, single_sample, n_reps=120, n_warmup=15, device=None):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(single_sample)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_reps):
            start = time.perf_counter()
            _ = model(single_sample)
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    times = np.array(times)
    return float(times.mean()), float(times.std())

# --- Feature Extraction & Overhead Profiling ---

def extract_telemetry(loader, model_a, model_c, device):
    rows = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs_a = torch.softmax(model_a(images), dim=1)
            conf_A, preds_a = torch.max(probs_a, dim=1)
            entropy_A = -torch.sum(probs_a * torch.log(probs_a + 1e-6), dim=1)
            top2, _ = torch.topk(probs_a, k=2, dim=1)
            margin_A = top2[:, 0] - top2[:, 1]
            _, preds_c = torch.max(model_c(images), dim=1)
            actual_route_to_c = ((preds_a != labels) & (preds_c == labels)).long()
            for i in range(images.size(0)):
                rows.append({
                    'confidence': conf_A[i].item(), 'entropy': entropy_A[i].item(), 'margin': margin_A[i].item(),
                    'correct_a': (preds_a[i] == labels[i]).item(), 'correct_c': (preds_c[i] == labels[i]).item(),
                    'target_route': actual_route_to_c[i].item(),
                })
    return pd.DataFrame(rows)

def extract_3stage_telemetry(loader, model_a, model_b, model_c, device):
    rows = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs_a = torch.softmax(model_a(images), dim=1)
            conf_a, preds_a = torch.max(probs_a, dim=1)
            entropy_a = -torch.sum(probs_a * torch.log(probs_a + 1e-6), dim=1)
            top2_a, _ = torch.topk(probs_a, k=2, dim=1)
            margin_a = top2_a[:, 0] - top2_a[:, 1]

            probs_b = torch.softmax(model_b(images), dim=1)
            conf_b, preds_b = torch.max(probs_b, dim=1)
            entropy_b = -torch.sum(probs_b * torch.log(probs_b + 1e-6), dim=1)
            top2_b, _ = torch.topk(probs_b, k=2, dim=1)
            margin_b = top2_b[:, 0] - top2_b[:, 1]

            _, preds_c = torch.max(model_c(images), dim=1)
            target_a_to_b = ((preds_a != labels) & (preds_b == labels)).long()
            target_b_to_c = ((preds_b != labels) & (preds_c == labels)).long()

            for i in range(images.size(0)):
                rows.append({
                    'conf_a': conf_a[i].item(), 'entropy_a': entropy_a[i].item(), 'margin_a': margin_a[i].item(),
                    'conf_b': conf_b[i].item(), 'entropy_b': entropy_b[i].item(), 'margin_b': margin_b[i].item(),
                    'correct_a': (preds_a[i] == labels[i]).item(), 'correct_b': (preds_b[i] == labels[i]).item(),
                    'correct_c': (preds_c[i] == labels[i]).item(),
                    'target_ab': target_a_to_b[i].item(), 'target_bc': target_b_to_c[i].item()
                })
    return pd.DataFrame(rows)

def measure_router_overhead_per_sample(router, X_test, n_reps=3):
    X_arr = X_test.values if hasattr(X_test, 'values') else X_test
    times = []
    for _ in range(n_reps):
        for row in X_arr:
            row = row.reshape(1, -1)
            start = time.perf_counter()
            _ = router.predict_proba(row)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    times = np.array(times)
    return float(times.mean()), float(times.std())

def evaluate_router_cascade(router, X_test, raw_test, LATENCY_A, LATENCY_C, theta=0.20, precomputed_overhead=None):
    n = len(raw_test)
    router_probs = router.predict_proba(X_test)[:, 1]
    decisions = np.where(router_probs >= theta, 1, 0)
    if precomputed_overhead is not None:
        router_overhead_mean = precomputed_overhead
    else:
        router_overhead_mean, _ = measure_router_overhead_per_sample(router, X_test.head(200))

    latency_total = 0.0
    correct = 0
    for i in range(n):
        latency_total += LATENCY_A + router_overhead_mean
        if raw_test.loc[i, 'confidence'] >= EARLY_EXIT_THRESHOLD:
            correct += raw_test.loc[i, 'correct_a']
        elif decisions[i] == 1:
            latency_total += LATENCY_C
            correct += raw_test.loc[i, 'correct_c']
        else:
            correct += raw_test.loc[i, 'correct_a']

    early_exit_count = (raw_test['confidence'] >= EARLY_EXIT_THRESHOLD).sum()
    return {'avg_latency': latency_total / n, 'accuracy': (correct / n) * 100,
            'router_overhead': router_overhead_mean, 'early_exit_rate': early_exit_count / n}

def naive_threshold_cascade(raw_df, tau, LATENCY_A, LATENCY_C):
    route = (raw_df['confidence'] < tau).astype(int)
    latency = LATENCY_A + route * LATENCY_C
    correct = np.where(route == 1, raw_df['correct_c'], raw_df['correct_a'])
    return {'avg_latency': float(latency.mean()), 'accuracy': float(correct.mean() * 100)}

def evaluate_3tier_threshold(df_telemetry, tau1=0.85, tau2=0.90, lat_a=0.0, lat_b=0.0, lat_c=0.0):
    total = len(df_telemetry)
    esc_to_b = (df_telemetry['conf_a'] < tau1).values
    exit_a = ~esc_to_b
    esc_to_c = esc_to_b & (df_telemetry['conf_b'] < tau2).values
    exit_b = esc_to_b & (~(df_telemetry['conf_b'] < tau2).values)

    correct = (
        (df_telemetry['correct_a'].values & exit_a).sum() +
        (df_telemetry['correct_b'].values & exit_b).sum() +
        (df_telemetry['correct_c'].values & esc_to_c).sum()
    )
    p_b = float(np.mean(esc_to_b))
    p_c = float(np.mean(esc_to_c))
    expected_latency = lat_a + (p_b * lat_b) + (p_c * lat_c)

    return {
        'Method': 'No Router (tau1={}, tau2={})'.format(tau1, tau2),
        'Accuracy': (correct / total) * 100,
        'Latency (ms)': expected_latency,
    }

def paired_report(a, b, label_a, label_b, unit='ms'):
    diffs = np.array(a) - np.array(b)
    t_stat, p_val = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = np.nan
    print("  Mean Diff ({} - {}): {:+.3f} {}, Std: {:.3f} {}".format(label_a, label_b, diffs.mean(), unit, diffs.std(), unit))
    print("  Paired t-test        : t = {:.3f}, p = {:.4f}".format(t_stat, p_val))
    print("  Wilcoxon signed-rank : p = {:.4f}\n".format(w_p))
    # Returned (in addition to printing) so callers can collect these into a CSV.
    return {
        'label_a': label_a, 'label_b': label_b, 'unit': unit,
        'mean_diff': float(diffs.mean()), 'std_diff': float(diffs.std()),
        't_stat': float(t_stat), 'p_ttest': float(p_val), 'p_wilcoxon': float(w_p),
    }

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Run IDK Cascade Study on Edge Devices (load pre-trained routers).")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--router_dir', type=str, required=True,
                         help='Directory containing router artifacts exported by train_and_export_routers.py')
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--skip_3tier', action='store_true')
    parser.add_argument('--early_exit_threshold', type=float, default=0.90,
                         help='SENSITIVITY-ANALYSIS FLAG ONLY. Do not change this for your primary '
                              'results without discussing with your advisor first -- 0.90 is the '
                              'value used consistently throughout the rest of the paper and Colab '
                              'results. Changing it is a methodology decision, not a bug fix.')
    parser.add_argument('--batch_size', type=int, default=128,
                         help='DataLoader batch size. Higher = fewer, larger forward-pass calls; '
                              'does not change any computed value, only throughput.')
    parser.add_argument('--n_samples_3tier', type=int, default=4000,
                         help='Subsample size for 3-tier telemetry extraction (was previously the '
                              'full ~10,000-image test set, unsubsampled). Set to 0 to use the full set.')
    args = parser.parse_args()

    global EARLY_EXIT_THRESHOLD
    EARLY_EXIT_THRESHOLD = args.early_exit_threshold
    if EARLY_EXIT_THRESHOLD != 0.90:
        print("!" * 70)
        print("WARNING: EARLY_EXIT_THRESHOLD overridden to {} (default is 0.90).".format(EARLY_EXIT_THRESHOLD))
        print("This is a methodology change relative to the rest of the paper's results.")
        print("Confirm this is intentional and disclosed before using these numbers.")
        print("!" * 70)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using execution device: {}".format(device))
    print("Loading routers from: {}".format(args.router_dir))

    # Baseline Model Latency Profiling
    cifar100_transform = DATASETS['CIFAR-100']['transform']
    val_cifar100 = get_dataset('CIFAR-100', cifar100_transform, data_root=args.data_dir)
    m_a, m_b, m_c = load_models_for_dataset('CIFAR-100', device)

    probe_batch = val_cifar100[0][0].unsqueeze(0).to(device)
    lat_a, lat_a_std = measure_single_sample_latency(m_a, probe_batch, device=device)
    lat_b, lat_b_std = measure_single_sample_latency(m_b, probe_batch, device=device)
    lat_c, lat_c_std = measure_single_sample_latency(m_c, probe_batch, device=device)

    print("Model A Latency: {:.4f} +/- {:.4f} ms".format(lat_a, lat_a_std))
    print("Model B Latency: {:.4f} +/- {:.4f} ms".format(lat_b, lat_b_std))
    print("Model C Latency: {:.4f} +/- {:.4f} ms\n".format(lat_c, lat_c_std))

    pd.DataFrame([
        {'Model': 'A (ResNet20)', 'Mean_ms': lat_a, 'Std_ms': lat_a_std},
        {'Model': 'B (ResNet32)', 'Mean_ms': lat_b, 'Std_ms': lat_b_std},
        {'Model': 'C (ResNet56)', 'Mean_ms': lat_c, 'Std_ms': lat_c_std},
    ]).to_csv(os.path.join(args.output_dir, "model_latency.csv"), index=False)

   
    subset_probe = torch.utils.data.Subset(val_cifar100, list(range(1000)))
    loader_probe = torch.utils.data.DataLoader(subset_probe, batch_size=args.batch_size, shuffle=False)
    df_probe = extract_telemetry(loader_probe, m_a, m_c, device)
    X_te_probe = df_probe[['confidence', 'entropy', 'margin']].iloc[int(len(df_probe) * 0.8):]

    router_overheads = {}
    router_overhead_records = []
    print("=== Router Overhead (predict_proba isolated cost, measured on THIS device) ===")
    for name, label in ROUTER_LABELS.items():
        bench_router = load_router(args.router_dir, "bench_{}.joblib".format(name))
        mean_oh, std_oh = measure_router_overhead_per_sample(bench_router, X_te_probe.head(200))
        router_overheads[name] = mean_oh
        router_overhead_records.append({'Router': label, 'Mean_ms': mean_oh, 'Std_ms': std_oh})
        print("[METRIC] {:<20s} Overhead : {:.5f} +/- {:.5f} ms/sample".format(label, mean_oh, std_oh))

    pd.DataFrame(router_overhead_records).to_csv(os.path.join(args.output_dir, "router_overhead.csv"), index=False)

    # Multi-Seed 2-Tier Replication
    all_trial_results = []
    t_pipeline_start = time.perf_counter()

    for dname in ['CIFAR-100', 'CIFAR-10']:
        print("\n=== Running 2-tier multi-router trial for dataset: {} ===".format(dname))
        curr_transform = DATASETS[dname]['transform']
        curr_models = load_models_for_dataset(dname, device)
        curr_dataset = get_dataset(dname, curr_transform, data_root=args.data_dir)

        for s in DEFAULT_SEEDS_2TIER:
            torch.manual_seed(s)
            idx = torch.randperm(len(curr_dataset))[:args.n_samples]
            sub = torch.utils.data.Subset(curr_dataset, idx)
            loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False)

            df = extract_telemetry(loader, curr_models[0], curr_models[2], device)
            X = df[['confidence', 'entropy', 'margin']]
            y = df['target_route']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
            raw_test = df.loc[X_test.index].reset_index(drop=True)
            X_test_reset = X_test.reset_index(drop=True)

            naive_res = naive_threshold_cascade(raw_test, tau=0.80, LATENCY_A=lat_a, LATENCY_C=lat_c)
            row = {
                'dataset': dname, 'seed': s,
                'expert_accuracy': raw_test['correct_c'].mean() * 100, 'expert_latency': lat_c,
                'naive_accuracy': naive_res['accuracy'], 'naive_latency': naive_res['avg_latency'],
            }

            for name, label in ROUTER_LABELS.items():
                router = load_router(args.router_dir, "2tier_{}_{}_{}.joblib".format(dname, s, name))
                res = evaluate_router_cascade(router, X_test_reset, raw_test, lat_a, lat_c, theta=CHOSEN_THETA,
                                               precomputed_overhead=router_overheads[name])
                row[f'{name}_accuracy'] = res['accuracy']
                row[f'{name}_latency'] = res['avg_latency']
                row[f'{name}_overhead'] = res['router_overhead']
                row['early_exit_rate'] = res['early_exit_rate']  # same value regardless of which router; stored once per row

            all_trial_results.append(row)

    t_pipeline_end = time.perf_counter()
    trial_df = pd.DataFrame(all_trial_results)
    trial_df.to_csv(os.path.join(args.output_dir, "two_tier_trials.csv"), index=False)

    print("\n=== Summary Performance Across All Seeds & Datasets ===")
    two_tier_summary_records = []
    for name, label in ROUTER_LABELS.items():
        acc_mean, acc_std = trial_df[f'{name}_accuracy'].mean(), trial_df[f'{name}_accuracy'].std()
        lat_mean, lat_std = trial_df[f'{name}_latency'].mean(), trial_df[f'{name}_latency'].std()
        ovh_mean, ovh_std = trial_df[f'{name}_overhead'].mean(), trial_df[f'{name}_overhead'].std()
        print("{:<20s} Accuracy : {:.2f} +/- {:.2f} %".format(label, acc_mean, acc_std))
        two_tier_summary_records.append({
            'Method': label, 'Accuracy_mean_pct': acc_mean, 'Accuracy_std_pct': acc_std,
            'Latency_mean_ms': lat_mean, 'Latency_std_ms': lat_std,
            'RouterOverhead_mean_ms': ovh_mean, 'RouterOverhead_std_ms': ovh_std,
        })
    print("{:<20s} Accuracy : {:.2f} +/- {:.2f} %".format(
        'Pure Expert', trial_df['expert_accuracy'].mean(), trial_df['expert_accuracy'].std()))
    two_tier_summary_records.append({
        'Method': 'Pure Expert', 'Accuracy_mean_pct': trial_df['expert_accuracy'].mean(),
        'Accuracy_std_pct': trial_df['expert_accuracy'].std(),
        'Latency_mean_ms': trial_df['expert_latency'].mean(), 'Latency_std_ms': trial_df['expert_latency'].std(),
        'RouterOverhead_mean_ms': None, 'RouterOverhead_std_ms': None,
    })
    print("{:<20s} Accuracy : {:.2f} +/- {:.2f} %\n".format(
        'Naive Baseline', trial_df['naive_accuracy'].mean(), trial_df['naive_accuracy'].std()))

    print("Early-exit rate (fraction of samples where confidence >= {}): {:.2f} +/- {:.2f} %".format(
        EARLY_EXIT_THRESHOLD, trial_df['early_exit_rate'].mean() * 100, trial_df['early_exit_rate'].std() * 100))
    print("(If this is high, the router's decision affects few samples -- check this before")
    print(" interpreting router-vs-router accuracy comparisons.)\n")
    two_tier_summary_records.append({
        'Method': 'Naive Baseline', 'Accuracy_mean_pct': trial_df['naive_accuracy'].mean(),
        'Accuracy_std_pct': trial_df['naive_accuracy'].std(),
        'Latency_mean_ms': trial_df['naive_latency'].mean(), 'Latency_std_ms': trial_df['naive_latency'].std(),
        'RouterOverhead_mean_ms': None, 'RouterOverhead_std_ms': None,
    })

    for name, label in ROUTER_LABELS.items():
        print("{:<20s} Cascade Latency : {:.4f} +/- {:.4f} ms".format(
            label, trial_df[f'{name}_latency'].mean(), trial_df[f'{name}_latency'].std()))
    print("{:<20s} Cascade Latency : {:.4f} ms\n".format('Pure Expert', trial_df['expert_latency'].mean()))

    pd.DataFrame(two_tier_summary_records).to_csv(os.path.join(args.output_dir, "two_tier_summary.csv"), index=False)

    print("=== Paired Statistical Tests ===")
    two_tier_sig_records = []
    r_keys = list(ROUTER_LABELS.keys())
    for dname in trial_df['dataset'].unique():
        print("\n--- Paired Tests for {} ---".format(dname))
        ds_df = trial_df[trial_df['dataset'] == dname]
        for name_a, name_b in itertools.combinations(r_keys, 2):
            la, lb = ROUTER_LABELS[name_a], ROUTER_LABELS[name_b]
            print("{} vs. {} (Cascade Latency):".format(la, lb))
            stat_result = paired_report(ds_df[f'{name_a}_latency'], ds_df[f'{name_b}_latency'], f'{la} Latency', f'{lb} Latency', unit='ms')
            stat_result['dataset'] = dname
            stat_result['metric'] = 'latency'
            two_tier_sig_records.append(stat_result)

    pd.DataFrame(two_tier_sig_records).to_csv(os.path.join(args.output_dir, "two_tier_significance.csv"), index=False)

    print("Total pipeline wall-clock time: {:.2f} s ({:.2f} min)\n".format(
        t_pipeline_end - t_pipeline_start, (t_pipeline_end - t_pipeline_start) / 60))

    # Three-Tier Study Execution
    if not args.skip_3tier:
        from sklearn.model_selection import train_test_split
        print("=== Running Extension: Three-Tier Cascade vs. Two-Tier ===")
        all_3tier_records = []
        feats_a, feats_b = ['conf_a', 'entropy_a', 'margin_a'], ['conf_b', 'entropy_b', 'margin_b']
        ren_a = {'conf_a': 'confidence', 'entropy_a': 'entropy', 'margin_a': 'margin'}
        ren_b = {'conf_b': 'confidence', 'entropy_b': 'entropy', 'margin_b': 'margin'}

        for dname in ['CIFAR-10', 'CIFAR-100']:
            tform = DATASETS[dname]['transform']
            dset = get_dataset(dname, tform, data_root=args.data_dir)
            ma, mb, mc = load_models_for_dataset(dname, device)

            p_img = dset[0][0].unsqueeze(0).to(device)
            l_a, _ = measure_single_sample_latency(ma, p_img, device=device)
            l_b, _ = measure_single_sample_latency(mb, p_img, device=device)
            l_c, _ = measure_single_sample_latency(mc, p_img, device=device)

            if args.n_samples_3tier and args.n_samples_3tier > 0:
                torch.manual_seed(999)
                idx_3t = torch.randperm(len(dset))[:args.n_samples_3tier]
                dset_3t = torch.utils.data.Subset(dset, idx_3t)
            else:
                dset_3t = dset
            loader_full = torch.utils.data.DataLoader(dset_3t, batch_size=args.batch_size, shuffle=False)
            df_3st = extract_3stage_telemetry(loader_full, ma, mb, mc, device)

            for seed in DEFAULT_SEEDS_3TIER:
                df_tr, df_te = train_test_split(df_3st, test_size=0.2, random_state=seed)

                esc_ac = (df_te['conf_a'] < EARLY_EXIT_THRESHOLD).values
                acc_ac = np.mean(np.where(esc_ac, df_te['correct_c'], df_te['correct_a']))
                lat_ac = l_a + (np.mean(esc_ac) * l_c)
                all_3tier_records.append({'Dataset': dname, 'Seed': seed, 'Method': 'Pure Threshold (A->C)',
                                          'Accuracy': acc_ac * 100, 'Latency': lat_ac})

                res_3t = evaluate_3tier_threshold(df_te, 0.85, 0.90, lat_a=l_a, lat_b=l_b, lat_c=l_c)
                all_3tier_records.append({'Dataset': dname, 'Seed': seed, 'Method': 'Pure Threshold (A->B->C)',
                                          'Accuracy': res_3t['Accuracy'], 'Latency': res_3t['Latency (ms)']})

        
                all_3tier_records.append({'Dataset': dname, 'Seed': seed, 'Method': 'Pure Expert (Model C)',
                                          'Accuracy': df_te['correct_c'].mean() * 100, 'Latency': l_c})

                for r_key, label in ROUTER_LABELS.items():
                    ovh = router_overheads.get(r_key, 0.0)

                    # 2-Tier (this comparison's own A->C router, trained on 3-tier-seed splits)
                    r_2t = load_router(args.router_dir, "3tier_{}_{}_{}_ac.joblib".format(dname, seed, r_key))
                    p_2t = r_2t.predict_proba(df_te[feats_a].rename(columns=ren_a))[:, 1]
                    ee_mask = (df_te['conf_a'] >= EARLY_EXIT_THRESHOLD).values
                    esc_2t = (p_2t >= CHOSEN_THETA) & (~ee_mask)
                    acc_2t = np.mean(np.where(esc_2t, df_te['correct_c'], df_te['correct_a']))
                    lat_2t = l_a + ovh + (np.mean(esc_2t) * l_c)
                    all_3tier_records.append({'Dataset': dname, 'Seed': seed, 'Method': f"{label} (A->C)",
                                              'Accuracy': acc_2t * 100, 'Latency': lat_2t})

                    # 3-Tier
                    r_ab = load_router(args.router_dir, "3tier_{}_{}_{}_ab.joblib".format(dname, seed, r_key))
                    r_bc = load_router(args.router_dir, "3tier_{}_{}_{}_bc.joblib".format(dname, seed, r_key))

                    p_ab = r_ab.predict_proba(df_te[feats_a].rename(columns=ren_a))[:, 1]
                    p_bc = r_bc.predict_proba(df_te[feats_b].rename(columns=ren_b))[:, 1]

                    
                    esc_b = (p_ab >= CHOSEN_THETA)
                    esc_c = esc_b & (p_bc >= CHOSEN_THETA)
                    exit_b_mask = esc_b & (~(p_bc >= 0.5))
                    exit_a_mask = ~esc_b

                    acc_3t = np.mean(
                        (df_te['correct_a'].values & exit_a_mask) +
                        (df_te['correct_b'].values & exit_b_mask) +
                        (df_te['correct_c'].values & esc_c)
                    )
                    lat_3t = l_a + ovh + (np.mean(esc_b) * (l_b + ovh)) + (np.mean(esc_c) * l_c)
                    all_3tier_records.append({'Dataset': dname, 'Seed': seed, 'Method': f"{label} (A->B->C)",
                                              'Accuracy': acc_3t * 100, 'Latency': lat_3t})

        df_3t_res = pd.DataFrame(all_3tier_records)
        df_3t_res.to_csv(os.path.join(args.output_dir, "three_tier_trials.csv"), index=False)
        summary_3tier = df_3t_res.groupby('Method', sort=False).agg({'Accuracy': ['mean', 'std'], 'Latency': ['mean', 'std']})
        print("\n=== Two-Tier vs. Three-Tier Summary ===")
        print(summary_3tier.to_string())

        summary_3tier_flat = summary_3tier.copy()
        summary_3tier_flat.columns = ['_'.join(c) for c in summary_3tier_flat.columns]
        summary_3tier_flat.reset_index().to_csv(os.path.join(args.output_dir, "three_tier_summary.csv"), index=False)

        # --- Statistical Significance Testing for the 3-Tier Study ---
        def get_series(dataset, method, column):
            rows = df_3t_res[(df_3t_res['Dataset'] == dataset) & (df_3t_res['Method'] == method)].sort_values('Seed')
            return rows[column].values

        three_tier_sig_records = []

        print("\n=== Paired Statistical Tests: Three-Tier Study ===")
        for dname in df_3t_res['Dataset'].unique():
            print("\n--- {} ---".format(dname))

            print("\n[1] Router vs. Router (3-Tier, A->B->C only)")
            for name_a, name_b in itertools.combinations(ROUTER_LABELS.keys(), 2):
                la, lb = ROUTER_LABELS[name_a], ROUTER_LABELS[name_b]
                m_a, m_b = "{} (A->B->C)".format(la), "{} (A->B->C)".format(lb)
                print("{} vs. {} (3-Tier Latency):".format(la, lb))
                res = paired_report(get_series(dname, m_a, 'Latency'), get_series(dname, m_b, 'Latency'),
                               "{} 3T Latency".format(la), "{} 3T Latency".format(lb), unit='ms')
                res.update({'dataset': dname, 'category': 'router_vs_router', 'metric': 'latency'})
                three_tier_sig_records.append(res)

            print("\n[2] Within-Router: 2-Tier vs. 3-Tier (does adding Model B help this router?)")
            for name, label in ROUTER_LABELS.items():
                m_2t, m_3t = "{} (A->C)".format(label), "{} (A->B->C)".format(label)
                print("{}: 2-Tier vs. 3-Tier (Accuracy):".format(label))
                res = paired_report(get_series(dname, m_3t, 'Accuracy'), get_series(dname, m_2t, 'Accuracy'),
                               "{} 3T Acc".format(label), "{} 2T Acc".format(label), unit='pp')
                res.update({'dataset': dname, 'category': 'within_router_2t_vs_3t', 'metric': 'accuracy'})
                three_tier_sig_records.append(res)
                print("{}: 2-Tier vs. 3-Tier (Latency):".format(label))
                res = paired_report(get_series(dname, m_3t, 'Latency'), get_series(dname, m_2t, 'Latency'),
                               "{} 3T Lat".format(label), "{} 2T Lat".format(label), unit='ms')
                res.update({'dataset': dname, 'category': 'within_router_2t_vs_3t', 'metric': 'latency'})
                three_tier_sig_records.append(res)

            print("\n[3] Router (3-Tier) vs. Pure Threshold (A->B->C) Baseline")
            for name, label in ROUTER_LABELS.items():
                m_3t = "{} (A->B->C)".format(label)
                print("{} vs. Pure Threshold (3-Tier Latency):".format(label))
                res = paired_report(get_series(dname, m_3t, 'Latency'), get_series(dname, 'Pure Threshold (A->B->C)', 'Latency'),
                               "{} 3T Lat".format(label), "Pure Threshold 3T Lat", unit='ms')
                res.update({'dataset': dname, 'category': 'router_vs_pure_threshold', 'metric': 'latency'})
                three_tier_sig_records.append(res)

            print("\n[4] Router (3-Tier) vs. Pure Expert (no cascade)")
            for name, label in ROUTER_LABELS.items():
                m_3t = "{} (A->B->C)".format(label)
                print("{} vs. Pure Expert (3-Tier Latency):".format(label))
                res = paired_report(get_series(dname, m_3t, 'Latency'), get_series(dname, 'Pure Expert (Model C)', 'Latency'),
                               "{} 3T Lat".format(label), "Pure Expert Lat", unit='ms')
                res.update({'dataset': dname, 'category': 'router_vs_pure_expert', 'metric': 'latency'})
                three_tier_sig_records.append(res)

        pd.DataFrame(three_tier_sig_records).to_csv(os.path.join(args.output_dir, "three_tier_significance.csv"), index=False)

if __name__ == '__main__':
    main()
