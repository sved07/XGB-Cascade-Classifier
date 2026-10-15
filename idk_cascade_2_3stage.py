import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
import sklearn
import xgboost as xgb
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAU = 0.90
THETA = 0.20
DATA_ROOT = './data_imagenetv2'
DELETE_ARCHIVE_AFTER_EXTRACT = True
BATCH_SIZE = 16
NUM_WORKERS = 2
STAGES = ['A', 'B', 'C']
STAGE_MODELS = ['resnet18', 'resnet34', 'resnet50']
ARCHS = ['2-stage', '3-stage']
MAX_ROUTER_CALLS = {'2-stage': 1, '3-stage': 2}
FAMILIES = ['xgb', 'rf', 'logreg', 'dtree', 'heuristic']
FAMILY_LABELS = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'logreg': 'Logistic Regression',
                 'dtree': 'Decision Tree', 'heuristic': 'GA Heuristic'}
ROUTER_FAMILY_OF = {v: k for k, v in FAMILY_LABELS.items()}
METHOD_ORDER = ['Pure Expert', 'Pure Threshold'] + [FAMILY_LABELS[f] for f in FAMILIES]

# =====================================================================================================
# Dataset
# =====================================================================================================
HF = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/'       # primary mirror (the S3 bucket now returns 403)
S3 = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/'                      # legacy fallback
DATASET_SPECS = {
    'TopImages':        dict(archive='imagenetv2-top-images.tar.gz',        folder='imagenetv2-top-images-format-val'),
    'Threshold0.7':     dict(archive='imagenetv2-threshold0.7.tar.gz',      folder='imagenetv2-threshold0.7-format-val'),
    'MatchedFrequency': dict(archive='imagenetv2-matched-frequency.tar.gz', folder='imagenetv2-matched-frequency-format-val'),
}
for _spec in DATASET_SPECS.values():
    _spec['urls'] = [HF + _spec['archive'], S3 + _spec['archive']]

# Standard ImageNet evaluation preprocessing 
IMAGENET_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ImageNetV2Folder(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.samples = []
        class_dirs = sorted((d for d in os.listdir(root) if d.isdigit()), key=int)
        for d in class_dirs:
            cdir = os.path.join(root, d)
            for f in sorted(os.listdir(cdir)):
                p = os.path.join(cdir, f)
                if not f.startswith('.') and os.path.isfile(p):
                    self.samples.append((p, int(d)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            img = im.convert('RGB')
        return self.transform(img), label

def dataset_fingerprint(ds):
    
    h = hashlib.sha1()
    for p, lbl in ds.samples:
        h.update('{}:{}\n'.format(os.path.relpath(p, ds.root).replace(os.sep, '/'), lbl).encode('utf-8'))
    return {'n_images': len(ds), 'sha1': h.hexdigest()}

def download_file(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (idk-cascade-notebook)'})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, 'wb') as out:
        total = int(resp.headers.get('Content-Length') or 0)
        done, last = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and (done * 100 // total) // 5 != last:
                last = (done * 100 // total) // 5
                print('  {:3d}%  ({:.0f} MB)'.format(done * 100 // total, done / 1e6), flush=True)
    if total and done != total:
        raise IOError('Incomplete download ({} of {} bytes)'.format(done, total))

def find_dataset_root(base):
    found = []
    for dirpath, dirnames, _ in os.walk(base):
        digit_dirs = [d for d in dirnames if d.isdigit()]       
        if '0' in digit_dirs and len(digit_dirs) >= 10:
            found.append(dirpath)
    if len(found) != 1:
        raise RuntimeError('Could not identify a unique class-folder root in the archive ({} candidates).'.format(len(found)))
    return found[0]

def prepare_dataset(name):
    spec = DATASET_SPECS[name]
    os.makedirs(DATA_ROOT, exist_ok=True)
    folder = os.path.join(DATA_ROOT, spec['folder'])
    if not os.path.isdir(folder):
        archive = os.path.join(DATA_ROOT, spec['archive'])
        if not os.path.isfile(archive):
            errors = []
            for url in spec['urls']:
                print('Downloading {} (~1.2 GB) from {}'.format(name, url), flush=True)
                try:
                    download_file(url, archive + '.part')
                    os.replace(archive + '.part', archive)     
                    break
                except Exception as e:                          
                    errors.append('{}: {}'.format(url, e))
                    print('  failed ({}) - trying next mirror'.format(e), flush=True)
                    if os.path.exists(archive + '.part'):
                        os.remove(archive + '.part')
            else:
                raise RuntimeError('All download mirrors failed:\n  ' + '\n  '.join(errors) +
                                   '\nDownload {} manually (e.g. from https://huggingface.co/datasets/vaishaal/ImageNetV2) '
                                   'and put it in {} - or extract it there so that {} exists.'.format(
                                       spec['archive'], os.path.abspath(DATA_ROOT), folder))
        print('Extracting {} ...'.format(spec['archive']), flush=True)
        tmp = os.path.join(DATA_ROOT, '_extract_tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        with tarfile.open(archive, 'r:*') as tf:                  
            try:
                tf.extractall(tmp, filter='data')
            except TypeError:                                   
                tf.extractall(tmp)
        os.replace(find_dataset_root(tmp), folder)             
        shutil.rmtree(tmp, ignore_errors=True)
        if DELETE_ARCHIVE_AFTER_EXTRACT:
            os.remove(archive)
    ds = ImageNetV2Folder(folder, IMAGENET_TF)
    n_cls = len(set(lbl for _, lbl in ds.samples))
    print('{}: {} images, {} classes'.format(name, len(ds), n_cls))
    if len(ds) != 10000 or n_cls != 1000:
        print('  WARNING: expected 10,000 images in 1,000 classes - check the extracted folder.')
    return ds


# =====================================================================================================
# Models, stage functions, timing, telemetry pass
# =====================================================================================================
WEIGHT_ENUMS = {'resnet18': 'ResNet18_Weights', 'resnet34': 'ResNet34_Weights', 'resnet50': 'ResNet50_Weights'}

def load_imagenet_resnet(arch):
    ctor = getattr(torchvision.models, arch)
    enum = getattr(torchvision.models, WEIGHT_ENUMS[arch], None)
    model = ctor(weights=enum.IMAGENET1K_V1) if enum is not None else ctor(pretrained=True)
    return model.to(device).eval()

@torch.no_grad()
def stage_with_features(model, x):
    probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(dim=1)
    ent = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return torch.stack([conf, ent, margin, pred.to(probs.dtype)], dim=1).cpu().numpy()

@torch.no_grad()
def stage_final(model, x):
    
    return model(x).argmax(dim=1).cpu().numpy()

def to_device_input(ds, i):
    return ds[i][0].unsqueeze(0).to(device)

def time_stage(fn, model, x, n_reps, n_warmup):
    use_cuda = device.type == 'cuda'
    for _ in range(n_warmup):
        fn(model, x)
    if use_cuda:
        torch.cuda.synchronize()
    ts = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter()
        fn(model, x)
        if use_cuda:
            torch.cuda.synchronize()
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return ts

def timing_stats(ts):
    ts = np.asarray(ts, dtype=float)
    return dict(mean=float(ts.mean()), std=float(ts.std(ddof=1)) if len(ts) > 1 else 0.0,
                p99=float(np.percentile(ts, 99)), max=float(ts.max()))

def profile_stages(models, probe_x, n_reps, n_warmup):
    fns = [stage_with_features, stage_with_features, stage_final]          # C is the last stage: argmax only
    return [timing_stats(time_stage(fns[k], models[k], probe_x, n_reps, n_warmup)) for k in range(3)]

def profile_dataset(name, ds, models):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=(device.type == 'cuda'))
    out = [[] for _ in range(3)]
    labels, n_done, t0 = [], 0, time.perf_counter()
    for bi, (imgs, labs) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        for k in range(3):
            out[k].append(stage_with_features(models[k], imgs))
        labels.append(labs.numpy())
        n_done += len(labs)
        if (bi + 1) % 20 == 0 or n_done == len(ds):
            el = time.perf_counter() - t0
            print('  {}: {}/{} images | {:.0f}s elapsed | ETA {:.0f}s'.format(
                name, n_done, len(ds), el, el / n_done * (len(ds) - n_done)), flush=True)
    raw = np.stack([np.concatenate(o) for o in out]).astype(np.float64)       # (3, N, 4)
    labels = np.concatenate(labels)
    pred = raw[:, :, 3].astype(np.int64)
    feat = np.ascontiguousarray(raw[:, :, :3])                                # (3, N, 3): conf, entropy, margin
    return dict(feat=feat, pred=pred, correct=(pred == labels[None, :]), labels=labels)


# =====================================================================================================
# Routers (five families, identical hyperparameters to the paper)
# =====================================================================================================
class HeuristicRouter:
    """Linear score + sigmoid, evaluated directly with array operations (no library dispatch)."""
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        score = X_arr @ self.weights[:3] + self.weights[3]
        prob = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob, prob])

class ConstantRouter:
    def __init__(self, p):
        self.p = float(p)
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])

def cost_sensitive_weight(y, rel_cost):
    # w = clip((N_neg / N_pos) / rho, 1, 10); rho = latency of the gated stage / latency of the cheapest gated stage
    return float(np.clip(np.sum(y == 0) / max(np.sum(y == 1), 1) / max(rel_cost, 1e-6), 1.0, 10.0))

def train_xgb(X, y, rel_cost, seed):
    w = cost_sensitive_weight(y, rel_cost)
    r = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=w,
                          eval_metric='logloss', random_state=seed, n_jobs=2)
    r.fit(X, y)
    return r

def train_rf(X, y, rel_cost, seed):
    w = cost_sensitive_weight(y, rel_cost)
    r = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight={0: 1.0, 1: w}, random_state=seed, n_jobs=2)
    r.fit(X, y)
    return r

def train_logreg(X, y, rel_cost, seed):
    w = cost_sensitive_weight(y, rel_cost)
    r = LogisticRegression(class_weight={0: 1.0, 1: w}, max_iter=1000, random_state=seed)
    r.fit(X, y)
    return r

def train_dtree(X, y, rel_cost, seed):
    w = cost_sensitive_weight(y, rel_cost)
    r = DecisionTreeClassifier(max_depth=3, class_weight={0: 1.0, 1: w}, random_state=seed)
    r.fit(X, y)
    return r

def _fitness(weights, X, y, sw, theta):
    score = X @ weights[:3] + weights[3]
    preds = (score >= np.log(theta / (1 - theta))).astype(int)
    return np.sum((preds == y) * sw) / np.sum(sw)

def _ga(X, y, sw, seed, theta, pop_size=40, generations=60, gene_range=8.0, mutation_rate=0.2, mutation_scale=0.6):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-gene_range, gene_range, size=(pop_size, 4))
    best, best_f = None, -np.inf
    for _ in range(generations):
        fit = np.array([_fitness(ind, X, y, sw, theta) for ind in pop])
        g = int(np.argmax(fit))
        if fit[g] > best_f:
            best_f, best = fit[g], pop[g].copy()
        new = [best.copy()]                                               # elitism
        while len(new) < pop_size:
            i, j = rng.integers(0, pop_size, size=2)                      # tournament (size 2)
            p1 = pop[i] if fit[i] > fit[j] else pop[j]
            i, j = rng.integers(0, pop_size, size=2)
            p2 = pop[i] if fit[i] > fit[j] else pop[j]
            child = np.where(rng.random(4) < 0.5, p1, p2)                 # uniform crossover
            child = child + (rng.random(4) < mutation_rate) * rng.normal(0, mutation_scale, size=4)   # Gaussian mutation
            new.append(np.clip(child, -gene_range, gene_range))
        pop = np.array(new[:pop_size])
    return best, best_f

def train_heuristic(X, y, rel_cost, seed, n_restarts=25):
    w = cost_sensitive_weight(y, rel_cost)
    X = np.asarray(X, dtype=float)
    sw = np.where(y == 1, w, 1.0)
    best_w, best_f = None, -np.inf
    for restart in range(n_restarts):
        wts, f = _ga(X, y, sw, seed * 1000 + restart, THETA)
        if f > best_f:
            best_f, best_w = f, wts
    return HeuristicRouter(best_w)

TRAIN_FNS = {'xgb': train_xgb, 'rf': train_rf, 'logreg': train_logreg, 'dtree': train_dtree, 'heuristic': train_heuristic}

def timed_train(fam, X, y, rel_cost, seed, train_time):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        print('  WARNING: single-class labels for {} - constant router used.'.format(FAMILY_LABELS[fam]))
        return ConstantRouter(float(y[0]) if len(y) else 1.0)
    t0 = time.perf_counter()
    r = TRAIN_FNS[fam](np.asarray(X, dtype=float), y, rel_cost, seed)
    train_time[fam] += time.perf_counter() - t0
    return r


# =====================================================================================================
# Nodes, simulators, latency / WCET accounting
# =====================================================================================================
NODES = ('AB', 'BC', 'AC')
NODE_SRC_DST = {'AB': (0, 1), 'BC': (1, 2), 'AC': (0, 2)}      # router R_xy reads the features of stage x and gates stage y
NODE_GATED = {'AB': 1, 'BC': 2, 'AC': 2}                        # index of the stage the router gates

def subset(tel, idx):
    return dict(feat=tel['feat'][:, idx, :], conf=tel['feat'][:, idx, 0], correct=tel['correct'][:, idx])

def node_dataset(trn, node):
    s, t = NODE_SRC_DST[node]
    y = (~trn['correct'][s]) & trn['correct'][t]       
    return trn['feat'][s], y.astype(int)

def sim_3stage(conf, correct, adv_AB, adv_BC):
    n = conf.shape[1]
    acc_A = conf[0] >= TAU
    ask_A = ~acc_A
    run_B = ask_A & adv_AB
    acc_B = run_B & (conf[1] >= TAU)
    ask_B = run_B & ~acc_B
    run_C = ask_B & adv_BC
    exit_A = acc_A | (ask_A & ~adv_AB)
    exit_B = acc_B | (ask_B & ~adv_BC)
    final = np.full(n, 2, dtype=np.int64)
    final[exit_B] = 1
    final[exit_A] = 0
    assert np.all(exit_A.astype(int) + exit_B + run_C == 1), 'every input must exit exactly once'
    return dict(executed=np.vstack([np.ones(n, dtype=bool), run_B, run_C]),
                calls=ask_A.astype(np.int64) + ask_B.astype(np.int64), final=final, correct=correct[final, np.arange(n)])

def sim_2stage(conf, correct, adv_AC):
    n = conf.shape[1]
    ask_A = conf[0] < TAU
    run_C = ask_A & adv_AC
    final = np.where(run_C, 2, 0).astype(np.int64)
    return dict(executed=np.vstack([np.ones(n, dtype=bool), np.zeros(n, dtype=bool), run_C]),
                calls=ask_A.astype(np.int64), final=final, correct=correct[final, np.arange(n)])

def sim_expert(correct):
    n = correct.shape[1]
    ex = np.zeros((3, n), dtype=bool); ex[2] = True
    return dict(executed=ex, calls=np.zeros(n, dtype=np.int64), final=np.full(n, 2, dtype=np.int64), correct=correct[2])

def decide(router, feats):
    return router.predict_proba(feats)[:, 1] >= THETA

def simulate(arch, method_routers, sub):
    """method_routers: dict node->router, or None for the always-escalate (Pure Threshold) policy."""
    n = sub['conf'].shape[1]
    ones = np.ones(n, dtype=bool)
    if arch == '2-stage':
        adv = ones if method_routers is None else decide(method_routers['AC'], sub['feat'][0])
        return sim_2stage(sub['conf'], sub['correct'], adv)
    if method_routers is None:
        return sim_3stage(sub['conf'], sub['correct'], ones, ones)
    return sim_3stage(sub['conf'], sub['correct'],
                      decide(method_routers['AB'], sub['feat'][0]), decide(method_routers['BC'], sub['feat'][1]))

def summarize(arch, dname, seed, method, sim, Lm, Lx, O=0.0, Cr=0.0, router=False):
    ex = sim['executed'].T.astype(float)
    calls = sim['calls'].astype(float) if router else np.zeros(len(sim['calls']))
    lat_naive = ex @ Lm                                        
    lat_corr = lat_naive + calls * O                           
    wcet_obs = float((ex @ Lx + calls * Cr).max())
    frac = np.bincount(sim['final'], minlength=3) / len(sim['final']) * 100
    return {'Arch': arch, 'Dataset': dname, 'Seed': seed, 'Method': method,
            'Accuracy': float(sim['correct'].mean() * 100),
            'Lat_naive': float(lat_naive.mean()), 'Lat_corr': float(lat_corr.mean()),
            'Router_share': float(calls.mean() * O / lat_corr.mean() * 100),
            'Calls': float(calls.mean()), 'WCET_obs': wcet_obs,
            'End_A': frac[0], 'End_B': frac[1], 'End_C': frac[2]}

def wcet_bound(arch, method, Lx, ovh_max):
    if method == 'Pure Expert':
        return float(Lx[2])
    path = float(Lx[0] + Lx[2]) if arch == '2-stage' else float(Lx.sum())
    if method in ROUTER_FAMILY_OF:
        return path + MAX_ROUTER_CALLS[arch] * float(ovh_max[ROUTER_FAMILY_OF[method]])
    return path

def measure_router_overhead(router, X_rows, n_reps, n_warmup):
    rows = [np.ascontiguousarray(r, dtype=float).reshape(1, -1) for r in X_rows]
    n_warmup = min(n_warmup, max(len(rows) - 1, 0))
    for r in rows[:n_warmup]:
        router.predict_proba(r)
    ts = []
    for _ in range(n_reps):
        for r in rows:
            t0 = time.perf_counter()
            router.predict_proba(r)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)

# ---- real per-image cascades (used for the end-to-end validation) ----
def step_3stage(models, x, adv_AB, adv_BC):
    oA = stage_with_features(models[0], x)[0]
    fA, pA = oA[:3].astype(np.float64), int(oA[3])
    if fA[0] >= TAU:
        return pA, 0, 0
    if not adv_AB(fA):
        return pA, 0, 1
    oB = stage_with_features(models[1], x)[0]
    fB, pB = oB[:3].astype(np.float64), int(oB[3])
    if fB[0] >= TAU:
        return pB, 1, 1
    if not adv_BC(fB):
        return pB, 1, 2
    return int(stage_final(models[2], x)[0]), 2, 2

def step_2stage(models, x, adv_AC):
    oA = stage_with_features(models[0], x)[0]
    fA, pA = oA[:3].astype(np.float64), int(oA[3])
    if fA[0] >= TAU:
        return pA, 0, 0
    if not adv_AC(fA):
        return pA, 0, 1
    return int(stage_final(models[2], x)[0]), 2, 1

ALWAYS = lambda f: True

def make_step(arch, models, r):
    node_fn = lambda rt: (lambda f: bool(rt.predict_proba(f.reshape(1, -1))[0, 1] >= THETA))
    if r is None:
        return (lambda x: step_2stage(models, x, ALWAYS)) if arch == '2-stage' else (lambda x: step_3stage(models, x, ALWAYS, ALWAYS))
    if arch == '2-stage':
        return lambda x: step_2stage(models, x, node_fn(r['AC']))
    return lambda x: step_3stage(models, x, node_fn(r['AB']), node_fn(r['BC']))

def run_timed(step_fn, inputs, warmup, reps):
    for x in inputs[:warmup]:
        step_fn(x)
    times, finals, preds = [], [], []
    for _ in range(reps):
        for x in inputs:
            t0 = time.perf_counter()
            pred, fin, _calls = step_fn(x)
            times.append((time.perf_counter() - t0) * 1000.0)
            finals.append(fin)
            preds.append(pred)
    return np.array(times), np.array(finals), np.array(preds)

def e2e_validation(models, ds, tel, te_idx, routers_by_fam, Lm, Lx, ovh_mean, ovh_max, dname, seed, n_samples, warmup, reps):
    idx = te_idx[:n_samples]
    n = len(idx)
    inputs = [to_device_input(ds, int(i)) for i in idx]
    sub = subset(tel, idx)
    labels = tel['labels'][idx]
    rows = []
    for arch in ARCHS:
        plan = [('Pure Expert', lambda x: (int(stage_final(models[2], x)[0]), 2, 0), sim_expert(sub['correct']), 0.0, 0.0, False),
                ('Pure Threshold', make_step(arch, models, None), simulate(arch, None, sub), 0.0, 0.0, False)]
        for fam in FAMILIES:
            r = routers_by_fam[fam]
            plan.append((FAMILY_LABELS[fam], make_step(arch, models, r), simulate(arch, r, sub), ovh_mean[fam], ovh_max[fam], True))
        for label, step_fn, sim, O, Cr, is_router in plan:
            times, finals, preds = run_timed(step_fn, inputs, warmup, reps)
            an = summarize(arch, dname, seed, label, sim, Lm, Lx, O=O, Cr=Cr, router=is_router)
            rows.append({'Arch': arch, 'Method': label, 'N_inputs': n,
                         'Measured_mean_ms': times.mean(), 'Analytic_mean_ms': an['Lat_corr'],
                         'Error_pct': (times.mean() - an['Lat_corr']) / an['Lat_corr'] * 100,
                         'Measured_p99_ms': np.percentile(times, 99),
                         'Measured_max_WCET_ms': times.max(), 'Analytic_WCET_obs_ms': an['WCET_obs'],
                         'Measured_max_within_analytic': bool(times.max() <= an['WCET_obs']),
                         'Same_path_as_simulator_pct': (finals[:n] == sim['final']).mean() * 100,
                         'Accuracy_measured_pct': (preds[:n] == labels).mean() * 100})
            print('  {} {:<20s} measured {:8.3f} ms | analytic {:8.3f} ms | max {:8.3f} ms'.format(
                arch, label, times.mean(), an['Lat_corr'], times.max()), flush=True)
    return pd.DataFrame(rows)

def paired_row(a, b, comparison, metric, unit):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, p_t = stats.ttest_rel(a, b)
    try:
        p_w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p_w = np.nan
    return {'Comparison': comparison, 'Metric': metric, 'Unit': unit, 'Mean_diff': diff.mean(),
            'Std_diff': diff.std(ddof=1) if len(diff) > 1 else np.nan, 't_stat': t, 'p_ttest': p_t, 'p_wilcoxon': p_w}

def show(title, df, nd=3):
    print('\n' + title)
    print(df.round(nd).to_string(index=False))


# =====================================================================================================
# Main
# =====================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='2- and 3-stage IDK cascade router-overhead study (ResNet-18/34/50, ImageNetV2).')
    p.add_argument('--data_dir', type=str, default='./data_imagenetv2')
    p.add_argument('--output_dir', type=str, default='./results_2_3stage')
    p.add_argument('--dataset', type=str, default='TopImages', choices=list(DATASET_SPECS))
    p.add_argument('--seeds', type=int, nargs='*', default=list(range(42, 50)))
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--tau', type=float, default=0.90, help='confidence threshold (study value 0.90)')
    p.add_argument('--theta', type=float, default=0.20, help='router escalation threshold (study value 0.20)')
    p.add_argument('--batch_size', type=int, default=16, help='telemetry pass only')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--lat_reps', type=int, default=500)
    p.add_argument('--lat_warmup', type=int, default=30)
    p.add_argument('--overhead_rows', type=int, default=300)
    p.add_argument('--overhead_reps', type=int, default=3)
    p.add_argument('--overhead_warmup', type=int, default=20)
    p.add_argument('--e2e_samples', type=int, default=200)
    p.add_argument('--e2e_warmup', type=int, default=5)
    p.add_argument('--e2e_reps', type=int, default=1)
    p.add_argument('--skip_e2e', action='store_true')
    p.add_argument('--keep_archive', action='store_true')
    return p.parse_args(argv)

def run(args, results):
    global TAU, THETA, DATA_ROOT, BATCH_SIZE, NUM_WORKERS, DELETE_ARCHIVE_AFTER_EXTRACT
    TAU, THETA, DATA_ROOT = args.tau, args.theta, args.data_dir
    BATCH_SIZE, NUM_WORKERS = args.batch_size, args.num_workers
    DELETE_ARCHIVE_AFTER_EXTRACT = not args.keep_archive
    seeds, test_size, dname = args.seeds, args.test_size, args.dataset
    print('Execution device: {}{}'.format(device, ' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''))
    print('tau = {}   theta = {}   seeds = {}   dataset = {}'.format(TAU, THETA, seeds, dname))

    ds = prepare_dataset(dname)
    models = [load_imagenet_resnet(a) for a in STAGE_MODELS]

    # ---- per-stage latency / WCET on this device ----
    print('\nTiming each stage: {} warm-up + {} measured single-input runs ...'.format(args.lat_warmup, args.lat_reps), flush=True)
    st = profile_stages(models, to_device_input(ds, 0), args.lat_reps, args.lat_warmup)
    Lm, Lx = np.array([s['mean'] for s in st]), np.array([s['max'] for s in st])
    results['model_latency.csv'] = pd.DataFrame([{'Stage': STAGES[k], 'Model': STAGE_MODELS[k], 'Mean_ms': st[k]['mean'], 'Std_ms': st[k]['std'],
                                                  'P99_ms': st[k]['p99'], 'Max_WCET_ms': st[k]['max']} for k in range(3)])
    show('Per-stage latency on this device (batch size 1)', results['model_latency.csv'])

    # class-weight cost of each node = gated-stage latency relative to the cheapest gated stage (B)
    cheapest = min(Lm[1], Lm[2])
    rel_cost = {n: float(Lm[NODE_GATED[n]] / cheapest) for n in NODES}
    print('\nRelative gated-stage cost used for the class weights:', {k: round(v, 3) for k, v in rel_cost.items()})

    # ---- telemetry pass: softmax features of A, B, C for every image (computed once) ----
    print('\nTelemetry pass: {}'.format(dname), flush=True)
    tel = profile_dataset(dname, ds, models)
    print('  Top-1 accuracy per stage (%):  ' + '   '.join('{} {:.2f}'.format(s, tel['correct'][k].mean() * 100) for k, s in enumerate(STAGES)))
    if tel['correct'][2].mean() < 0.30:
        raise RuntimeError('ResNet-50 accuracy is implausibly low - labels/data are probably wrong.')
    N = tel['feat'].shape[1]

    # ---- train 5 families x 3 nodes x seeds in memory ----
    train_time = {f: 0.0 for f in FAMILIES}
    routers = {}                                    
    t_train = time.perf_counter()
    for si, seed in enumerate(seeds):
        tr_idx, _ = train_test_split(np.arange(N), test_size=test_size, random_state=seed)
        trn = subset(tel, tr_idx)
        for node in NODES:
            X, y = node_dataset(trn, node)
            if si == 0:
                print('  R_{}: {} training images, {:.1f}% positive, class weight {:.2f}'.format(
                    node, len(y), y.mean() * 100, cost_sensitive_weight(y, rel_cost[node])), flush=True)
            for fam in FAMILIES:
                routers.setdefault((seed, fam), {})[node] = timed_train(fam, X, y, rel_cost[node], seed, train_time)
        print('  seed {} trained ({}/{})'.format(seed, si + 1, len(seeds)), flush=True)
    total_train = time.perf_counter() - t_train
    tot = sum(train_time.values())
    results['compute_time.csv'] = pd.DataFrame([{'Router Family': FAMILY_LABELS[f], 'Time_s': train_time[f],
                                                 'Share_pct': train_time[f] / tot * 100 if tot > 0 else 0.0} for f in FAMILIES])
    show('Cumulative router training time (total wall-clock {:.1f} s)'.format(total_train), results['compute_time.csv'], 2)

    # ---- router overhead (bench router = R_AB of the first seed; rows = features of A that actually reach a router) ----
    _, te0 = train_test_split(np.arange(N), test_size=test_size, random_state=seeds[0])
    rows_x = subset(tel, te0)['feat'][0]
    rows_x = rows_x[rows_x[:, 0] < TAU]
    ovh_mean, ovh_max, recs_o = {}, {}, []
    print('\nMeasuring router overhead: {} rows x {} reps per family ...'.format(min(args.overhead_rows, len(rows_x)), args.overhead_reps), flush=True)
    for fam in FAMILIES:
        s = timing_stats(measure_router_overhead(routers[(seeds[0], fam)]['AB'], rows_x[:args.overhead_rows], args.overhead_reps, args.overhead_warmup))
        ovh_mean[fam], ovh_max[fam] = s['mean'], s['max']
        recs_o.append({'Router': FAMILY_LABELS[fam], 'Mean_ms': s['mean'], 'Std_ms': s['std'], 'P99_ms': s['p99'], 'Max_WCET_ms': s['max']})
    ovh_df = pd.DataFrame(recs_o)
    ovh_df['Mean_vs_cheapest_x'] = ovh_df['Mean_ms'] / ovh_df['Mean_ms'].min()
    ovh_df['Mean_pct_of_stage_A'] = ovh_df['Mean_ms'] / Lm[0] * 100
    results['router_overhead.csv'] = ovh_df
    show('Isolated router overhead (one decision, this device)', ovh_df, 4)

    # ---- evaluation on the held-out split of every seed ----
    recs = []
    for si, seed in enumerate(seeds):
        _, te_idx = train_test_split(np.arange(N), test_size=test_size, random_state=seed)
        tst = subset(tel, te_idx)
        for arch in ARCHS:
            recs.append(summarize(arch, dname, seed, 'Pure Expert', sim_expert(tst['correct']), Lm, Lx))
            recs.append(summarize(arch, dname, seed, 'Pure Threshold', simulate(arch, None, tst), Lm, Lx))
            for fam in FAMILIES:
                recs.append(summarize(arch, dname, seed, FAMILY_LABELS[fam], simulate(arch, routers[(seed, fam)], tst), Lm, Lx,
                                      O=ovh_mean[fam], Cr=ovh_max[fam], router=True))
        print('  seed {} evaluated ({}/{})'.format(seed, si + 1, len(seeds)), flush=True)
    R = pd.DataFrame(recs)
    results['trials.csv'] = R.rename(columns={'Accuracy': 'Accuracy_pct', 'Lat_naive': 'Naive_latency_ms', 'Lat_corr': 'Corrected_latency_ms',
                                              'Router_share': 'Router_share_pct', 'Calls': 'Router_calls_per_input',
                                              'WCET_obs': 'WCET_observed_paths_ms', 'End_A': 'Answered_by_A_pct',
                                              'End_B': 'Answered_by_B_pct', 'End_C': 'Answered_by_C_pct'})

    sum_rows, wcet_rows, path_rows, sig_rows = [], [], [], []
    for arch in ARCHS:
        d = R[R.Arch == arch]
        exp_bound = wcet_bound(arch, 'Pure Expert', Lx, ovh_max)
        for mth in METHOD_ORDER:
            g = d[d.Method == mth]
            is_r = mth in ROUTER_FAMILY_OF
            sum_rows.append({'Arch': arch, 'Method': mth, 'N_seeds': len(g),
                             'Accuracy_mean_pct': g.Accuracy.mean(), 'Accuracy_std_pct': g.Accuracy.std(),
                             'Naive_latency_mean_ms': g.Lat_naive.mean(), 'Naive_latency_std_ms': g.Lat_naive.std(),
                             'Corrected_latency_mean_ms': g.Lat_corr.mean(), 'Corrected_latency_std_ms': g.Lat_corr.std(),
                             'Router_overhead_ms_per_call': ovh_mean[ROUTER_FAMILY_OF[mth]] if is_r else 0.0,
                             'Router_share_mean_pct': g.Router_share.mean() if is_r else 0.0,
                             'Router_calls_per_input': g.Calls.mean() if is_r else 0.0})
            b = wcet_bound(arch, mth, Lx, ovh_max)
            wcet_rows.append({'Arch': arch, 'Method': mth, 'WCET_bound_ms': b, 'WCET_observed_paths_ms': g.WCET_obs.max(),
                              'Bound_vs_Pure_Expert_x': b / exp_bound, 'Bound_exceeds_Pure_Expert': b > exp_bound})
            path_rows.append({'Arch': arch, 'Method': mth, 'Answered_by_A_pct': g.End_A.mean(),
                              'Answered_by_B_pct': g.End_B.mean(), 'Answered_by_C_pct': g.End_C.mean()})
        lat = d.pivot(index='Seed', columns='Method', values='Lat_corr')
        acc = d.pivot(index='Seed', columns='Method', values='Accuracy')
        rn = [FAMILY_LABELS[f] for f in FAMILIES]
        for a, b2 in itertools.combinations(rn, 2):
            sig_rows.append(dict(Arch=arch, **paired_row(lat[a], lat[b2], '{} vs. {}'.format(a, b2), 'corrected_latency', 'ms')))
            sig_rows.append(dict(Arch=arch, **paired_row(acc[a], acc[b2], '{} vs. {}'.format(a, b2), 'accuracy', 'pp')))
        for a in rn:
            for ref in ('Pure Expert', 'Pure Threshold'):
                sig_rows.append(dict(Arch=arch, **paired_row(lat[a], lat[ref], '{} vs. {}'.format(a, ref), 'corrected_latency', 'ms')))
                sig_rows.append(dict(Arch=arch, **paired_row(acc[a], acc[ref], '{} vs. {}'.format(a, ref), 'accuracy', 'pp')))
    summary, wcet, paths = pd.DataFrame(sum_rows), pd.DataFrame(wcet_rows), pd.DataFrame(path_rows)
    results['summary.csv'], results['wcet.csv'], results['path_distribution.csv'] = summary, wcet, paths
    results['significance.csv'] = pd.DataFrame(sig_rows)
    for arch in ARCHS:
        show('[{}] Accuracy and average-case latency (mean over {} seeds)'.format(arch, len(seeds)),
             summary[summary.Arch == arch].drop(columns=['Arch', 'N_seeds']))
        show('[{}] Worst-case execution time (measured high-water marks)'.format(arch), wcet[wcet.Arch == arch].drop(columns='Arch'))
        show('[{}] Which stage gives the final answer (%)'.format(arch), paths[paths.Arch == arch].drop(columns='Arch'), 2)

    # ---- real end-to-end validation (first seed) ----
    if not args.skip_e2e:
        _, te_first = train_test_split(np.arange(N), test_size=test_size, random_state=seeds[0])
        print('\nReal end-to-end validation ({} inputs, seed {}) ...'.format(min(args.e2e_samples, len(te_first)), seeds[0]), flush=True)
        by_fam = {f: routers[(seeds[0], f)] for f in FAMILIES}
        e2e = e2e_validation(models, ds, tel, te_first, by_fam, Lm, Lx, ovh_mean, ovh_max, dname, seeds[0],
                             args.e2e_samples, args.e2e_warmup, args.e2e_reps)
        results['e2e_validation.csv'] = e2e
        show('Measured (real wall-clock) vs. analytic', e2e)

    results['run_info.json'] = dict(
        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        python=sys.version, platform=platform.platform(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgb.__version__,
                      torch=torch.__version__, torchvision=torchvision.__version__),
        tau=TAU, theta=THETA, seeds=seeds, test_size=test_size, dataset=dname, class_weight_relative_cost=rel_cost,
        stage_latency_ms={STAGES[k]: float(Lm[k]) for k in range(3)}, train_wallclock_s=total_train,
        note='Set the Jetson power mode / clocks (nvpmodel -m 0; jetson_clocks) before running for stable WCET numbers.')

def write_results(out, results):
    os.makedirs(out, exist_ok=True)
    for name, obj in results.items():
        path = os.path.join(out, name)
        if name.endswith('.json'):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, default=str)
        else:
            obj.to_csv(path, index=False)
    print('\nWrote {} files to {}:'.format(len(results), os.path.abspath(out)))
    for name in results:
        print('  ' + name)

def main(argv=None):
    args = parse_args(argv)
    results = {}                 
    completed = False
    try:
        run(args, results)
        completed = True
    finally:
        if results:
            if not completed:
                print('\nThe run did NOT finish - saving the partial results collected so far.')
            write_results(args.output_dir, results)

if __name__ == '__main__':
    main()
