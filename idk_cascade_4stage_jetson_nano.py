#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
import sklearn
import xgboost
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- defaults ----------------------------------------------
TAU = 0.90                       
THETA = 0.20                     
DATA_ROOT = './data_imagenetv2'
DELETE_ARCHIVE_AFTER_EXTRACT = True
BATCH_SIZE = 16
NUM_WORKERS = 2
STAGES = ['A', 'B', 'C', 'D']
STAGE_MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet152']
MAX_ROUTER_CALLS = 3             # the longest path (A->B->C->D) consults 3 routers
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
WEIGHT_ENUMS = {'resnet18': 'ResNet18_Weights', 'resnet34': 'ResNet34_Weights',
                'resnet50': 'ResNet50_Weights', 'resnet152': 'ResNet152_Weights'}

def load_imagenet_resnet(arch):
    ctor = getattr(torchvision.models, arch)
    enum = getattr(torchvision.models, WEIGHT_ENUMS[arch], None)
    if enum is not None:
        model = ctor(weights=enum.IMAGENET1K_V1)
    else:                                                    
        model = ctor(pretrained=True)
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
    fns = [stage_with_features, stage_with_features, stage_with_features, stage_final]
    return [timing_stats(time_stage(fns[k], models[k], probe_x, n_reps, n_warmup)) for k in range(4)]

def profile_dataset(name, ds, models):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=(device.type == 'cuda'))
    out = [[] for _ in range(4)]
    labels, n_done, t0 = [], 0, time.perf_counter()
    for bi, (imgs, labs) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        for k in range(4):
            out[k].append(stage_with_features(models[k], imgs))
        labels.append(labs.numpy())
        n_done += len(labs)
        if (bi + 1) % 20 == 0 or n_done == len(ds):
            el = time.perf_counter() - t0
            print('  {}: {}/{} images | {:.0f}s elapsed | ETA {:.0f}s'.format(
                name, n_done, len(ds), el, el / n_done * (len(ds) - n_done)), flush=True)
    raw = np.stack([np.concatenate(o) for o in out]).astype(np.float64)       # (4, N, 4)
    labels = np.concatenate(labels)
    pred = raw[:, :, 3].astype(np.int64)
    feat = np.ascontiguousarray(raw[:, :, :3])                                # (4, N, 3): conf, entropy, margin
    return dict(feat=feat, pred=pred, correct=(pred == labels[None, :]), labels=labels)


import numpy as np  
class HeuristicRouter:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        score = X_arr @ self.weights[:3] + self.weights[3]
        prob = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob, prob])


# =====================================================================================================
# Nodes, simulator, latency / WCET accounting
# =====================================================================================================
NODES = ('AB', 'BC', 'CD')
NODE_STAGE = {'AB': 0, 'BC': 1, 'CD': 2}          # router R_xy reads the features of stage x and gates stage y

def subset(tel, idx):
    return dict(feat=tel['feat'][:, idx, :], conf=tel['feat'][:, idx, 0], correct=tel['correct'][:, idx])

def node_dataset(trn, node):
    s = NODE_STAGE[node]
    y = (~trn['correct'][s]) & trn['correct'][s + 1]
    return trn['feat'][s], y.astype(int)

def router_filename(dname, seed, family, node):
    return '{}_{}_{}_{}.joblib'.format(dname, seed, family, node.lower())


def simulate_policy(conf, correct, adv_A, adv_B, adv_C, tau=None):
    tau = TAU if tau is None else tau
    n = conf.shape[1]
    acc_A = conf[0] >= tau
    ask_A = ~acc_A                                   
    run_B = ask_A & adv_A
    acc_B = run_B & (conf[1] >= tau)
    ask_B = run_B & ~acc_B                           
    run_C = ask_B & adv_B
    acc_C = run_C & (conf[2] >= tau)
    ask_C = run_C & ~acc_C                           
    run_D = ask_C & adv_C
    exit_A = acc_A | (ask_A & ~adv_A)                
    exit_B = acc_B | (ask_B & ~adv_B)
    exit_C = acc_C | (ask_C & ~adv_C)
    final = np.full(n, 3, dtype=np.int64)
    final[exit_C] = 2
    final[exit_B] = 1
    final[exit_A] = 0
    assert np.all(exit_A.astype(int) + exit_B + exit_C + run_D == 1), 'every input must exit exactly once'
    assert np.array_equal(final == 3, run_D), 'simulator invariant violated'
    return dict(executed=np.vstack([np.ones(n, dtype=bool), run_B, run_C, run_D]),
                calls=ask_A.astype(np.int64) + ask_B.astype(np.int64) + ask_C.astype(np.int64),
                final=final, correct=correct[final, np.arange(n)])

def expert_policy(correct):
    n = correct.shape[1]
    ex = np.zeros((4, n), dtype=bool); ex[3] = True
    return dict(executed=ex, calls=np.zeros(n, dtype=np.int64), final=np.full(n, 3, dtype=np.int64), correct=correct[3])

def router_decisions(routers, sub):
    adv_A = routers['AB'].predict_proba(sub['feat'][0])[:, 1] >= THETA
    adv_B = routers['BC'].predict_proba(sub['feat'][1])[:, 1] >= THETA
    adv_C = routers['CD'].predict_proba(sub['feat'][2])[:, 1] >= THETA
    return adv_A, adv_B, adv_C

def summarize(dname, seed, method, sim, Lm, Lx, O=0.0, Cr=0.0, router=False):
    ex = sim['executed'].T.astype(float)                       
    calls = sim['calls'].astype(float) if router else np.zeros(len(sim['calls']))
    lat_naive = ex @ Lm                                        
    lat_corr = lat_naive + calls * O                           
    wcet_obs = float((ex @ Lx + calls * Cr).max())             
    frac = np.bincount(sim['final'], minlength=4) / len(sim['final']) * 100
    return {'Dataset': dname, 'Seed': seed, 'Method': method,
            'Accuracy': float(sim['correct'].mean() * 100),
            'Lat_naive': float(lat_naive.mean()), 'Lat_corr': float(lat_corr.mean()),
            'Router_share': float(calls.mean() * O / lat_corr.mean() * 100),
            'Calls': float(calls.mean()), 'WCET_obs': wcet_obs,
            'End_A': frac[0], 'End_B': frac[1], 'End_C': frac[2], 'End_D': frac[3]}


# =====================================================================================================
# Script-only helpers
# =====================================================================================================
def load_manifest(router_dir):
    path = os.path.join(router_dir, 'manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError('Missing {} - did you copy the whole exported router folder from '
                                'train_routers.ipynb (unzip exported_routers.zip) and pass it as --router_dir?'.format(path))
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_router(router_dir, filename):
    import __main__
    if not hasattr(__main__, 'HeuristicRouter'):     
        __main__.HeuristicRouter = HeuristicRouter
    path = os.path.join(router_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError('Missing router artifact: {}'.format(path))
    return joblib.load(path)

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

def wcet_bound(method, stage_max, ovh_max):
    if method == 'Pure Expert':
        return float(stage_max[3])
    full_path = float(stage_max.sum())
    if method in ROUTER_FAMILY_OF:
        return full_path + MAX_ROUTER_CALLS * float(ovh_max[ROUTER_FAMILY_OF[method]])
    return full_path                                                  

ALWAYS = lambda f: True

def cascade_step(models, x, adv_A, adv_B, adv_C):
    calls = 0
    oA = stage_with_features(models[0], x)[0]
    fA, pA = oA[:3].astype(np.float64), int(oA[3])
    if fA[0] >= TAU:
        return pA, 0, calls
    calls += 1                                          
    if not adv_A(fA):
        return pA, 0, calls
    oB = stage_with_features(models[1], x)[0]
    fB, pB = oB[:3].astype(np.float64), int(oB[3])
    if fB[0] >= TAU:
        return pB, 1, calls
    calls += 1                                          
    if not adv_B(fB):
        return pB, 1, calls
    oC = stage_with_features(models[2], x)[0]
    fC, pC = oC[:3].astype(np.float64), int(oC[3])
    if fC[0] >= TAU:
        return pC, 2, calls
    calls += 1                                          
    if not adv_C(fC):
        return pC, 2, calls
    return int(stage_final(models[3], x)[0]), 3, calls

def make_router_step(models, r):
    ab = lambda f: bool(r['AB'].predict_proba(f.reshape(1, -1))[0, 1] >= THETA)
    bc = lambda f: bool(r['BC'].predict_proba(f.reshape(1, -1))[0, 1] >= THETA)
    cd = lambda f: bool(r['CD'].predict_proba(f.reshape(1, -1))[0, 1] >= THETA)
    return lambda x: cascade_step(models, x, ab, bc, cd)

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

def e2e_validation(models, ds, tel, te_idx, routers_by_fam, Lm, Lx, ovh_mean, ovh_max, dname, seed,
                   n_samples, warmup, reps):
    idx = te_idx[:n_samples]
    n = len(idx)
    inputs = [to_device_input(ds, int(i)) for i in idx]
    sub = subset(tel, idx)
    labels = tel['labels'][idx]
    ones = np.ones(n, dtype=bool)
    plan = [('Pure Expert', lambda x: (int(stage_final(models[3], x)[0]), 3, 0), expert_policy(sub['correct']), 0.0, 0.0, False),
            ('Pure Threshold', lambda x: cascade_step(models, x, ALWAYS, ALWAYS, ALWAYS),
             simulate_policy(sub['conf'], sub['correct'], ones, ones, ones), 0.0, 0.0, False)]
    for fam in FAMILIES:
        r = routers_by_fam[fam]
        plan.append((FAMILY_LABELS[fam], make_router_step(models, r),
                     simulate_policy(sub['conf'], sub['correct'], *router_decisions(r, sub)),
                     ovh_mean[fam], ovh_max[fam], True))
    rows = []
    for label, step_fn, sim, O, Cr, is_router in plan:
        times, finals, preds = run_timed(step_fn, inputs, warmup, reps)
        an = summarize(dname, seed, label, sim, Lm, Lx, O=O, Cr=Cr, router=is_router)
        rows.append({'Method': label, 'N_inputs': n,
                     'Measured_mean_ms': times.mean(), 'Analytic_mean_ms': an['Lat_corr'],
                     'Error_pct': (times.mean() - an['Lat_corr']) / an['Lat_corr'] * 100,
                     'Measured_p99_ms': np.percentile(times, 99),
                     'Measured_max_WCET_ms': times.max(), 'Analytic_WCET_obs_ms': an['WCET_obs'],
                     'Measured_max_within_analytic': bool(times.max() <= an['WCET_obs']),
                     'Same_path_as_simulator_pct': (finals[:n] == sim['final']).mean() * 100,
                     'Accuracy_measured_pct': (preds[:n] == labels).mean() * 100})
        print('  {:<20s} measured mean {:8.3f} ms | analytic {:8.3f} ms | max {:8.3f} ms'.format(
            label, times.mean(), an['Lat_corr'], times.max()), flush=True)
    return pd.DataFrame(rows)

def paired_row(a, b, comparison, metric, unit):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, p_t = stats.ttest_rel(a, b)
    try:
        p_w = stats.wilcoxon(a, b).pvalue
    except ValueError:                                  # all differences zero
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
    p = argparse.ArgumentParser(description='4-stage IDK cascade router-overhead study on the Jetson (loads exported routers).')
    p.add_argument('--router_dir', type=str, required=True, help='folder exported by train_routers.ipynb (contains manifest.json)')
    p.add_argument('--data_dir', type=str, default='./data_imagenetv2', help='where ImageNetV2 is (or will be downloaded to)')
    p.add_argument('--output_dir', type=str, default='./results', help='where the CSV files are written')
    p.add_argument('--datasets', nargs='*', default=None, help='subset of the exported datasets (default: all)')
    p.add_argument('--tau', type=float, default=0.90,
                   help='SENSITIVITY-ANALYSIS FLAG ONLY: confidence threshold. 0.90 is the value of the study.')
    p.add_argument('--theta', type=float, default=None,
                   help='router threshold. Default: the value the routers were trained with (manifest).')
    p.add_argument('--batch_size', type=int, default=16, help='telemetry pass only; lower it if the Nano runs out of memory')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--lat_reps', type=int, default=500, help='timed single-input runs per stage (more = better WCET estimate)')
    p.add_argument('--lat_warmup', type=int, default=30)
    p.add_argument('--overhead_rows', type=int, default=300)
    p.add_argument('--overhead_reps', type=int, default=3)
    p.add_argument('--overhead_warmup', type=int, default=20)
    p.add_argument('--e2e_samples', type=int, default=200, help='inputs for the real end-to-end validation')
    p.add_argument('--e2e_warmup', type=int, default=5)
    p.add_argument('--e2e_reps', type=int, default=1)
    p.add_argument('--skip_e2e', action='store_true')
    p.add_argument('--keep_archive', action='store_true', help='keep the downloaded .tar.gz after extraction')
    return p.parse_args(argv)

def run(args, results):
    global TAU, THETA, DATA_ROOT, BATCH_SIZE, NUM_WORKERS, DELETE_ARCHIVE_AFTER_EXTRACT
    TAU, DATA_ROOT = args.tau, args.data_dir
    BATCH_SIZE, NUM_WORKERS = args.batch_size, args.num_workers
    DELETE_ARCHIVE_AFTER_EXTRACT = not args.keep_archive

    manifest = load_manifest(args.router_dir)
    THETA = manifest['theta'] if args.theta is None else args.theta
    seeds, test_size = manifest['seeds'], manifest['test_size']
    datasets = args.datasets or list(manifest['datasets'].keys())
    for d in datasets:
        if d not in manifest['datasets']:
            raise ValueError('Dataset {} was not exported (exported: {})'.format(d, list(manifest['datasets'])))
    if TAU != 0.90:
        print('!' * 70)
        print('WARNING: confidence threshold overridden to {} (study value: 0.90).'.format(TAU))
        print('!' * 70)
    if THETA != manifest['theta']:
        print('WARNING: theta={} differs from the value the routers were trained with ({}).'.format(THETA, manifest['theta']))
    print('Execution device: {}{}'.format(device, ' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''))
    print('Loading routers from: {}   |   tau = {}   theta = {}   seeds = {}'.format(args.router_dir, TAU, THETA, seeds))
    mv = manifest.get('versions', {})
    for pkg, cur in (('scikit-learn', sklearn.__version__), ('xgboost', xgboost.__version__)):
        if mv.get(pkg) and mv[pkg] != cur:
            print('WARNING: {} is {} here but the routers were exported with {} - unpickling may fail.'.format(pkg, cur, mv[pkg]))


    wanted = ['bench_{}.joblib'.format(f) for f in FAMILIES]
    wanted += [router_filename(d, s, f, n) for d in datasets for s in seeds for f in FAMILIES for n in NODES]
    missing = [w for w in wanted if not os.path.exists(os.path.join(args.router_dir, w))]
    if missing:
        raise FileNotFoundError('{} router files are missing, e.g. {}'.format(len(missing), missing[:3]))

    # ---- data ----
    ds_objs = {}
    for name in datasets:
        ds = prepare_dataset(name)
        fp, expected = dataset_fingerprint(ds), manifest['datasets'][name]['fingerprint']
        if fp != expected:
            raise RuntimeError('The {} image list here ({} images, sha1 {}...) differs from the one the routers were trained '
                               'on ({} images, sha1 {}...). The train/test split indices would not line up - use the same '
                               'ImageNetV2 archive.'.format(name, fp['n_images'], fp['sha1'][:10], expected['n_images'], expected['sha1'][:10]))
        ds_objs[name] = ds

    # ---- models and per-stage latency / WCET ----
    models = [load_imagenet_resnet(a) for a in STAGE_MODELS]
    print('\nTiming each stage: {} warm-up + {} measured single-input runs ...'.format(args.lat_warmup, args.lat_reps), flush=True)
    st = profile_stages(models, to_device_input(ds_objs[datasets[0]], 0), args.lat_reps, args.lat_warmup)
    Lm, Lx = np.array([s['mean'] for s in st]), np.array([s['max'] for s in st])
    lat_df = pd.DataFrame([{'Stage': STAGES[k], 'Model': STAGE_MODELS[k], 'Mean_ms': st[k]['mean'], 'Std_ms': st[k]['std'],
                            'P99_ms': st[k]['p99'], 'Max_WCET_ms': st[k]['max']} for k in range(4)])
    results['model_latency.csv'] = lat_df
    show('Per-stage latency on this device (batch size 1)', lat_df)

    # ---- telemetry pass on this device ----
    tel = {}
    for name in datasets:
        print('\nTelemetry pass: {}'.format(name), flush=True)
        tel[name] = profile_dataset(name, ds_objs[name], models)
        t = tel[name]
        print('  Top-1 accuracy per stage (%):  ' + '   '.join('{} {:.2f}'.format(s, t['correct'][k].mean() * 100) for k, s in enumerate(STAGES)))
        if t['correct'][3].mean() < 0.30:
            raise RuntimeError('ResNet-152 accuracy is implausibly low - labels/data are probably wrong.')

    # ---- router overhead (bench routers = R_AB of the first dataset / first seed) ----
    first = datasets[0]
    _, te0 = train_test_split(np.arange(tel[first]['feat'].shape[1]), test_size=test_size, random_state=seeds[0])
    rows_x = subset(tel[first], te0)['feat'][0]
    rows_x = rows_x[rows_x[:, 0] < TAU]                    
    ovh_mean, ovh_max, recs_o = {}, {}, []
    print('\nMeasuring router overhead: {} rows x {} reps per family ...'.format(min(args.overhead_rows, len(rows_x)), args.overhead_reps), flush=True)
    for fam in FAMILIES:
        bench = load_router(args.router_dir, 'bench_{}.joblib'.format(fam))
        s = timing_stats(measure_router_overhead(bench, rows_x[:args.overhead_rows], args.overhead_reps, args.overhead_warmup))
        ovh_mean[fam], ovh_max[fam] = s['mean'], s['max']
        recs_o.append({'Router': FAMILY_LABELS[fam], 'Mean_ms': s['mean'], 'Std_ms': s['std'], 'P99_ms': s['p99'], 'Max_WCET_ms': s['max']})
    ovh_df = pd.DataFrame(recs_o)
    ovh_df['Mean_vs_cheapest_x'] = ovh_df['Mean_ms'] / ovh_df['Mean_ms'].min()
    ovh_df['Mean_pct_of_stage_A'] = ovh_df['Mean_ms'] / Lm[0] * 100
    results['router_overhead.csv'] = ovh_df
    show('Isolated router overhead (one decision, measured on this device)', ovh_df, 4)

    # ---- evaluation of every method on the held-out test split of every seed ----
    recs, routers_first, te_first = [], {}, {}
    t_eval = time.perf_counter()
    for dname in datasets:
        N = tel[dname]['feat'].shape[1]
        for si, seed in enumerate(seeds):
            _, te_idx = train_test_split(np.arange(N), test_size=test_size, random_state=seed)
            tst = subset(tel[dname], te_idx)
            ones = np.ones(len(te_idx), dtype=bool)
            recs.append(summarize(dname, seed, 'Pure Expert', expert_policy(tst['correct']), Lm, Lx))
            recs.append(summarize(dname, seed, 'Pure Threshold', simulate_policy(tst['conf'], tst['correct'], ones, ones, ones), Lm, Lx))
            by_fam = {}
            for fam in FAMILIES:
                routers = {n: load_router(args.router_dir, router_filename(dname, seed, fam, n)) for n in NODES}
                by_fam[fam] = routers
                sim = simulate_policy(tst['conf'], tst['correct'], *router_decisions(routers, tst))
                recs.append(summarize(dname, seed, FAMILY_LABELS[fam], sim, Lm, Lx, O=ovh_mean[fam], Cr=ovh_max[fam], router=True))
            if si == 0:
                routers_first[dname], te_first[dname] = by_fam, te_idx
            print('  {} seed {} evaluated ({}/{})'.format(dname, seed, si + 1, len(seeds)), flush=True)
    eval_s = time.perf_counter() - t_eval
    R = pd.DataFrame(recs)

    # ---- CSV tables ----
    trials = R.rename(columns={'Accuracy': 'Accuracy_pct', 'Lat_naive': 'Naive_latency_ms', 'Lat_corr': 'Corrected_latency_ms',
                               'Router_share': 'Router_share_pct', 'Calls': 'Router_calls_per_input', 'WCET_obs': 'WCET_observed_paths_ms',
                               'End_A': 'Answered_by_A_pct', 'End_B': 'Answered_by_B_pct', 'End_C': 'Answered_by_C_pct', 'End_D': 'Answered_by_D_pct'})
    results['trials.csv'] = trials

    exp_bound = wcet_bound('Pure Expert', Lx, ovh_max)
    sum_rows, wcet_rows, path_rows = [], [], []
    for dname in datasets:
        d = R[R.Dataset == dname]
        for mth in METHOD_ORDER:
            g = d[d.Method == mth]
            is_r = mth in ROUTER_FAMILY_OF
            sum_rows.append({'Dataset': dname, 'Method': mth, 'N_seeds': len(g),
                             'Accuracy_mean_pct': g.Accuracy.mean(), 'Accuracy_std_pct': g.Accuracy.std(),
                             'Naive_latency_mean_ms': g.Lat_naive.mean(), 'Naive_latency_std_ms': g.Lat_naive.std(),
                             'Corrected_latency_mean_ms': g.Lat_corr.mean(), 'Corrected_latency_std_ms': g.Lat_corr.std(),
                             'Router_overhead_ms_per_call': ovh_mean[ROUTER_FAMILY_OF[mth]] if is_r else 0.0,
                             'Router_share_mean_pct': g.Router_share.mean() if is_r else 0.0,
                             'Router_calls_per_input': g.Calls.mean() if is_r else 0.0})
            b = wcet_bound(mth, Lx, ovh_max)
            wcet_rows.append({'Dataset': dname, 'Method': mth, 'WCET_bound_ms': b, 'WCET_observed_paths_ms': g.WCET_obs.max(),
                              'Bound_vs_Pure_Expert_x': b / exp_bound, 'Bound_exceeds_Pure_Expert': b > exp_bound})
            path_rows.append({'Dataset': dname, 'Method': mth, 'Answered_by_A_pct': g.End_A.mean(), 'Answered_by_B_pct': g.End_B.mean(),
                              'Answered_by_C_pct': g.End_C.mean(), 'Answered_by_D_pct': g.End_D.mean()})
    summary, wcet, paths = pd.DataFrame(sum_rows), pd.DataFrame(wcet_rows), pd.DataFrame(path_rows)
    results['summary.csv'] = summary
    results['wcet.csv'] = wcet
    results['path_distribution.csv'] = paths

    sig_rows = []
    rn = [FAMILY_LABELS[f] for f in FAMILIES]
    for dname in datasets:
        d = R[R.Dataset == dname]
        lat = d.pivot(index='Seed', columns='Method', values='Lat_corr')
        acc = d.pivot(index='Seed', columns='Method', values='Accuracy')
        for a, b in itertools.combinations(rn, 2):
            sig_rows.append(dict(Dataset=dname, **paired_row(lat[a], lat[b], '{} vs. {}'.format(a, b), 'corrected_latency', 'ms')))
            sig_rows.append(dict(Dataset=dname, **paired_row(acc[a], acc[b], '{} vs. {}'.format(a, b), 'accuracy', 'pp')))
        for a in rn:
            for ref in ('Pure Expert', 'Pure Threshold'):
                sig_rows.append(dict(Dataset=dname, **paired_row(lat[a], lat[ref], '{} vs. {}'.format(a, ref), 'corrected_latency', 'ms')))
                sig_rows.append(dict(Dataset=dname, **paired_row(acc[a], acc[ref], '{} vs. {}'.format(a, ref), 'accuracy', 'pp')))
    results['significance.csv'] = pd.DataFrame(sig_rows)

    # ---- printed results ----
    for dname in datasets:
        show('[{}] Accuracy and average-case latency (mean over {} seeds)'.format(dname, len(seeds)),
             summary[summary.Dataset == dname].drop(columns=['Dataset', 'N_seeds']), 3)
        show('[{}] Worst-case execution time (measured high-water marks)'.format(dname), wcet[wcet.Dataset == dname].drop(columns='Dataset'), 3)
        show('[{}] Which stage gives the final answer (%)'.format(dname), paths[paths.Dataset == dname].drop(columns='Dataset'), 2)

    # ---- real end-to-end validation ----
    if not args.skip_e2e:
        dname = datasets[0]
        print('\nReal end-to-end validation on {} ({} inputs, seed {}) ...'.format(dname, min(args.e2e_samples, len(te_first[dname])), seeds[0]), flush=True)
        e2e = e2e_validation(models, ds_objs[dname], tel[dname], te_first[dname], routers_first[dname], Lm, Lx, ovh_mean, ovh_max,
                             dname, seeds[0], args.e2e_samples, args.e2e_warmup, args.e2e_reps)
        results['e2e_validation.csv'] = e2e
        show('Measured (real wall-clock) vs. analytic', e2e)

    info = dict(device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                python=sys.version, platform=platform.platform(),
                versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                              torch=torch.__version__, torchvision=torchvision.__version__),
                tau=TAU, theta=THETA, seeds=seeds, test_size=test_size, datasets=datasets, batch_size=BATCH_SIZE,
                lat_reps=args.lat_reps, overhead_rows=args.overhead_rows, overhead_reps=args.overhead_reps,
                e2e_samples=args.e2e_samples, evaluation_seconds=eval_s, router_dir=os.path.abspath(args.router_dir),
                note='Set the Jetson power mode / clocks (nvpmodel -m 0; jetson_clocks) before running for stable WCET numbers.')
    results['run_info.json'] = info


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
