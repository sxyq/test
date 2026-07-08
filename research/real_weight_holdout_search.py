#!/usr/bin/env python3
import gc
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


REPO_ROOT = Path("/Users/sunyiyang/Desktop/Project/Baidu  GRAB")
DATASET_ROOT = REPO_ROOT / "research_datasets" / "2026_cti_data_meta" / "dataset" / "history"
WORK_ROOT = REPO_ROOT / "12_workspace" / "holdout_proxy"
WEIGHTS_PATH = REPO_ROOT / "weights" / "ckpt.pt"
MODULE_PATH = REPO_ROOT / "submission" / "V178-FFN-SVD32-sxyq" / "infer.py"
STAGE1_RATIOS = [0.25, 0.40, 0.45, 0.50]
STAGE2_RANKS = [128, 96, 64]
BIAS_CANDIDATES = [0.0, math.log(1.0 / 1.05913)]
HISTORY_FILES = ["train_0000.csv", "train_0001.csv"]
LINES_PER_FILE = 5000
MODULE_TAG_COUNTER = 0


def prepare_subset_files():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    subset_paths = []
    for name in HISTORY_FILES:
        src = DATASET_ROOT / name
        dst = WORK_ROOT / name
        if not dst.exists():
            with open(src, "r") as fin, open(dst, "w") as fout:
                for idx, line in enumerate(fin):
                    if idx >= LINES_PER_FILE:
                        break
                    fout.write(line)
        subset_paths.append(dst)
    return subset_paths


def load_version_module(env_overrides):
    global MODULE_TAG_COUNTER
    MODULE_TAG_COUNTER += 1
    old_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    try:
        spec = importlib.util.spec_from_file_location(f"holdout_search_{MODULE_TAG_COUNTER}", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_holdout(module, subset_paths):
    item_dict, user_seq = module.load_sample_files(subset_paths)
    ordered = defaultdict(list)
    for logid, rec in item_dict.items():
        ordered[rec["userid"]].append((rec["timestamp"], logid, rec["clk"]))
    holdout_logids = []
    holdout_labels = []
    for _, items in ordered.items():
        items.sort(key=lambda x: x[0])
        if len(items) >= 2:
            holdout_logids.append(items[-1][1])
            holdout_labels.append(float(items[-1][2]))
    pred_logids = set(holdout_logids)
    dataset = module.CTRUserDataset(
        item_dict,
        user_seq,
        max_feasign_per_slot={1: 2},
        pred_logids=pred_logids,
    )
    loader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=False,
        num_workers=0,
        collate_fn=module.make_collate_fn(dataset.max_slot_id),
    )
    return loader, holdout_logids, np.array(holdout_labels, dtype=np.float64), {
        "rows": len(item_dict),
        "users": len(ordered),
        "holdout_count": len(holdout_logids),
        "holdout_positives": int(sum(holdout_labels)),
    }


def evaluate_biases(logits, labels):
    results = []
    if labels.mean() > 0:
        low, high = -4.0, 4.0
        for _ in range(40):
            mid = (low + high) / 2.0
            probs = 1.0 / (1.0 + np.exp(-(logits + mid)))
            if probs.mean() > labels.mean():
                high = mid
            else:
                low = mid
        BIAS_CANDIDATES_EXT = BIAS_CANDIDATES + [mid]
    else:
        BIAS_CANDIDATES_EXT = list(BIAS_CANDIDATES)
    seen = set()
    for bias in BIAS_CANDIDATES_EXT:
        key = round(float(bias), 8)
        if key in seen:
            continue
        seen.add(key)
        probs = 1.0 / (1.0 + np.exp(-(logits + bias)))
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        pcoc = float(probs.mean() / labels.mean()) if labels.mean() > 0 else float("inf")
        results.append(
            {
                "bias": float(bias),
                "auc": float(auc),
                "pcoc": pcoc,
                "pred_mean": float(probs.mean()),
            }
        )
    results.sort(key=lambda x: (abs(x["pcoc"] - 1.0), -x["auc"]))
    return results


def run_config(config_name, prune_ratio, svd_rank, subset_paths):
    env_overrides = {
        "GRAB_V14_FP16_WEIGHTS": "0",
        "GRAB_V15_FP16_DEEP": "0",
        "GRAB_V16_FP16_EMB": "0",
        "GRAB_V129_LOGIT_BIAS": "0.0",
        "GRAB_V153_PRUNE_RATIO": f"{prune_ratio:.2f}",
        "GRAB_V168_SVD_RANK": str(svd_rank),
        "GRAB_V130_COMPILE": "0",
        "GRAB_VERBOSE_INFER": "0",
    }
    module = load_version_module(env_overrides)
    loader, holdout_logids, labels, holdout_info = build_holdout(module, subset_paths)
    load_start = time.perf_counter()
    model, _ = module.load_model(ckpt_path=str(WEIGHTS_PATH), device="cpu")
    load_seconds = time.perf_counter() - load_start
    for moe in model.seq_encoder.moe:
        for attr in ("_v168_fc1_left", "_v168_fc1_right", "_v168_fc2_left", "_v168_fc2_right"):
            if hasattr(moe, attr):
                setattr(moe, attr, getattr(moe, attr).float())

    logit_map = {}
    batch_latencies = []
    with torch.inference_mode():
        for batch in loader:
            start = time.perf_counter()
            output = model(batch)
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.squeeze(-1).detach().cpu().numpy()
            batch_latencies.append(time.perf_counter() - start)
            batch_logids = batch["logid"].cpu().numpy()
            pred_mask = batch["pred_mask"].bool().cpu().numpy()
            for logid, logit, is_pred in zip(batch_logids, logits, pred_mask):
                if is_pred:
                    logit_map[int(logid)] = float(logit)

    ordered_logits = np.array([logit_map[x] for x in holdout_logids], dtype=np.float64)
    bias_results = evaluate_biases(ordered_logits, labels)
    best_bias = bias_results[0]

    result = {
        "config_name": config_name,
        "prune_ratio": prune_ratio,
        "svd_rank": svd_rank,
        "load_seconds": load_seconds,
        "latency_total_seconds": float(sum(batch_latencies)),
        "latency_mean_seconds": float(np.mean(batch_latencies)),
        "holdout": holdout_info,
        "bias_results": bias_results,
        "best_bias_result": best_bias,
    }

    del model
    del module
    gc.collect()
    return result


def choose_recommendation(stage1_results, stage2_results):
    baseline = next(item for item in stage1_results if abs(item["prune_ratio"] - 0.25) < 1e-9)
    baseline_auc = baseline["best_bias_result"]["auc"]
    candidates = stage2_results if stage2_results else stage1_results
    eligible = []
    for item in candidates:
        best = item["best_bias_result"]
        if 0.85 <= best["pcoc"] <= 1.15 and best["auc"] >= baseline_auc - 0.002:
            eligible.append(item)
    if eligible:
        eligible.sort(key=lambda x: (x["latency_total_seconds"], -x["best_bias_result"]["auc"]))
        return {
            "baseline": baseline["config_name"],
            "selection_rule": "fastest config with pcoc in [0.85,1.15] and auc within 0.002 of baseline",
            "winner": eligible[0],
        }
    fallback = sorted(candidates, key=lambda x: (abs(x["best_bias_result"]["pcoc"] - 1.0), x["latency_total_seconds"]))[0]
    return {
        "baseline": baseline["config_name"],
        "selection_rule": "fallback: closest pcoc to 1.0, then lower latency",
        "winner": fallback,
    }


def main():
    subset_paths = prepare_subset_files()
    stage1_results = []
    for ratio in STAGE1_RATIOS:
        name = f"prune{int(ratio * 100):02d}_rank128"
        print(f"[RUN] {name}")
        stage1_results.append(run_config(name, ratio, 128, subset_paths))

    stage1_results.sort(key=lambda x: x["latency_total_seconds"])
    best_stage1 = min(
        stage1_results,
        key=lambda x: (x["latency_total_seconds"], -x["best_bias_result"]["auc"]),
    )

    stage2_results = []
    for rank in STAGE2_RANKS:
        name = f"prune{int(best_stage1['prune_ratio'] * 100):02d}_rank{rank}"
        print(f"[RUN] {name}")
        stage2_results.append(run_config(name, best_stage1["prune_ratio"], rank, subset_paths))

    recommendation = choose_recommendation(stage1_results, stage2_results)
    output = {
        "dataset_subset": {
            "history_files": HISTORY_FILES,
            "lines_per_file": LINES_PER_FILE,
        },
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "recommendation": recommendation,
    }
    out_path = WORK_ROOT / "real_weight_holdout_search_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()
