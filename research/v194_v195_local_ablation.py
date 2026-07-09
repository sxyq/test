#!/usr/bin/env python3
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


REPO_ROOT = Path("/Users/sunyiyang/Desktop/Project/Baidu  GRAB")
SUBMISSION_ROOT = REPO_ROOT / "submission"
REAL_SAMPLE_ROOT = REPO_ROOT / "12_workspace" / "real_sample_128"
OUTPUT_JSON = REPO_ROOT / "12_workspace" / "v194_v195_local_ablation_results.json"

CONFIGS = [
    ("v194_default", "V194-TRT-MOE-STACKED-sxyq", {}),
    ("v195_default", "V195-TRT-MOE-HIGHRISK-sxyq", {}),
    ("v195_no_balance", "V195-TRT-MOE-HIGHRISK-sxyq", {"BAIDU_BALANCED_USER_BATCH": "0"}),
    ("v195_no_bias", "V195-TRT-MOE-HIGHRISK-sxyq", {"BAIDU_LOGIT_BIAS": "0.0"}),
    ("v195_no_prune", "V195-TRT-MOE-HIGHRISK-sxyq", {"BAIDU_PRUNE_FFN": "0"}),
]


def version_module_name(version_name, tag):
    return f"{version_name.replace('-', '_')}_{tag}"


def load_module(version_name, env_overrides, tag):
    infer_path = SUBMISSION_ROOT / version_name / "infer.py"
    old_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    try:
        spec = importlib.util.spec_from_file_location(version_module_name(version_name, tag), infer_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def batch_token_stats(batches):
    token_counts = [int(batch["user_offsets"][-1]) for batch in batches]
    pred_counts = [int(batch["pred_mask"].sum()) for batch in batches]
    users_per_batch = [int(batch["user_offsets"].numel() - 1) for batch in batches]
    return {
        "batch_count": len(batches),
        "token_sum": int(sum(token_counts)),
        "token_min": int(min(token_counts)) if token_counts else 0,
        "token_max": int(max(token_counts)) if token_counts else 0,
        "token_mean": float(statistics.mean(token_counts)) if token_counts else 0.0,
        "token_stdev": float(statistics.pstdev(token_counts)) if len(token_counts) > 1 else 0.0,
        "pred_sum": int(sum(pred_counts)),
        "pred_mean": float(statistics.mean(pred_counts)) if pred_counts else 0.0,
        "users_mean": float(statistics.mean(users_per_batch)) if users_per_batch else 0.0,
    }


def build_batches(module):
    history_files = sorted((REAL_SAMPLE_ROOT / "history").glob("*.csv"))
    all_files = history_files + [REAL_SAMPLE_ROOT / "test.csv"]

    t0 = time.perf_counter()
    item_dict, user_seq = module.load_sample_files(sample_files_list=all_files)
    t1 = time.perf_counter()
    pred_logids = module.load_logids_from_file(REAL_SAMPLE_ROOT / "test.csv")
    dataset = module.CTRUserDataset(
        item_dict=item_dict,
        user_seq=user_seq,
        max_feasign_per_slot={1: 2},
        pred_logids=pred_logids,
    )
    t2 = time.perf_counter()
    batch_size = getattr(module, "BAIDU_BATCH_USERS", 50)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=module.make_collate_fn(dataset.max_slot_id),
    )
    batches = [batch for batch in loader]
    t3 = time.perf_counter()

    if os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") == "1":
        grouped_batches = batches
        group_meta = {"mode": "disabled", "token_cap": 0}
    else:
        token_cap = int(os.environ.get("BAIDU_BATCH_GROUP_TOKEN_CAP", "300000"))
        grouped_batches = module.group_user_batches_by_token_cap(batches, token_cap)
        group_meta = {"mode": "token_cap", "token_cap": token_cap}
    t4 = time.perf_counter()

    return {
        "dataset_info": {
            "num_users": dataset.num_users,
            "total_samples": dataset.total_samples,
            "pred_samples": len(pred_logids),
            "max_sign_id": dataset.max_sign_id,
            "balanced_enabled": bool(getattr(module, "BAIDU_BALANCED_USER_BATCH", False)),
            "batch_size": batch_size,
        },
        "timing": {
            "load_csv_seconds": t1 - t0,
            "dataset_build_seconds": t2 - t1,
            "collate_seconds": t3 - t2,
            "group_seconds": t4 - t3,
        },
        "pre_group_stats": batch_token_stats(batches),
        "post_group_stats": batch_token_stats(grouped_batches),
        "group_meta": group_meta,
        "batches": grouped_batches,
    }


def benchmark_cached_load(batches):
    with tempfile.TemporaryDirectory() as tmpdir:
        shard = Path(tmpdir) / "shard_0000.pt"
        t0 = time.perf_counter()
        torch.save(batches, shard)
        t1 = time.perf_counter()
        reloaded = torch.load(shard, weights_only=False)
        t2 = time.perf_counter()
    return {
        "save_seconds": t1 - t0,
        "load_seconds": t2 - t1,
        "reloaded_batch_count": len(reloaded),
    }


def benchmark_seq_proxy(module, batches, device_name="cpu", max_batches=6):
    device = torch.device(device_name)
    torch.manual_seed(0)
    seq_encoder = module.TransformerEncoder(
        d_model=512,
        n_heads=8,
        num_layers=8,
        dim_ff=1024,
        act="relu",
    ).eval().to(device)

    times = []
    batch_details = []
    with torch.inference_mode():
        for batch in batches[:max_batches]:
            tokens = int(batch["user_offsets"][-1])
            users = int(batch["user_offsets"].numel() - 1)
            x = torch.randn(tokens, 512, device=device, dtype=torch.float32)
            extension = {
                "user_offsets": batch["user_offsets"],
                "user_offsets_cuda": batch.get("_user_offsets_cuda"),
                "max_user_len": batch.get("_max_user_len"),
            }
            seq_encoder(x, extension)
            start = time.perf_counter()
            seq_encoder(x, extension)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            batch_details.append({
                "tokens": tokens,
                "users": users,
                "seconds": elapsed,
            })
    return {
        "device": device_name,
        "bench_batches": len(batch_details),
        "total_seconds": float(sum(times)),
        "mean_seconds": float(statistics.mean(times)) if times else 0.0,
        "max_seconds": float(max(times)) if times else 0.0,
        "batch_details": batch_details,
    }


def analyze_results(results):
    by_name = {entry["name"]: entry for entry in results["configs"]}
    v194 = by_name["v194_default"]
    v195 = by_name["v195_default"]
    no_balance = by_name["v195_no_balance"]
    no_bias = by_name["v195_no_bias"]
    no_prune = by_name["v195_no_prune"]

    analysis = {
        "primary_online_symptom": (
            "V195 online score regression is dominated by calibration drift: "
            "AUC stayed near V194, but user-reported PCOC dropped from 1.05895 to 0.86655."
        ),
        "local_findings": [],
        "suspect_order": [],
    }

    def seq_total(entry):
        return entry["seq_proxy_cpu"]["total_seconds"]

    if abs(seq_total(v195) - seq_total(no_balance)) > 1e-9:
        if seq_total(v195) > seq_total(no_balance):
            analysis["local_findings"].append(
                "Balanced batching made the local seq-shape proxy slower than V195_no_balance."
            )
            analysis["suspect_order"].append("balanced_batching")
        else:
            analysis["local_findings"].append(
                "Balanced batching made the local seq-shape proxy faster than V195_no_balance."
            )
    else:
        analysis["local_findings"].append(
            "Balanced batching did not change the local seq-shape proxy on real_sample_128."
        )

    analysis["local_findings"].append(
        "BAIDU_LOGIT_BIAS only changes output calibration and should have near-zero latency impact locally."
    )
    analysis["local_findings"].append(
        "BAIDU_PRUNE_FFN is implemented inside prepare_trt_moe() and is CUDA/TRT-only in this package; "
        "Apple local lightweight validation cannot directly replay its real online contribution."
    )
    analysis["suspect_order"].extend([
        "logit_bias",
        "plugin_side_ffn_prune",
        "force_rebuild_batches_wallclock",
    ])
    return analysis


def main():
    results = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "sample_root": str(REAL_SAMPLE_ROOT),
        "configs": [],
    }

    for name, version_name, env_overrides in CONFIGS:
        module = load_module(version_name, env_overrides, name)
        built = build_batches(module)
        cache_probe = benchmark_cached_load(built["batches"])
        seq_proxy_cpu = benchmark_seq_proxy(module, built["batches"], "cpu")
        entry = {
            "name": name,
            "version": version_name,
            "env_overrides": env_overrides,
            "dataset_info": built["dataset_info"],
            "timing": built["timing"],
            "pre_group_stats": built["pre_group_stats"],
            "post_group_stats": built["post_group_stats"],
            "group_meta": built["group_meta"],
            "cache_probe": cache_probe,
            "seq_proxy_cpu": seq_proxy_cpu,
            "plugin_prune_active_locally": False,
        }
        if torch.backends.mps.is_available():
            entry["seq_proxy_mps"] = benchmark_seq_proxy(module, built["batches"], "mps")
        results["configs"].append(entry)

    results["analysis"] = analyze_results(results)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
