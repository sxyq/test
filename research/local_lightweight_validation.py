#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path("/Users/sunyiyang/Desktop/Project/Baidu  GRAB")
SUBMISSION_ROOT = REPO_ROOT / "submission"
VERSIONS = [
    "V169-HYBRID-SVD-sxyq",
    "V175-HEAD-PRUNE-FIX-sxyq",
    "V176-FFN-PRUNE-sxyq",
    "V177-FFN-NOBIAS-sxyq",
    "V178-FFN-SVD32-sxyq",
    "V179-PURE-ENGINEERING-sxyq",
    "V180-MAXAUTOTUNE-sxyq",
    "V181-FFN-PRUNE50-sxyq",
    "V182-FFN25-SVD64FP32-sxyq",
]
CPU_TOKENS = [256, 512, 1024]
MPS_TOKENS = [256, 512]
TOP_MPS_VERSIONS = 4
TARGETED_VARIANTS = [
    ("V176-default", "V176-FFN-PRUNE-sxyq", {}),
    ("V176-nosvd", "V176-FFN-PRUNE-sxyq", {"GRAB_V168_SVD_LOWRANK": "0"}),
    (
        "V176-prune50-nosvd",
        "V176-FFN-PRUNE-sxyq",
        {"GRAB_V168_SVD_LOWRANK": "0", "GRAB_V153_PRUNE_RATIO": "0.50"},
    ),
    ("V177-default", "V177-FFN-NOBIAS-sxyq", {}),
    ("V181-default", "V181-FFN-PRUNE50-sxyq", {}),
    ("V181-nosvd", "V181-FFN-PRUNE50-sxyq", {"GRAB_V168_SVD_LOWRANK": "0"}),
    ("V182-default", "V182-FFN25-SVD64FP32-sxyq", {}),
    ("V182-rank128", "V182-FFN25-SVD64FP32-sxyq", {"GRAB_V168_SVD_RANK": "128"}),
]
REAL_SAMPLE_VERSIONS = [
    "V176-FFN-PRUNE-sxyq",
    "V177-FFN-NOBIAS-sxyq",
    "V181-FFN-PRUNE50-sxyq",
    "V182-FFN25-SVD64FP32-sxyq",
]


def version_number(version_name):
    return int(version_name.split("-")[0][1:])


def load_version_module(version_name, env_overrides=None, module_tag=None):
    env_overrides = env_overrides or {}
    infer_path = SUBMISSION_ROOT / version_name / "infer.py"
    module_name = version_name.replace("-", "_")
    if module_tag:
        module_name = f"{module_name}_{module_tag}"
    old_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    try:
        spec = importlib.util.spec_from_file_location(module_name, infer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_fake_batch(batch_size=64, slot_num=28, vocab_size=4096):
    batch = {
        "user_offsets": torch.tensor([0, batch_size // 2, batch_size], dtype=torch.long),
        "pred_mask": torch.tensor([(i % 2) == 0 for i in range(batch_size)], dtype=torch.bool),
        "logid": torch.arange(batch_size, dtype=torch.long),
    }
    for slot_idx in range(slot_num):
        values = []
        offsets = [0]
        for row_idx in range(batch_size):
            width = ((slot_idx + row_idx) % 3) + 1
            for inner_idx in range(width):
                values.append((slot_idx * 97 + row_idx * 13 + inner_idx * 7) % vocab_size)
            offsets.append(offsets[-1] + width)
        batch[slot_idx + 1] = (
            torch.tensor(values, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long),
        )
    return batch


def build_user_offsets(total_tokens, users):
    users = max(1, min(users, total_tokens))
    base = total_tokens // users
    rem = total_tokens % users
    offsets = [0]
    for idx in range(users):
        seg = base + (1 if idx < rem else 0)
        offsets.append(offsets[-1] + seg)
    return torch.tensor(offsets, dtype=torch.long)


def apply_v175_head_prune(seq_encoder):
    old_n_heads = seq_encoder.n_heads
    head_dim = seq_encoder.head_dim
    d_model = seq_encoder.d_model
    new_n_heads = int(old_n_heads * (1.0 - 0.25))
    with torch.no_grad():
        for layer_idx in range(seq_encoder.num_layers):
            qkv = seq_encoder.qkv_proj[layer_idx]
            out_proj = seq_encoder.out_proj[layer_idx]
            qkv_w = qkv.weight.data.contiguous().view(old_n_heads, 3 * head_dim, d_model)
            out_w = out_proj.weight.data.contiguous().view(d_model, old_n_heads, head_dim)
            importance = qkv_w.norm(dim=(1, 2)) * out_w.norm(dim=(0, 2))
            keep_heads = importance.topk(new_n_heads).indices.sort().values
            qkv.weight.data = qkv_w[keep_heads].contiguous().view(-1, d_model)
            qkv.bias.data = (
                qkv.bias.data.contiguous().view(old_n_heads, 3 * head_dim)[keep_heads].contiguous().view(-1)
            )
            out_proj.weight.data = out_w[:, keep_heads, :].contiguous().view(d_model, -1)
        seq_encoder.n_heads = new_n_heads


def prepare_seq_encoder(version_name, module, device):
    seq_encoder = module.TransformerEncoder(
        d_model=512,
        n_heads=8,
        num_layers=8,
        dim_ff=1024,
        act="relu",
    ).eval().to(device)
    # Match the submission hot path more closely on accelerator devices:
    # the real CUDA path applies FP16 transforms before the dense SMoE/SVD
    # caches are built, so MPS proxy runs should do the same.
    if device.type == "mps":
        seq_encoder = seq_encoder.half()
    if version_name.startswith("V175-"):
        apply_v175_head_prune(seq_encoder)
    prep_start = time.perf_counter()
    if getattr(module, "V109_DENSE_SMOE", False):
        with contextlib.redirect_stdout(io.StringIO()):
            for moe in seq_encoder.moe:
                moe._v109_prepare_dense_weights()
    prep_seconds = time.perf_counter() - prep_start
    return seq_encoder, prep_seconds


def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_seq_encoder(seq_encoder, device, total_tokens, users, warmup_iters, bench_iters):
    model_dtype = next(seq_encoder.parameters()).dtype
    x = torch.randn(total_tokens, 512, device=device, dtype=model_dtype)
    user_offsets = build_user_offsets(total_tokens, users)
    with torch.inference_mode():
        for _ in range(warmup_iters):
            seq_encoder(x, user_offsets_cpu=user_offsets)
            sync_device(device)
    samples = []
    with torch.inference_mode():
        for _ in range(bench_iters):
            start = time.perf_counter()
            seq_encoder(x, user_offsets_cpu=user_offsets)
            sync_device(device)
            samples.append(time.perf_counter() - start)
    return statistics.mean(samples)


def smoke_rep_encoder(module):
    batch = build_fake_batch(batch_size=8)
    rep_encoder = module.RepEncoder(
        vocab_size=4096,
        emb_dim=512,
        padding_idx=0,
        slot_num=28,
        d_model=512,
    ).eval()
    result = {"cpu": None, "mps": None}
    cpu_out = rep_encoder(batch)
    result["cpu"] = {"ok": True, "shape": list(cpu_out.shape)}
    if torch.backends.mps.is_available():
        rep_mps = rep_encoder.to("mps")
        batch_mps = {}
        for key, value in batch.items():
            if isinstance(value, tuple):
                batch_mps[key] = (value[0].to("mps"), value[1].to("mps"))
            else:
                batch_mps[key] = value.to("mps") if torch.is_tensor(value) else value
        try:
            rep_mps(batch_mps)
            result["mps"] = {"ok": True}
        except Exception as exc:
            result["mps"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    return result


def benchmark_split_device(version_name, module, seq_device_name):
    batch = build_fake_batch(batch_size=128)
    rep_encoder = module.RepEncoder(
        vocab_size=4096,
        emb_dim=512,
        padding_idx=0,
        slot_num=28,
        d_model=512,
    ).eval()
    seq_device = torch.device(seq_device_name)
    seq_encoder, prep_seconds = prepare_seq_encoder(version_name, module, seq_device)
    linear = nn.Linear(512, 1).eval().to(seq_device)
    seq_dtype = next(seq_encoder.parameters()).dtype
    cpu_times = []
    seq_times = []
    total_times = []
    with torch.inference_mode():
        for _ in range(1):
            seq_input = rep_encoder(batch)
            moved = seq_input.to(seq_device, dtype=seq_dtype)
            user_offsets = batch["user_offsets"]
            seq_output, _ = seq_encoder(moved, user_offsets_cpu=user_offsets)
            logits = linear(seq_output.reshape(-1, 512).float())
            _ = torch.sigmoid(logits)
            sync_device(seq_device)
        for _ in range(3):
            t0 = time.perf_counter()
            seq_input = rep_encoder(batch)
            t1 = time.perf_counter()
            moved = seq_input.to(seq_device, dtype=seq_dtype)
            user_offsets = batch["user_offsets"]
            seq_output, _ = seq_encoder(moved, user_offsets_cpu=user_offsets)
            logits = linear(seq_output.reshape(-1, 512).float())
            _ = torch.sigmoid(logits)
            sync_device(seq_device)
            t2 = time.perf_counter()
            cpu_times.append(t1 - t0)
            seq_times.append(t2 - t1)
            total_times.append(t2 - t0)
    return {
        "prep_seconds": prep_seconds,
        "cpu_rep_seconds": statistics.mean(cpu_times),
        "seq_and_head_seconds": statistics.mean(seq_times),
        "total_seconds": statistics.mean(total_times),
    }


def build_real_sample_batch(reference_module):
    sample_root = REPO_ROOT / "12_workspace" / "real_sample"
    files = [
        sample_root / "history_train_0000.csv",
        sample_root / "history_train_0001.csv",
        sample_root / "test.csv",
    ]
    if not all(path.exists() for path in files):
        return None
    item_dict, user_seq = reference_module.load_sample_files(files)
    pred_logids = reference_module.load_logids_from_file(sample_root / "test.csv")
    dataset = reference_module.CTRUserDataset(
        item_dict,
        user_seq,
        max_feasign_per_slot={1: 2},
        pred_logids=pred_logids,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=reference_module.make_collate_fn(dataset.max_slot_id),
    )
    batch = next(iter(loader))
    user_lengths = (batch["user_offsets"][1:] - batch["user_offsets"][:-1]).tolist()
    info = {
        "num_users": dataset.num_users,
        "total_samples": dataset.total_samples,
        "batch_samples": int(batch["logid"].numel()),
        "pred_count": int(batch["pred_mask"].sum()),
        "avg_user_len": statistics.mean(user_lengths),
        "max_user_len": max(user_lengths),
    }
    return batch, info


def benchmark_real_sample_split(version_name, module, real_batch, seq_device_name):
    small_batch = {}
    for key, value in real_batch.items():
        if isinstance(key, int):
            vals, offsets = value
            small_batch[key] = (vals.remainder(4096), offsets)
        else:
            small_batch[key] = value
    rep_encoder = module.RepEncoder(
        vocab_size=4096,
        emb_dim=512,
        padding_idx=0,
        slot_num=28,
        d_model=512,
    ).eval()
    with torch.inference_mode():
        rep_start = time.perf_counter()
        seq_input = rep_encoder(small_batch)
        rep_seconds = time.perf_counter() - rep_start
    seq_device = torch.device(seq_device_name)
    seq_encoder, prep_seconds = prepare_seq_encoder(version_name, module, seq_device)
    linear = nn.Linear(512, 1).eval().to(seq_device)
    seq_dtype = next(seq_encoder.parameters()).dtype
    moved = seq_input.to(seq_device, dtype=seq_dtype)
    samples = []
    with torch.inference_mode():
        seq_encoder(moved, user_offsets_cpu=small_batch["user_offsets"])
        sync_device(seq_device)
        for _ in range(3):
            start = time.perf_counter()
            encoded, _ = seq_encoder(moved, user_offsets_cpu=small_batch["user_offsets"])
            pred = torch.sigmoid(linear(encoded.reshape(-1, 512).float()))
            _ = pred
            sync_device(seq_device)
            samples.append(time.perf_counter() - start)
    return {
        "prep_seconds": prep_seconds,
        "rep_cpu_seconds": rep_seconds,
        "seq_and_head_seconds": statistics.mean(samples),
        "seq_input_shape": list(seq_input.shape),
    }


def dataset_probe():
    data_dir = REPO_ROOT / "research_datasets" / "2026_cti_data_meta"
    history_file = data_dir / "dataset" / "history" / "train_0000.csv"
    probe = {
        "meta_dir_exists": data_dir.exists(),
        "history_file_exists": history_file.exists(),
        "readme_exists": (data_dir / "README.md").exists(),
    }
    if history_file.exists():
        lines = []
        with history_file.open("r", errors="ignore") as handle:
            for _ in range(3):
                line = handle.readline().strip()
                if line:
                    lines.append(line[:280])
        probe["history_head"] = lines
        probe["history_file_size_bytes"] = history_file.stat().st_size
    return probe


def stable_average(entry, token_list):
    values = []
    for token in token_list:
        bench = entry["benchmarks"].get(str(token))
        if bench and bench.get("ok"):
            values.append(bench["seconds"])
    if not values:
        return None
    return statistics.mean(values)


def run_targeted_variant(label, version_name, env_overrides, device_name):
    module = load_version_module(version_name, env_overrides=env_overrides, module_tag=label.replace("-", "_"))
    device = torch.device(device_name)
    seq_encoder, prep_seconds = prepare_seq_encoder(version_name, module, device)
    result = {
        "version": version_name,
        "env_overrides": env_overrides,
        "prep_seconds": prep_seconds,
        "benchmarks": {},
    }
    if device.type == "cpu":
        token_list = [512, 1024]
        bench_iters = 2
    else:
        token_list = [512]
        bench_iters = 2
    for total_tokens in token_list:
        seconds = benchmark_seq_encoder(
            seq_encoder,
            device,
            total_tokens=total_tokens,
            users=max(16, total_tokens // 8),
            warmup_iters=1,
            bench_iters=bench_iters,
        )
        result["benchmarks"][str(total_tokens)] = seconds
    return result


def main():
    results = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset_probe": dataset_probe(),
        "rep_encoder_smoke": {},
        "seq_bench_cpu": {},
        "seq_bench_mps": {},
        "split_device": {},
        "real_sample": {},
        "targeted_variants": {"cpu": {}, "mps": {}},
        "recommendation": {},
    }

    reference_module = load_version_module("V176-FFN-PRUNE-sxyq")
    results["rep_encoder_smoke"] = smoke_rep_encoder(reference_module)

    cpu_summary = {}
    for version_name in VERSIONS:
        module = load_version_module(version_name)
        entry = {"prep_seconds": None, "benchmarks": {}}
        try:
            seq_encoder, prep_seconds = prepare_seq_encoder(version_name, module, torch.device("cpu"))
            entry["prep_seconds"] = prep_seconds
            for total_tokens in CPU_TOKENS:
                try:
                    seconds = benchmark_seq_encoder(
                        seq_encoder,
                        torch.device("cpu"),
                        total_tokens=total_tokens,
                        users=max(16, total_tokens // 8),
                        warmup_iters=1,
                        bench_iters=3,
                    )
                    entry["benchmarks"][str(total_tokens)] = {"ok": True, "seconds": seconds}
                except Exception as exc:
                    entry["benchmarks"][str(total_tokens)] = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
        except Exception as exc:
            entry["error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        avg = stable_average(entry, CPU_TOKENS)
        if avg is not None:
            entry["avg_seconds"] = avg
            cpu_summary[version_name] = avg
        results["seq_bench_cpu"][version_name] = entry

    stable_versions = [name for name, _ in sorted(cpu_summary.items(), key=lambda item: item[1])]
    mps_versions = stable_versions[:TOP_MPS_VERSIONS]
    if torch.backends.mps.is_available():
        for version_name in mps_versions:
            module = load_version_module(version_name)
            entry = {"prep_seconds": None, "benchmarks": {}}
            try:
                seq_encoder, prep_seconds = prepare_seq_encoder(version_name, module, torch.device("mps"))
                entry["prep_seconds"] = prep_seconds
                for total_tokens in MPS_TOKENS:
                    try:
                        seconds = benchmark_seq_encoder(
                            seq_encoder,
                            torch.device("mps"),
                            total_tokens=total_tokens,
                            users=max(16, total_tokens // 8),
                            warmup_iters=1,
                            bench_iters=2,
                        )
                        entry["benchmarks"][str(total_tokens)] = {"ok": True, "seconds": seconds}
                    except Exception as exc:
                        entry["benchmarks"][str(total_tokens)] = {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
            except Exception as exc:
                entry["error"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            avg = stable_average(entry, MPS_TOKENS)
            if avg is not None:
                entry["avg_seconds"] = avg
            results["seq_bench_mps"][version_name] = entry

        for version_name in mps_versions[:3]:
            module = load_version_module(version_name)
            try:
                results["split_device"][version_name] = benchmark_split_device(version_name, module, "mps")
            except Exception as exc:
                results["split_device"][version_name] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }

    real_sample = build_real_sample_batch(reference_module)
    if real_sample is not None:
        real_batch, real_info = real_sample
        results["real_sample"]["dataset_info"] = real_info
        if torch.backends.mps.is_available():
            for version_name in REAL_SAMPLE_VERSIONS:
                module = load_version_module(version_name)
                try:
                    results["real_sample"][version_name] = benchmark_real_sample_split(
                        version_name,
                        module,
                        real_batch,
                        "mps",
                    )
                except Exception as exc:
                    results["real_sample"][version_name] = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }

    for label, version_name, env_overrides in TARGETED_VARIANTS:
        try:
            results["targeted_variants"]["cpu"][label] = run_targeted_variant(
                label,
                version_name,
                env_overrides,
                "cpu",
            )
        except Exception as exc:
            results["targeted_variants"]["cpu"][label] = {
                "version": version_name,
                "env_overrides": env_overrides,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        if torch.backends.mps.is_available():
            try:
                results["targeted_variants"]["mps"][label] = run_targeted_variant(
                    label,
                    version_name,
                    env_overrides,
                    "mps",
                )
            except Exception as exc:
                results["targeted_variants"]["mps"][label] = {
                    "version": version_name,
                    "env_overrides": env_overrides,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }

    stable_cpu_versions = {
        version_name: avg for version_name, avg in cpu_summary.items() if results["seq_bench_cpu"][version_name]["benchmarks"]
    }
    range_175_180 = {
        version_name: avg
        for version_name, avg in stable_cpu_versions.items()
        if 175 <= version_number(version_name) <= 180
    }
    outside_175_180 = {
        version_name: avg
        for version_name, avg in stable_cpu_versions.items()
        if version_number(version_name) < 175 or version_number(version_name) > 180
    }
    if range_175_180 and outside_175_180:
        best_inside_name, best_inside_avg = min(range_175_180.items(), key=lambda item: item[1])
        best_outside_name, best_outside_avg = min(outside_175_180.items(), key=lambda item: item[1])
        results["recommendation"] = {
            "best_inside_175_180": {
                "version": best_inside_name,
                "avg_cpu_seconds": best_inside_avg,
            },
            "best_outside_175_180": {
                "version": best_outside_name,
                "avg_cpu_seconds": best_outside_avg,
            },
            "outside_beats_inside": best_outside_avg < best_inside_avg,
            "delta_seconds": best_inside_avg - best_outside_avg,
        }

    targeted_cpu = {}
    for label, entry in results["targeted_variants"]["cpu"].items():
        benches = entry.get("benchmarks", {})
        if benches:
            targeted_cpu[label] = statistics.mean(benches.values())
    if targeted_cpu:
        best_label, best_avg = min(targeted_cpu.items(), key=lambda item: item[1])
        results["recommendation"]["best_targeted_cpu_variant"] = {
            "label": best_label,
            "avg_seconds": best_avg,
        }

    targeted_mps = {}
    for label, entry in results["targeted_variants"]["mps"].items():
        benches = entry.get("benchmarks", {})
        if benches and "512" in benches:
            targeted_mps[label] = benches["512"]
    if targeted_mps:
        best_label, best_seconds = min(targeted_mps.items(), key=lambda item: item[1])
        results["recommendation"]["best_targeted_mps_512_variant"] = {
            "label": best_label,
            "seconds": best_seconds,
        }

    real_sample_entries = {}
    for version_name, entry in results["real_sample"].items():
        if version_name == "dataset_info":
            continue
        if "seq_and_head_seconds" in entry:
            real_sample_entries[version_name] = entry["seq_and_head_seconds"]
    if real_sample_entries:
        best_version, best_seconds = min(real_sample_entries.items(), key=lambda item: item[1])
        results["recommendation"]["best_real_sample_split_mps"] = {
            "version": best_version,
            "seconds": best_seconds,
        }

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
