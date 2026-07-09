#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path("/Users/sunyiyang/Desktop/Project/Baidu  GRAB")
INFER_PATH = ROOT / "submission" / "V201-TRT-MOE-UNQUANT-sxyq" / "infer.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("grab_v201", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def classify_linear(name: str) -> str:
    if name == "rep_encoder.linear":
        return "safe_ptq_candidate"
    if name == "linear":
        return "safe_ptq_candidate"
    if name.startswith("seq_encoder.qkv_proj."):
        return "safe_ptq_candidate"
    if name.startswith("seq_encoder.out_proj."):
        return "safe_ptq_candidate"
    if name.startswith("seq_encoder.ffn1."):
        return "safe_ptq_candidate"
    if name.startswith("seq_encoder.ffn2."):
        return "safe_ptq_candidate"
    if ".moe." in name and ".gate.w_g" in name:
        return "routing_sensitive_hold_fp16"
    if ".moe." in name and (".fc1" in name or ".fc2" in name):
        return "plugin_sensitive_hold_fp16"
    return "review_manually"


def main() -> None:
    module = load_module(INFER_PATH)
    with io.StringIO() as buf, redirect_stdout(buf):
        model, dev = module.load_model(device="cpu", ckpt_path=Path("/tmp/nonexistent_ckpt.pt"))
    del dev

    rows = []
    summary = {}
    for name, submodule in model.named_modules():
        if not isinstance(submodule, nn.Linear):
            continue
        bucket = classify_linear(name)
        params = sum(p.numel() for p in submodule.parameters())
        rows.append(
            {
                "name": name,
                "bucket": bucket,
                "in_features": int(submodule.in_features),
                "out_features": int(submodule.out_features),
                "params": int(params),
            }
        )
        summary.setdefault(bucket, {"modules": 0, "params": 0})
        summary[bucket]["modules"] += 1
        summary[bucket]["params"] += int(params)

    output = {
        "source": str(INFER_PATH),
        "policy": {
            "safe_ptq_candidate": "Prefer these for selective PTQ / Q-DQ exploration first",
            "routing_sensitive_hold_fp16": "Routing logits affect expert selection; hold FP16 until separately validated",
            "plugin_sensitive_hold_fp16": "MoE expert weights are packed into the TRT plugin path; hold FP16",
            "review_manually": "Unclassified linear, inspect before quantization",
        },
        "summary": summary,
        "modules": rows,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
