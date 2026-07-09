#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import shutil
from pathlib import Path


TOOLS = ["trtexec", "nsys", "nvprof", "polygraphy"]
MODULES = ["torch", "onnx", "onnxruntime", "polygraphy", "numpy"]


def main() -> None:
    print("== Host ==")
    print(platform.platform())
    print(platform.machine())
    print()

    print("== CLI tools ==")
    for tool in TOOLS:
        print(f"{tool}: {shutil.which(tool) or 'MISSING'}")
    print()

    print("== Python modules ==")
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"{name}: {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")
    print()

    submit_infer = Path("/tmp/grab_submit1/infer.py")
    if submit_infer.exists():
        text = submit_infer.read_text()
        print("== submit(1) feature inventory ==")
        checks = {
            "custom_top2_kernel": "top2_softmax8" in text,
            "custom_embedding_kernels": "embedding_bag_16_fp16_strided" in text,
            "rep_direct_fill": "_forward_direct_fill" in text,
            "segmented_attention": "BAIDU_SEGMENTED_ATTENTION_DISABLE" in text,
            "varlen_flash_attention": "BAIDU_VARLEN_FLASH_ATTENTION_DISABLE" in text,
            "logit_bias": "BAIDU_LOGIT_BIAS" in text,
            "force_rebuild_batches": "BAIDU_FORCE_REBUILD_BATCHES" in text,
            "balanced_batching": "BAIDU_BALANCED_USER_BATCH" in text,
            "plugin_side_prune": "BAIDU_PRUNE_FFN" in text,
        }
        for key, value in checks.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
