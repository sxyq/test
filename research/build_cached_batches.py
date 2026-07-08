from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_infer_module(infer_path: Path):
    spec = importlib.util.spec_from_file_location("grab_infer_module", infer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        default="submission/V184-STACKED-RECALIB-sxyq/infer.py",
        help="Reference infer.py used to build dataset objects",
    )
    parser.add_argument(
        "--dataset",
        default="research_datasets/2026_cti_data_meta/dataset",
        help="Dataset root containing history/, test.csv, label_data.txt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size used by infer.py",
    )
    parser.add_argument(
        "--batches-per-shard",
        type=int,
        default=64,
        help="How many DataLoader batches to store per shard",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    infer_path = (repo_root / args.infer).resolve()
    dataset_root = (repo_root / args.dataset).resolve()
    cache_dir = dataset_root / "cached_batches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading infer module from {infer_path}")
    infer_mod = load_infer_module(infer_path)

    history_dir = dataset_root / "history"
    input_file = dataset_root / "test.csv"
    history_files = sorted(history_dir.glob("*.csv")) if history_dir.exists() else []
    all_files = history_files + [input_file]

    print(f"[INFO] loading CSV sources: {len(history_files)} history files + test.csv")
    item_dict, user_seq = infer_mod.load_sample_files(sample_files_list=all_files)
    test_pred_logids = infer_mod.load_logids_from_file(input_file)
    max_feasign_per_slot = {1: 2}
    dataset = infer_mod.CTRUserDataset(
        item_dict,
        user_seq,
        max_feasign_per_slot=max_feasign_per_slot,
        pred_logids=test_pred_logids,
    )
    print(
        f"[INFO] dataset ready: num_users={dataset.num_users}, total_samples={dataset.total_samples}, "
        f"pred_logids={len(test_pred_logids)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=infer_mod.make_collate_fn(dataset.max_slot_id),
        pin_memory=False,
    )

    # Clear any previous shards so all version runs see the same cache set.
    for old in cache_dir.glob("shard_*.pt"):
        old.unlink()

    shard_batches = []
    batch_count = 0
    shard_idx = 0
    for batch in loader:
        shard_batches.append(batch)
        batch_count += 1
        if len(shard_batches) >= args.batches_per_shard:
            shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(shard_batches, shard_path)
            print(f"[INFO] wrote {shard_path.name} with {len(shard_batches)} batches (total={batch_count})")
            shard_batches = []
            shard_idx += 1

    if shard_batches:
        shard_path = cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_batches, shard_path)
        print(f"[INFO] wrote {shard_path.name} with {len(shard_batches)} batches (total={batch_count})")

    print(f"[INFO] done: {batch_count} batches written into {cache_dir}")


if __name__ == "__main__":
    main()
