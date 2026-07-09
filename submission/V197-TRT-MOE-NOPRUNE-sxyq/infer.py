import sys
import os
import ctypes
import heapq

# 获取当前环境脚本所在目录或指定绝对路径
if os.path.exists("../libraries"):
    lib_path = os.path.abspath("../libraries")
    sys.path.append(lib_path)

import math
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REP_COUNT_MATMUL_SLOTS = (15, 17, 19, 20, 21, 24, 26)
BAIDU_CPU_METADATA = os.environ.get("BAIDU_CPU_METADATA", "1") != "0"
BAIDU_CHUNKED_COLLECT = os.environ.get("BAIDU_CHUNKED_COLLECT", "1") != "0"
BAIDU_COLLECT_CHUNK_PREDS = int(os.environ.get("BAIDU_COLLECT_CHUNK_PREDS", "65536"))
BAIDU_ADAPT_PRED_INDICES = os.environ.get("BAIDU_ADAPT_PRED_INDICES", "1") != "0"
BAIDU_PRED_INDICES_MAX_DENSITY = float(os.environ.get("BAIDU_PRED_INDICES_MAX_DENSITY", "0.125"))
BAIDU_SILENT_RUNNER = os.environ.get("BAIDU_SILENT_RUNNER", "1") != "0"
BAIDU_LOGIT_BIAS = float(os.environ.get("BAIDU_LOGIT_BIAS", "-0.20734744717444656"))
BAIDU_BATCH_USERS = max(1, int(os.environ.get("BAIDU_BATCH_USERS", "50")))
BAIDU_BALANCED_USER_BATCH = os.environ.get("BAIDU_BALANCED_USER_BATCH", "1") != "0"
BAIDU_LOG_BATCH_STATS = os.environ.get("BAIDU_LOG_BATCH_STATS", "0") != "0"
BAIDU_PRUNE_FFN = os.environ.get("BAIDU_PRUNE_FFN", "0") != "0"
BAIDU_PRUNE_RATIO = float(os.environ.get("BAIDU_PRUNE_RATIO", "0.25"))
BAIDU_FORCE_REBUILD_BATCHES = os.environ.get("BAIDU_FORCE_REBUILD_BATCHES", "1") != "0"

if os.environ.get("BAIDU_TF32_DISABLE", "0") != "1":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _progress(iterable, **kwargs):
    if BAIDU_SILENT_RUNNER:
        return iterable
    return tqdm(iterable, **kwargs)


class _BaiduMoeTop2Launcher:
    _instance = None

    def __init__(self):
        cur_path = Path(__file__).parent.absolute()
        if "BAIDU_TRT_MOE_PLUGIN" in os.environ:
            so_candidates = [Path(os.environ["BAIDU_TRT_MOE_PLUGIN"])]
        else:
            so_candidates = [
                cur_path / "libbaidu_moe_top2_plugin.so",
                cur_path / "trt_moe_plugin" / "build" / "libbaidu_moe_top2_plugin.so",
            ]
        so_path = next((path for path in so_candidates if path.exists()), so_candidates[0])
        if not so_path.exists():
            raise FileNotFoundError(f"TensorRT MoE plugin not found: {so_path}")

        trt_lib = cur_path / ".tools" / "tensorrt-10.7" / "usr" / "lib" / "x86_64-linux-gnu"
        nvinfer_candidates = [
            Path(os.environ["BAIDU_TRT_NVINFER_LIB"]) if "BAIDU_TRT_NVINFER_LIB" in os.environ else None,
            cur_path / "libnvinfer.so.10",
            trt_lib / "libnvinfer.so.10",
        ]
        for nvinfer in nvinfer_candidates:
            if nvinfer is not None and nvinfer.exists():
                ctypes.CDLL(str(nvinfer), mode=ctypes.RTLD_GLOBAL)
                break

        self._lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
        self._workspace_size_fp32 = self._lib.baidu_moe_top2_fp32_workspace_size
        self._workspace_size_fp32.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        self._workspace_size_fp32.restype = ctypes.c_size_t
        self._workspace_size_fp16 = self._lib.baidu_moe_top2_fp16_workspace_size
        self._workspace_size_fp16.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        self._workspace_size_fp16.restype = ctypes.c_size_t

        self._fn_fp32 = self._lib.baidu_moe_top2_fp32_launch
        self._fn_fp32.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.c_uint64,
        ]
        self._fn_fp32.restype = ctypes.c_int
        self._fn_fp16 = self._lib.baidu_moe_top2_fp16_launch
        self._fn_fp16.argtypes = self._fn_fp32.argtypes
        self._fn_fp16.restype = ctypes.c_int

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def workspace_size(self, token_count, d_model, hidden_dim, num_experts, top_k, dtype=torch.float32):
        if dtype == torch.float16:
            return self._workspace_size_fp16(token_count, d_model, hidden_dim, num_experts, top_k)
        return self._workspace_size_fp32(token_count, d_model, hidden_dim, num_experts, top_k)

    def launch(self, x, topk_idx, topk_score, fc1_weight, fc1_bias, fc2_weight, fc2_bias, workspace, output):
        if topk_idx.dtype == torch.int32:
            index_type = 3
        elif topk_idx.dtype == torch.int64:
            index_type = 8
        else:
            raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")

        token_count = x.shape[0]
        d_model = x.shape[1]
        hidden_dim = fc1_bias.shape[1]
        num_experts = fc1_weight.shape[0]
        top_k = topk_idx.shape[1]
        stream = torch.cuda.current_stream(x.device).cuda_stream

        if x.dtype == torch.float16:
            fn = self._fn_fp16
        elif x.dtype == torch.float32:
            fn = self._fn_fp32
        else:
            raise TypeError(f"x must be float16 or float32, got {x.dtype}")

        status = fn(
            x.data_ptr(), topk_idx.data_ptr(), topk_score.data_ptr(),
            fc1_weight.data_ptr(), fc1_bias.data_ptr(), fc2_weight.data_ptr(), fc2_bias.data_ptr(),
            workspace.data_ptr(), output.data_ptr(),
            index_type, token_count, d_model, hidden_dim, num_experts, top_k,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"baidu_moe_top2_fp32_launch failed with status={status}")


_TRT_MOE_WORKSPACE_CACHE = {}


def _trt_moe_workspace_key(device, dtype, d_model, hidden_dim, num_experts, top_k):
    dev = torch.device(device)
    index = dev.index
    if index is None and dev.type == "cuda":
        index = torch.cuda.current_device()
    return (dev.type, index, dtype, d_model, hidden_dim, num_experts, top_k)


def _reserve_trt_moe_workspace(
    token_count,
    device,
    dtype,
    d_model=512,
    hidden_dim=1024,
    num_experts=8,
    top_k=2,
):
    if (
        token_count <= 0
        or torch.device(device).type != "cuda"
        or os.environ.get("BAIDU_TRT_MOE_DISABLE", "0") == "1"
        or dtype not in (torch.float16, torch.float32)
    ):
        return None

    try:
        launcher = _BaiduMoeTop2Launcher.get()
        workspace_bytes = launcher.workspace_size(
            int(token_count), d_model, hidden_dim, num_experts, top_k, dtype,
        )
    except Exception:
        return None

    key = _trt_moe_workspace_key(device, dtype, d_model, hidden_dim, num_experts, top_k)
    workspace = _TRT_MOE_WORKSPACE_CACHE.get(key)
    if workspace is None or workspace.numel() < workspace_bytes:
        workspace = torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
        _TRT_MOE_WORKSPACE_CACHE[key] = workspace
    return workspace


def _default_seq_encoder_dtype(device):
    dev = torch.device(device)
    seq_fp16_env = os.environ.get("BAIDU_SEQ_ENCODER_FP16")
    if seq_fp16_env is None:
        use_fp16 = dev.type == "cuda" and os.environ.get("BAIDU_SEQ_ENCODER_FP16_DISABLE", "0") != "1"
    else:
        use_fp16 = dev.type == "cuda" and seq_fp16_env == "1"
    return torch.float16 if use_fp16 else torch.float32


def _rep_count_matmul_limits():
    max_unique = int(os.environ.get("BAIDU_REP_COUNT_MATMUL_MAX_UNIQUE", "1024"))
    max_elements = int(os.environ.get("BAIDU_REP_COUNT_MATMUL_MAX_ELEMENTS", "320000000"))
    return max_unique, max_elements


# ============================================================
# 数据加载（来自 train/dataset.py）
# ============================================================

def _detect_has_clk(file_path):
    """检测 CSV 文件是否包含 clk 列（5列 vs 4列格式）。
    5列格式: logid,userid,adid,clk,timestamp,sign:slot...
    4列格式: logid,userid,adid,timestamp,sign:slot...
    通过第5个字段是否包含 ':' 来判断：有 ':' 说明已经是 sign:slot，即无 clk 列。
    """
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                return ':' not in parts[4]
            return False
    return False


def load_sample_files(sample_files_list):
    """加载 CSV sample 文件，返回 item_dict 和 user_seq。
    自动检测每个文件是 5列（含clk）还是 4列（无clk）格式。
    """
    sample_files = sorted([Path(f) for f in sample_files_list])
    print(f'[INFO] loading {len(sample_files)} files: {[str(f) for f in sample_files]}')

    item_dict = {}
    user_logs = defaultdict(list)

    for sample_file in _progress(sample_files, desc='Loading sample files'):
        has_clk = _detect_has_clk(sample_file)
        min_parts = 5 if has_clk else 4
        print(f'  {sample_file.name}: has_clk={has_clk}')

        with open(sample_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < min_parts:
                    continue

                logid = int(parts[0])
                userid = int(parts[1])
                adid = int(parts[2])

                if has_clk:
                    clk = int(parts[3])
                    timestamp = int(parts[4])
                    feat_start = 5
                else:
                    clk = 0
                    timestamp = int(parts[3])
                    feat_start = 4

                signs = []
                slots = []
                for pair in parts[feat_start:]:
                    if ':' in pair:
                        s, sl = pair.split(':', 1)
                        signs.append(int(s))
                        slots.append(int(sl))

                item_dict[logid] = {
                    'logid': logid,
                    'userid': userid,
                    'adid': adid,
                    'clk': clk,
                    'timestamp': timestamp,
                    'signs': np.array(signs, dtype=np.int64),
                    'slots': np.array(slots, dtype=np.int64),
                }
                user_logs[userid].append((timestamp, logid))

    user_seq = {}
    for userid, logs in user_logs.items():
        logs.sort(key=lambda x: x[0])
        user_seq[userid] = [logid for _, logid in logs]

    print(f'[INFO] loaded {len(item_dict)} records, {len(user_seq)} users')
    return item_dict, user_seq


def load_logids_from_file(file_path):
    """快速读取一个 sample 文件中的所有 logid"""
    logids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.index(',')
            logids.add(int(line[:comma]))
    return logids


def build_balanced_user_order(user_items, batch_users):
    user_ids = sorted(user_items.keys())
    if not user_ids:
        return [], []

    lengths = {userid: len(user_items[userid]) for userid in user_ids}
    num_batches = (len(user_ids) + batch_users - 1) // batch_users
    batches = [[] for _ in range(num_batches)]
    heap = [(0, batch_idx, 0) for batch_idx in range(num_batches)]

    for userid, seq_len in sorted(lengths.items(), key=lambda item: (-item[1], item[0])):
        skipped = []
        while heap:
            total_tokens, batch_idx, count = heapq.heappop(heap)
            if count < batch_users:
                batches[batch_idx].append(userid)
                heapq.heappush(heap, (total_tokens + seq_len, batch_idx, count + 1))
                break
            skipped.append((total_tokens, batch_idx, count))
        for item in skipped:
            heapq.heappush(heap, item)

    batch_stats = []
    ordered_user_ids = []
    for batch_idx, batch_user_ids in enumerate(batches):
        if not batch_user_ids:
            continue
        token_total = sum(lengths[userid] for userid in batch_user_ids)
        batch_stats.append({
            "batch_idx": batch_idx,
            "users": len(batch_user_ids),
            "token_total": token_total,
            "min_len": min(lengths[userid] for userid in batch_user_ids),
            "max_len": max(lengths[userid] for userid in batch_user_ids),
        })
        ordered_user_ids.extend(batch_user_ids)
    return ordered_user_ids, batch_stats


class CTRUserDataset(Dataset):
    """按用户组织的 CTR 数据集"""

    def __init__(self, item_dict, user_seq=None, max_feasign_per_slot=None, pred_logids=None):
        super().__init__()
        self.item_dict = item_dict
        self.user_seq = user_seq if user_seq else {}
        self.max_feasign_per_slot = max_feasign_per_slot
        self.pred_logids = pred_logids if pred_logids is not None else set()

        self.user_items = defaultdict(list)
        for logid, rec in item_dict.items():
            userid = rec['userid']
            feasign = defaultdict(list)
            for slot, sign in zip(rec['slots'].tolist(), rec['signs'].tolist()):
                feasign[slot].append(sign)
            if max_feasign_per_slot is not None:
                feasign = {slot: signs[:max_feasign_per_slot[slot]]
                           if max_feasign_per_slot.get(slot, -1) != -1 else signs
                           for slot, signs in feasign.items()}
            feasign = dict(feasign)
            label = rec['clk']
            self.user_items[userid].append((logid, feasign, label))

        if BAIDU_BALANCED_USER_BATCH:
            self.user_ids, self.batch_stats = build_balanced_user_order(self.user_items, BAIDU_BATCH_USERS)
        else:
            self.user_ids = sorted(self.user_items.keys())
            self.batch_stats = []
        self.num_users = len(self.user_ids)
        self.total_samples = len(item_dict)

        all_signs = set()
        for rec in item_dict.values():
            all_signs.update(rec['signs'].tolist())
        self.max_slot_id = 28
        self.max_sign_id = max(all_signs) if all_signs else 0

        if BAIDU_LOG_BATCH_STATS and self.batch_stats:
            token_totals = [stat["token_total"] for stat in self.batch_stats]
            print(
                "[INFO] balanced user batching enabled: "
                f"batches={len(self.batch_stats)}, "
                f"users_per_batch<={BAIDU_BATCH_USERS}, "
                f"token_total[min/mean/max]="
                f"{min(token_totals)}/{sum(token_totals)/len(token_totals):.1f}/{max(token_totals)}"
            )

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        userid = self.user_ids[index]
        items = self.user_items[userid]

        if self.user_seq and userid in self.user_seq:
            seq_order = {logid: i for i, logid in enumerate(self.user_seq[userid])}
            items.sort(key=lambda x: seq_order.get(x[0], x[0]))
        else:
            items.sort(key=lambda x: x[0])

        feasigns = []
        labels = []
        logids = []
        for logid, feasign, label in items:
            logids.append(logid)
            feasigns.append(feasign)
            labels.append(label)

        return {
            'userid': userid,
            'logids': logids,
            'feasigns': feasigns,
            'labels': labels,
            'pred_mask': [1 if logid in self.pred_logids else 0 for logid in logids],
        }


class CTRTestSeqDataset(CTRUserDataset):
    """提交接口兼容类；保持完整用户序列，不在这里截断 max_ctx_len。"""

    def __init__(
        self,
        test_logids_ordered,
        item_dict,
        user_seq,
        max_feasign_per_slot=None,
        max_ctx_len=None,
    ):
        self.test_logids_ordered = list(test_logids_ordered)
        self.max_ctx_len = max_ctx_len
        super().__init__(
            item_dict=item_dict,
            user_seq=user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=set(self.test_logids_ordered),
        )


def make_collate_fn(max_slot_id):
    def collate_user_batch(batch):
        all_userids = []
        all_logids = []
        all_labels = []
        all_pred_masks = []
        all_feasigns = []
        user_offsets = [0]

        for item in batch:
            for i, logid in enumerate(item['logids']):
                all_userids.append(item['userid'])
                all_logids.append(logid)
                all_labels.append(item['labels'][i])
                all_pred_masks.append(item['pred_mask'][i])
                all_feasigns.append(item['feasigns'][i])
            user_offsets.append(len(all_labels))

        slot_data = {}
        for slot in range(1, max_slot_id + 1):
            values = []
            offsets = [0]
            for feasign in all_feasigns:
                if slot in feasign:
                    values.extend(feasign[slot])
                offsets.append(len(values))
            slot_data[slot] = (
                torch.tensor(values, dtype=torch.long),
                torch.tensor(offsets, dtype=torch.long),
            )

        result = {
            'userid': torch.tensor(all_userids, dtype=torch.long),
            'logid': torch.tensor(all_logids, dtype=torch.long),
            'label': torch.tensor(all_labels, dtype=torch.float32),
            'pred_mask': torch.tensor(all_pred_masks, dtype=torch.bool),
            'user_offsets': torch.tensor(user_offsets, dtype=torch.long),
        }
        result.update(slot_data)
        return result

    return collate_user_batch


def merge_user_batches(batches):
    if len(batches) == 1:
        return batches[0]

    merged = {}
    for key in ("userid", "logid", "label", "pred_mask"):
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)

    user_offsets = [0]
    token_base = 0
    for batch in batches:
        offsets = batch["user_offsets"].tolist()
        user_offsets.extend(token_base + int(offset) for offset in offsets[1:])
        token_base += int(offsets[-1])
    merged["user_offsets"] = torch.tensor(user_offsets, dtype=batches[0]["user_offsets"].dtype)

    for slot in range(1, 29):
        values_list = []
        offsets = [0]
        value_base = 0
        for batch in batches:
            values, slot_offsets = batch[slot]
            values_list.append(values)
            offsets.extend((slot_offsets[1:] + value_base).tolist())
            value_base += int(slot_offsets[-1])
        merged[slot] = (
            torch.cat(values_list, dim=0),
            torch.tensor(offsets, dtype=batches[0][slot][1].dtype),
        )

    return merged


def group_user_batches(batches, group_factor):
    if group_factor <= 1:
        return batches
    return [
        merge_user_batches(batches[i:i + group_factor])
        for i in range(0, len(batches), group_factor)
    ]


def group_user_batches_by_token_cap(batches, token_cap):
    if token_cap <= 0:
        return batches

    grouped = []
    current = []
    current_tokens = 0
    for batch in batches:
        batch_tokens = int(batch["user_offsets"][-1])
        if current and current_tokens + batch_tokens > token_cap:
            grouped.append(merge_user_batches(current))
            current = []
            current_tokens = 0
        current.append(batch)
        current_tokens += batch_tokens
    if current:
        grouped.append(merge_user_batches(current))
    return grouped


def _prepare_rep_count_matmul_indices(batch):
    if os.environ.get("BAIDU_REP_COUNT_MATMUL_DISABLE", "0") == "1":
        return batch
    if os.environ.get("BAIDU_REP_COUNT_MATMUL_PRECOMP_DISABLE", "0") == "1":
        return batch
    if not all(slot in batch for slot in REP_COUNT_MATMUL_SLOTS):
        return batch

    max_idx = 5_000_000 - 1
    count_dtype = (
        torch.float32
        if os.environ.get("BAIDU_REP_ENCODER_FP16_DISABLE", "0") == "1"
        else torch.float16
    )
    precompute_count_matrix = os.environ.get("BAIDU_REP_COUNT_MATRIX_PRECOMP_DISABLE", "0") != "1"
    for slot in REP_COUNT_MATMUL_SLOTS:
        values, offsets = batch[slot]
        if not (torch.is_tensor(values) and values.is_cuda and torch.is_tensor(offsets)):
            continue

        values = values.clamp(0, max_idx)
        unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
        token_count = offsets.numel() - 1
        unique_count = unique.numel()
        max_unique, max_elements = _rep_count_matmul_limits()
        if unique_count > max_unique or token_count * unique_count > max_elements:
            continue

        lengths = offsets[1:] - offsets[:-1]
        bag_idx = torch.repeat_interleave(
            torch.arange(token_count, device=values.device, dtype=torch.int64),
            lengths,
        )
        if precompute_count_matrix:
            counts = torch.zeros(
                (token_count, unique_count),
                device=values.device,
                dtype=count_dtype,
            )
            counts.index_put_(
                (bag_idx, inverse),
                torch.ones_like(inverse, dtype=count_dtype),
                accumulate=True,
            )
            batch[f"_rep_cm_counts_{slot}"] = (unique, counts)
        else:
            batch[f"_rep_cm_{slot}"] = (unique, inverse, bag_idx)
    return batch


# ============================================================
# 模型定义（来自 main.py）
# ============================================================

def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        moved = {
            k: (v if k == "user_offsets" else move_batch_to_device(v, device))
            for k, v in batch.items()
        }
        dev = torch.device(device)
        if (
            dev.type == "cuda"
            and "user_offsets" in moved
            and torch.is_tensor(moved["user_offsets"])
            and moved["user_offsets"].numel() > 1
            and os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") != "1"
            and os.environ.get("BAIDU_VARLEN_FLASH_ATTENTION_DISABLE", "0") != "1"
        ):
            offsets = moved["user_offsets"]
            moved["_user_offsets_cuda"] = offsets.to(device=dev, dtype=torch.int32)
            moved["_max_user_len"] = int((offsets[1:] - offsets[:-1]).max())
        if dev.type == "cuda":
            _prepare_rep_count_matmul_indices(moved)
            if (
                os.environ.get("BAIDU_TRT_MOE_DISABLE", "0") != "1"
                and "user_offsets" in moved
                and torch.is_tensor(moved["user_offsets"])
                and moved["user_offsets"].numel() > 1
            ):
                _reserve_trt_moe_workspace(
                    int(moved["user_offsets"][-1]),
                    dev,
                    _default_seq_encoder_dtype(dev),
                )
        return moved
    elif isinstance(batch, (list, tuple)):
        return [move_batch_to_device(x, device) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    else:
        return batch


def move_model_inputs_to_device(batch, device):
    dev = torch.device(device)
    moved = {"user_offsets": batch["user_offsets"].to(device=dev)}
    for key, value in batch.items():
        if isinstance(key, int):
            values, offsets = value
            moved[key] = (values.to(device=dev), offsets.to(device=dev))
    if (
        dev.type == "cuda"
        and moved["user_offsets"].numel() > 1
        and os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") != "1"
        and os.environ.get("BAIDU_VARLEN_FLASH_ATTENTION_DISABLE", "0") != "1"
    ):
        offsets = moved["user_offsets"]
        moved["_user_offsets_cuda"] = offsets.to(device=dev, dtype=torch.int32)
        moved["_max_user_len"] = int((offsets[1:] - offsets[:-1]).max())
    if dev.type == "cuda":
        _prepare_rep_count_matmul_indices(moved)
        if os.environ.get("BAIDU_TRT_MOE_DISABLE", "0") != "1" and moved["user_offsets"].numel() > 1:
            _reserve_trt_moe_workspace(
                int(moved["user_offsets"][-1]),
                dev,
                _default_seq_encoder_dtype(dev),
            )
    return moved


class ResultCollector:
    def __init__(self, enabled=True, chunk_preds=65536):
        self.enabled = enabled
        self.chunk_preds = max(1, int(chunk_preds))
        self.all_logids = []
        self.all_probs = []
        self.logid_parts = []
        self.prob_parts = []
        self.pending = 0

    def add(self, logids, probs, pred_mask, logid_mask=None):
        logid_mask = pred_mask if logid_mask is None else logid_mask
        if not self.enabled:
            self.all_logids.extend(logids[logid_mask].cpu().tolist())
            self.all_probs.extend(probs[pred_mask].cpu().tolist())
            return
        masked_logids = logids[logid_mask]
        masked_probs = probs[pred_mask]
        if masked_logids.numel() == 0:
            return
        self.logid_parts.append(masked_logids)
        self.prob_parts.append(masked_probs)
        self.pending += int(masked_logids.numel())
        if self.pending >= self.chunk_preds:
            self.flush()

    def add_indices(self, logids, probs, pred_indices_cpu, pred_indices_device):
        if not self.enabled:
            self.all_logids.extend(logids.index_select(0, pred_indices_cpu).cpu().tolist())
            self.all_probs.extend(probs.index_select(0, pred_indices_device).cpu().tolist())
            return
        if pred_indices_cpu.numel() == 0:
            return
        masked_logids = logids.index_select(0, pred_indices_cpu)
        masked_probs = probs.index_select(0, pred_indices_device)
        self.logid_parts.append(masked_logids)
        self.prob_parts.append(masked_probs)
        self.pending += int(masked_logids.numel())
        if self.pending >= self.chunk_preds:
            self.flush()

    def flush(self):
        if not self.logid_parts:
            return
        logids = torch.cat(self.logid_parts, dim=0)
        probs = torch.cat(self.prob_parts, dim=0)
        self.all_logids.extend(logids.cpu().tolist())
        self.all_probs.extend(probs.cpu().tolist())
        self.logid_parts.clear()
        self.prob_parts.clear()
        self.pending = 0

    def result(self):
        self.flush()
        return self.all_logids, self.all_probs


class RepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0, slot_num=0, d_model=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.single_value_slots = {1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 16, 18}
        self.count_matmul_slots = set(REP_COUNT_MATMUL_SLOTS)
        self.input_norm = nn.LayerNorm(slot_num * emb_dim)
        self.linear = nn.Linear(in_features=slot_num * emb_dim, out_features=d_model)

    def _count_matmul_embedding_bag(self, values, offsets, precomputed=None):
        if precomputed is None:
            unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
            bag_idx = None
        else:
            unique, inverse, bag_idx = precomputed

        token_count = offsets.numel() - 1
        unique_count = unique.numel()
        max_unique, max_elements = _rep_count_matmul_limits()
        if unique_count > max_unique or token_count * unique_count > max_elements:
            return None

        if bag_idx is None:
            lengths = offsets[1:] - offsets[:-1]
            bag_idx = torch.repeat_interleave(
                torch.arange(token_count, device=values.device, dtype=torch.int64),
                lengths,
            )
        counts = torch.zeros(
            (token_count, unique_count),
            device=values.device,
            dtype=self.emb.weight.dtype,
        )
        counts.index_put_(
            (bag_idx, inverse),
            torch.ones_like(inverse, dtype=self.emb.weight.dtype),
            accumulate=True,
        )
        return counts.matmul(self.emb(unique))

    def forward(self, batch):
        pooled_embs = []
        max_idx = self.emb.num_embeddings - 1
        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            offsets = offsets.to(values.device)
            values = values.clamp(0, max_idx)  # 超出 vocab_size 的 sign id 截断，避免越界
            if (
                os.environ.get("BAIDU_REP_SINGLE_EMBED_DISABLE", "0") != "1"
                and i + 1 in self.single_value_slots
                and values.numel() == offsets.numel() - 1
            ):
                res = self.emb(values)
            elif (
                os.environ.get("BAIDU_REP_COUNT_MATMUL_DISABLE", "0") != "1"
                and i + 1 in self.count_matmul_slots
            ):
                precomputed_counts = batch.get(f"_rep_cm_counts_{i + 1}")
                if precomputed_counts is not None:
                    unique, counts = precomputed_counts
                    if counts.dtype != self.emb.weight.dtype:
                        counts = counts.to(dtype=self.emb.weight.dtype)
                    res = counts.matmul(self.emb(unique))
                else:
                    precomputed = batch.get(f"_rep_cm_{i + 1}")
                    res = self._count_matmul_embedding_bag(values, offsets, precomputed=precomputed)
                if res is None:
                    res = F.embedding_bag(
                        values,
                        self.emb.weight,
                        offsets,
                        mode='sum',
                        include_last_offset=True,
                    )
            elif os.environ.get("BAIDU_REP_EMBEDDING_BAG_DISABLE", "0") == "1":
                sign_emb = self.emb(values)
                res = torch.segment_reduce(sign_emb, reduce='sum', offsets=offsets, initial=0)
            else:
                res = F.embedding_bag(
                    values,
                    self.emb.weight,
                    offsets,
                    mode='sum',
                    include_last_offset=True,
                )
            pooled_embs.append(res)
        fused_embs = torch.cat(pooled_embs, dim=1)
        norm_emb = self.input_norm(fused_embs)
        rep_emb = self.linear(norm_emb)
        return rep_emb


def scaled_dot_product(q, k, v, extension):
    if extension is not None and "user_offsets" in extension and os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") != "1":
        offsets = extension["user_offsets"]

        if (
            q.is_cuda
            and q.size(0) == 1
            and q.size(-1) % 8 == 0
            and os.environ.get("BAIDU_VARLEN_FLASH_ATTENTION_DISABLE", "0") != "1"
        ):
            offsets_cuda = extension.get("user_offsets_cuda")
            max_user_len = extension.get("max_user_len")
            offsets_list = None
            if offsets_cuda is None:
                if torch.is_tensor(offsets):
                    offsets_cuda = offsets.to(device=q.device, dtype=torch.int32)
                else:
                    offsets_list = [int(offset) for offset in offsets]
                    offsets_cuda = torch.tensor(offsets_list, device=q.device, dtype=torch.int32)
            if max_user_len is None:
                if offsets_list is None:
                    if torch.is_tensor(offsets) and offsets.device.type == "cpu":
                        max_user_len = int((offsets[1:] - offsets[:-1]).max())
                    else:
                        offsets_list = offsets_cuda.detach().cpu().tolist()
                        max_user_len = max(end - start for start, end in zip(offsets_list[:-1], offsets_list[1:]))
                else:
                    max_user_len = max(end - start for start, end in zip(offsets_list[:-1], offsets_list[1:]))

            _, heads, tokens, head_dim = q.shape
            q_flash = q.permute(0, 2, 1, 3).reshape(tokens, heads, head_dim).contiguous().to(torch.float16)
            k_flash = k.permute(0, 2, 1, 3).reshape(tokens, heads, head_dim).contiguous().to(torch.float16)
            v_flash = v.permute(0, 2, 1, 3).reshape(tokens, heads, head_dim).contiguous().to(torch.float16)
            out_flash, _, _, _, _ = torch.ops.aten._flash_attention_forward(
                q_flash,
                k_flash,
                v_flash,
                offsets_cuda,
                offsets_cuda,
                max_user_len,
                max_user_len,
                0.0,
                True,
                False,
                scale=None,
            )
            return out_flash.to(q.dtype).reshape(1, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous()

        if torch.is_tensor(offsets):
            if offsets.device.type != "cpu":
                offsets = offsets.detach().cpu()
            offsets = offsets.tolist()

        out = torch.empty_like(q)
        for start, end in zip(offsets[:-1], offsets[1:]):
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            out[:, :, start:end, :] = F.scaled_dot_product_attention(
                q[:, :, start:end, :],
                k[:, :, start:end, :],
                v[:, :, start:end, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        return out

    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if extension is not None and "mask" in extension:
        mask = extension["mask"]
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


class Expert(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TopKGate(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noisy_gating=True):
        super().__init__()
        self.w_g = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def forward(self, x, return_probs=True):
        # x: [B,S,D]
        logits = self.w_g(x)  # [B,S,E]

        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * 0.1

        probs = torch.softmax(logits, dim=-1)  # [B,S,E]

        topk_score, topk_idx = torch.topk(probs, self.k, dim=-1)  # [B,S,k]

        return topk_idx, topk_score, probs if return_probs else None

class SMoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self._trt_moe_enabled = False
        self._trt_moe_launcher = None
        self._trt_moe_workspace = None

        self.experts = nn.ModuleList([
            Expert(d_model, dim_ff) for _ in range(num_experts)
        ])

        self.gate = TopKGate(d_model, num_experts, k=k)

    def prepare_trt_moe(self):
        if not torch.cuda.is_available():
            return False
        if self.k != 2 or self.num_experts != 8:
            return False

        launcher = _BaiduMoeTop2Launcher.get()

        with torch.no_grad():
            fc1_weight = torch.stack(
                [expert.fc1.weight.detach().transpose(0, 1).contiguous() for expert in self.experts],
                dim=0,
            ).contiguous()
            fc1_bias = torch.stack([expert.fc1.bias.detach() for expert in self.experts], dim=0).contiguous()
            fc2_weight = torch.stack(
                [expert.fc2.weight.detach().transpose(0, 1).contiguous() for expert in self.experts],
                dim=0,
            ).contiguous()
            fc2_bias = torch.stack([expert.fc2.bias.detach() for expert in self.experts], dim=0).contiguous()

            if BAIDU_PRUNE_FFN:
                dim_ff = fc1_weight.shape[2]
                keep_count = int(dim_ff * (1.0 - BAIDU_PRUNE_RATIO))
                if 0 < keep_count < dim_ff:
                    fc1_norms = fc1_weight.norm(dim=1)
                    fc2_norms = fc2_weight.norm(dim=2)
                    importance = fc1_norms * fc2_norms
                    keep_indices = importance.topk(keep_count, dim=1).indices.sort(dim=1).values
                    idx_expanded1 = keep_indices.unsqueeze(1).expand(-1, fc1_weight.shape[1], -1)
                    idx_expanded2 = keep_indices.unsqueeze(2).expand(-1, -1, fc2_weight.shape[2])
                    fc1_weight = fc1_weight.gather(2, idx_expanded1).contiguous()
                    fc1_bias = fc1_bias.gather(1, keep_indices).contiguous()
                    fc2_weight = fc2_weight.gather(1, idx_expanded2).contiguous()
                    print(
                        f"[INFO] BAIDU_PRUNE_FFN: plugin dim_ff {dim_ff}->{keep_count} "
                        f"({BAIDU_PRUNE_RATIO * 100:.0f}% pruned)"
                    )

        self.register_buffer("_trt_fc1_weight", fc1_weight, persistent=False)
        self.register_buffer("_trt_fc1_bias", fc1_bias, persistent=False)
        self.register_buffer("_trt_fc2_weight", fc2_weight, persistent=False)
        self.register_buffer("_trt_fc2_bias", fc2_bias, persistent=False)
        self._trt_moe_launcher = launcher
        self._trt_moe_enabled = True
        return True

    def forward(self, x):
        # x: [B,S,D]
        B, S, D = x.shape

        need_probs = self.training or os.environ.get("BAIDU_MOE_LOSS_DISABLE", "1") != "1"
        topk_idx, topk_score, probs = self.gate(x, return_probs=need_probs)

        # flatten
        x_flat = x.reshape(-1, D)                # [B*S, D]
        idx_flat = topk_idx.reshape(-1, self.k)  # [B*S, k]
        score_flat = topk_score.reshape(-1, self.k)

        if self._trt_moe_enabled and x.is_cuda and x.dtype in (torch.float16, torch.float32):
            out = torch.empty_like(x)
            out_flat = out.reshape(-1, D)
            hidden_dim = self._trt_fc1_bias.shape[1]
            workspace = _reserve_trt_moe_workspace(
                x_flat.shape[0], x.device, x.dtype, D, hidden_dim, self.num_experts, self.k,
            )
            if workspace is None:
                workspace_bytes = self._trt_moe_launcher.workspace_size(
                    x_flat.shape[0], D, hidden_dim, self.num_experts, self.k, x.dtype,
                )
                if self._trt_moe_workspace is None or self._trt_moe_workspace.numel() < workspace_bytes:
                    self._trt_moe_workspace = torch.empty(workspace_bytes, device=x.device, dtype=torch.uint8)
                workspace = self._trt_moe_workspace
            self._trt_moe_launcher.launch(
                x_flat.contiguous(),
                idx_flat,
                score_flat,
                self._trt_fc1_weight,
                self._trt_fc1_bias,
                self._trt_fc2_weight,
                self._trt_fc2_bias,
                workspace,
                out_flat,
            )
        else:
            out = torch.zeros_like(x)
            out_flat = out.reshape(-1, D)
            for i in range(self.num_experts):
                # 找到被路由到 expert i 的 token
                mask = (idx_flat == i)  # [B*S, k]

                # 哪些 token 命中了 expert i
                token_idx, k_idx = mask.nonzero(as_tuple=True)

                selected_x = x_flat[token_idx]  # [N, D]

                expert_out = self.experts[i](selected_x)  # [N, D]

                weight = score_flat[token_idx, k_idx].unsqueeze(-1)

                out_flat[token_idx] += expert_out * weight

        if need_probs:
            importance = probs.sum(dim=(0,1))  # [E]
            moe_loss = (importance.std() / (importance.mean() + 1e-6))
        else:
            moe_loss = x.new_zeros(())

        return out, moe_loss


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu",
                 attention_fn=scaled_dot_product):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_layers = num_layers
        assert d_model % n_heads == 0

        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(num_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.ffn1 = nn.ModuleList([nn.Linear(d_model, dim_ff) for _ in range(num_layers)])
        self.ffn2 = nn.ModuleList([nn.Linear(dim_ff, d_model) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.act = getattr(F, act)
        self.attention_fn = attention_fn
        self.moe = nn.ModuleList([
            SMoE(d_model, dim_ff, num_experts=8, k=2)
            for _ in range(num_layers)
        ])

    def forward(self, x, extension):
        x = x.unsqueeze(0)
        B, S, D = x.shape

        moe_loss_total = 0.0
        for i in range(self.num_layers):
            residual = x
            x = self.norm1[i](x)
            qkv = self.qkv_proj[i](x)
            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            attn_out = self.attention_fn(q, k, v, extension)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            x = residual + self.out_proj[i](attn_out)
            residual = x
            x = self.norm2[i](x)

            moe_out, moe_loss = self.moe[i](x)

            x = residual + moe_out

            moe_loss_total = moe_loss_total + moe_loss

        return x, moe_loss_total


class CTRModel(nn.Module):
    def __init__(self, rep_encoder, seq_encoder, d_model):
        super().__init__()
        self.rep_encoder = rep_encoder
        self.seq_encoder = seq_encoder
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def get_sequence_causal_mask(self, seq_info):
        lengths = seq_info[1:] - seq_info[:-1]
        lengths = lengths.view(-1)
        indices = torch.cumsum(torch.ones_like(lengths), dim=0) - 1
        result = torch.repeat_interleave(indices, lengths)
        a = result.view(1, -1) - result.view(-1, 1)
        out_mask = torch.tril((a == 0).to(torch.int32)).bool()
        return out_mask

    def forward(self, batch):
        seq_input = self.rep_encoder(batch)
        seq_dtype = self.seq_encoder.qkv_proj[0].weight.dtype
        if seq_input.dtype != seq_dtype:
            seq_input = seq_input.to(dtype=seq_dtype)
        if os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") != "1":
            extension = {
                "user_offsets": batch["user_offsets"],
                "user_offsets_cuda": batch.get("_user_offsets_cuda"),
                "max_user_len": batch.get("_max_user_len"),
            }
        else:
            seq_mask = self.get_sequence_causal_mask(batch["user_offsets"].to(seq_input.device))
            extension = {"mask": seq_mask.unsqueeze(0).unsqueeze(0)}
        encoder_output, moe_loss = self.seq_encoder(
            x=seq_input,
            extension=extension,
        )
        encoder_output_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
        pred = self.linear(encoder_output)
        pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
        return pred_logits, moe_loss


# ============================================================
# 模型加载入口
# ============================================================

def load_model(device='cuda:0', ckpt_path=None):
    """加载模型并返回，供 evaluation.py 调用。

    Args:
        device: 推理设备（默认 'cuda:0'）
        ckpt_path: checkpoint 文件路径，默认使用 infer.py 同目录下的 ckpt.pt

    Returns:
        (model, device) 元组
    """
    if ckpt_path is None and isinstance(device, Path):
        ckpt_path = device
        device = 'cuda:0'
    elif ckpt_path is None and isinstance(device, str):
        maybe_path = Path(device)
        if maybe_path.suffix == ".pt" or "/" in device:
            ckpt_path = maybe_path
            device = 'cuda:0'

    emb_dim = 512
    slot_num = 28
    vocab_size = 5000000
    d_model = 512
    n_heads = 8
    num_layers = 8
    dim_ff = 1024

    rep_encoder = RepEncoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        padding_idx=0,
        slot_num=slot_num,
        d_model=d_model,
    )
    seq_encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        act="relu",
    )
    model = CTRModel(rep_encoder, seq_encoder, d_model=d_model)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    # 若需要加载自定义修改的权重，请修改 479-488行逻辑，强制使用你文件夹中的权重
    # 测评系统默认使用原始官方权重
    if ckpt_path is None:
        ckpt_path = Path(__file__).parent / 'ckpt.pt'
    else:
        ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found, using random weights")

    model.to(dev)
    model.eval()
    if dev.type == "cuda" and os.environ.get("BAIDU_REP_ENCODER_FP16_DISABLE", "0") != "1":
        model.rep_encoder.to(dtype=torch.float16)
        print("[INFO] RepEncoder FP16 enabled")
    seq_fp16_env = os.environ.get("BAIDU_SEQ_ENCODER_FP16")
    if seq_fp16_env is None:
        seq_encoder_fp16 = dev.type == "cuda" and os.environ.get("BAIDU_SEQ_ENCODER_FP16_DISABLE", "0") != "1"
    else:
        seq_encoder_fp16 = dev.type == "cuda" and seq_fp16_env == "1"
    if seq_encoder_fp16:
        model.seq_encoder.to(dtype=torch.float16)
        model.linear.to(dtype=torch.float16)
        print("[INFO] SeqEncoder FP16 enabled")
    if dev.type == "cuda" and os.environ.get("BAIDU_TRT_MOE_DISABLE", "0") != "1":
        try:
            prepared = 0
            for moe in model.seq_encoder.moe:
                prepared += int(moe.prepare_trt_moe())
            print(f"[INFO] TensorRT MoE plugin enabled for {prepared}/{len(model.seq_encoder.moe)} layers")
        except Exception as exc:
            print(f"[WARNING] TensorRT MoE plugin unavailable, falling back to PyTorch SMoE: {exc}")
    if BAIDU_LOGIT_BIAS != 0.0:
        model.linear.bias.data.add_(BAIDU_LOGIT_BIAS)
        print(f"[INFO] Applied BAIDU_LOGIT_BIAS={BAIDU_LOGIT_BIAS:.6f}")
    print(f"[INFO] Model ready. Device: {dev}")

    return model, dev


# ============================================================
# 打分工具（与 evaluation.py 保持一致）
# ============================================================

def _read_predict(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(float(line))
    import numpy as np
    return np.array(predictions)


def _read_label(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    labels.append(float(parts[3]))
                else:
                    labels.append(float(line))
    import numpy as np
    return np.array(labels)


def _cal_score(predict_file, label_file, default_latency=0.0):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    predictions = _read_predict(predict_file)
    labels = _read_label(label_file)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print('[WARNING] only one class present in labels, AUC is not defined, returning 0.5')
        auc = 0.5
    else:
        auc = roc_auc_score(labels, predictions)

    mean_pred = np.mean(predictions)
    mean_label = np.mean(labels)
    if mean_label == 0:
        pcoc = 1.0 if mean_pred == 0 else float('inf')
    else:
        pcoc = float(mean_pred / mean_label)

    latency = default_latency
    base_latency = 300
    score_latency = max(0.0, (base_latency - latency) / base_latency) if latency < base_latency else 0.0

    if pcoc < 0.85 or pcoc > 1.15:
        score_model = 0.0
    else:
        score_model = ((auc - 0.65) * 1000 + (0.15 - abs(pcoc - 1)) / 0.15 * 10) / 360

    score_all = score_latency * 70 + score_model * 30

    return {
        'auc': auc,
        'pcoc': pcoc,
        'latency': latency,
        'score_latency': score_latency,
        'score_model': score_model,
        'score_all': score_all,
    }


# ============================================================
# main：直接运行 infer.py 进行测试
# ============================================================

def main():
    import io
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint 文件路径，默认使用同目录下的 ckpt.pt')
    args = parser.parse_args()

    cur_path = Path(__file__).parent.absolute()
    ref_dir = cur_path / 'dataset'
    history_dir = ref_dir / 'history'
    input_file = ref_dir / 'test.csv'
    output_file = Path('predict.txt')
    label_file = ref_dir / 'label_data.txt'

    # ----- 数据加载，优先从缓存读取 -----
    MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB per shard
    batches_cache_dir = ref_dir / 'cached_batches'

    use_cached_batches = (
        not BAIDU_FORCE_REBUILD_BATCHES
        and batches_cache_dir.exists()
        and any(batches_cache_dir.glob('shard_*.pt'))
    )
    if use_cached_batches:
        print(f'[INFO] loading cached batch shards from {batches_cache_dir}')
        all_batches = []
        shard_files = sorted(batches_cache_dir.glob('shard_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]))
        for sf in shard_files:
            shard_batches = torch.load(sf, weights_only=False)
            all_batches.extend(shard_batches)
            print(f'[INFO] loaded {len(shard_batches)} batches from {sf.name}')
        print(f'[INFO] loaded {len(all_batches)} cached batches total from {len(shard_files)} shards')
    else:
        if BAIDU_FORCE_REBUILD_BATCHES and batches_cache_dir.exists():
            print(f'[INFO] BAIDU_FORCE_REBUILD_BATCHES=1, ignoring cached batches in {batches_cache_dir}')
        print('[INFO] start loading data from CSV')
        history_files = sorted(history_dir.glob('*.csv')) if history_dir.exists() else []
        all_files = history_files + [input_file]

        item_dict, user_seq = load_sample_files(sample_files_list=all_files)
        test_pred_logids = load_logids_from_file(input_file)
        print(f'[INFO] Test pred logids count: {len(test_pred_logids)}')

        max_feasign_per_slot = {1: 2}
        test_dataset = CTRUserDataset(
            item_dict, user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=test_pred_logids,
        )
        print(f'[INFO] num_users={test_dataset.num_users}, '
              f'total_samples={test_dataset.total_samples}, '
              f'pred_samples={len(test_pred_logids)}, '
              f'max_sign_id={test_dataset.max_sign_id}')

        test_loader = DataLoader(
            test_dataset,
            batch_size=BAIDU_BATCH_USERS,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(test_dataset.max_slot_id),
        )

        # 收集 batches 并按分片缓存
        print('[INFO] collecting batches and saving sharded cache...')
        all_batches = [batch for batch in test_loader]

        batches_cache_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        current_shard = []
        current_size = 0
        for batch in all_batches:
            buf = io.BytesIO()
            torch.save(batch, buf)
            batch_size_bytes = buf.tell()
            if current_shard and current_size + batch_size_bytes > MAX_SHARD_BYTES:
                shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
                torch.save(current_shard, shard_path)
                print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                      f'~{current_size / 1024**3:.2f}GB')
                shard_idx += 1
                current_shard = []
                current_size = 0
            current_shard.append(batch)
            current_size += batch_size_bytes
        if current_shard:
            shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
            torch.save(current_shard, shard_path)
            print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                  f'~{current_size / 1024**3:.2f}GB')
            shard_idx += 1
        print(f'[INFO] saved {len(all_batches)} batches to {shard_idx} shards in {batches_cache_dir}')

    if os.environ.get("BAIDU_SEGMENTED_ATTENTION_DISABLE", "0") == "1":
        group_token_cap = 0
        group_factor = 1
    elif "BAIDU_BATCH_GROUP_TOKEN_CAP" in os.environ:
        group_token_cap = int(os.environ["BAIDU_BATCH_GROUP_TOKEN_CAP"])
        group_factor = 1
    elif "BAIDU_BATCH_GROUP_FACTOR" in os.environ:
        group_token_cap = 0
        group_factor = int(os.environ["BAIDU_BATCH_GROUP_FACTOR"])
    else:
        group_token_cap = 300000
        group_factor = 1
    if group_token_cap > 0:
        original_batch_count = len(all_batches)
        all_batches = group_user_batches_by_token_cap(all_batches, group_token_cap)
        max_tokens = max(int(batch["user_offsets"][-1]) for batch in all_batches) if all_batches else 0
        print(f'[INFO] grouped batches: {original_batch_count} -> {len(all_batches)} '
              f'(token_cap={group_token_cap}, max_tokens={max_tokens})')
    elif group_factor > 1:
        original_batch_count = len(all_batches)
        all_batches = group_user_batches(all_batches, group_factor)
        max_tokens = max(int(batch["user_offsets"][-1]) for batch in all_batches) if all_batches else 0
        print(f'[INFO] grouped batches: {original_batch_count} -> {len(all_batches)} '
              f'(factor={group_factor}, max_tokens={max_tokens})')

    print('[INFO] data loading done')

    # ----- 加载模型 -----
    model, dev = load_model(ckpt_path=args.ckpt)
    if dev.type == "cuda" and os.environ.get("BAIDU_TRT_MOE_DISABLE", "0") != "1" and all_batches:
        max_tokens = max(int(batch["user_offsets"][-1]) for batch in all_batches)
        _reserve_trt_moe_workspace(
            max_tokens,
            dev,
            model.seq_encoder.qkv_proj[0].weight.dtype,
        )

    # ----- 推理 -----
    print('*' * 20 + ' start inference ' + '*' * 20)
    all_logids = []
    all_probs = []
    time_sum = 0.0

    with torch.inference_mode():
        collector = ResultCollector(
            enabled=BAIDU_CHUNKED_COLLECT,
            chunk_preds=BAIDU_COLLECT_CHUNK_PREDS,
        )
        use_cuda = dev.type == "cuda"
        for batch in _progress(all_batches, desc="Inference"):
            if BAIDU_CPU_METADATA:
                pred_mask_cpu = batch["pred_mask"].bool()
                cpu_logids = batch["logid"].cpu() if torch.is_tensor(batch["logid"]) else batch["logid"]
                model_batch = move_model_inputs_to_device(batch, dev)
                pred_count = int(pred_mask_cpu.sum().item())
                pred_total = int(pred_mask_cpu.numel())
                use_pred_indices = (
                    BAIDU_ADAPT_PRED_INDICES
                    and (pred_count == 0 or (pred_total > 0 and (pred_count / pred_total) < BAIDU_PRED_INDICES_MAX_DENSITY))
                )
                if use_pred_indices:
                    pred_indices_cpu = pred_mask_cpu.nonzero(as_tuple=False).flatten()
                    pred_indices_device = pred_indices_cpu.to(dev)
                    pred_mask_device = None
                else:
                    pred_indices_cpu = pred_indices_device = None
                    pred_mask_device = pred_mask_cpu.to(dev)
            else:
                model_batch = move_batch_to_device(batch, dev)
                pred_mask_device = model_batch["pred_mask"].bool()
                pred_mask_cpu = pred_mask_device.cpu()
                pred_indices_cpu = pred_indices_device = None
                cpu_logids = model_batch["logid"]

            t_start = time.time()
            logits, moe_loss = model(model_batch)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            time_sum += time.time() - t_start

            if pred_indices_cpu is not None and BAIDU_CPU_METADATA:
                collector.add_indices(cpu_logids, probs, pred_indices_cpu, pred_indices_device)
            else:
                collector.add(cpu_logids, probs, pred_mask_device, logid_mask=pred_mask_cpu)

        all_logids, all_probs = collector.result()

    print(f'[INFO] inference time: {round(time_sum, 4)}s')
    print('*' * 20 + ' end inference ' + '*' * 20)

    # ----- 按 test.csv 顺序写预测文件 -----
    logid_to_prob = dict(zip(all_logids, all_probs))
    test_logids_in_order = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_logids_in_order.append(int(line.split(',')[0]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for logid in test_logids_in_order:
            f.write(f"{logid_to_prob[logid]}\n")
    print(f'[INFO] predictions written to {output_file}, total: {len(test_logids_in_order)}')

    # ----- 打分 -----
    if label_file.exists():
        result = _cal_score(output_file, label_file, default_latency=time_sum)
        print(f'[INFO] AUC:            {result["auc"]:.6f}')
        print(f'[INFO] PCOC:           {result["pcoc"]:.6f}')
        print(f'[INFO] Latency:        {result["latency"]:.4f}s')
        print(f'[INFO] score_latency:  {result["score_latency"]:.6f}')
        print(f'[INFO] score_model:    {result["score_model"]:.6f}')
        print(f'[INFO] score_all:      {result["score_all"]:.6f}')
        return result
    else:
        print(f'[WARNING] label file {label_file} not found, skipping scoring')
        return None


if __name__ == '__main__':
    main()
