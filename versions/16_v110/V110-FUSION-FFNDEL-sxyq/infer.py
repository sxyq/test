import sys
import os
from contextlib import nullcontext

# 获取当前环境脚本所在目录或指定绝对路径
for lib_dir in ("../libraries", "./libraries"):
    if os.path.exists(lib_dir):
        lib_path = os.path.abspath(lib_dir)
        if lib_path not in sys.path:
            sys.path.append(lib_path)

import math
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


V14_FP16_WEIGHTS_ENABLED = os.environ.get("GRAB_V14_FP16_WEIGHTS", "1").lower() not in {"0", "false", "no"}
V15_FP16_DEEP_ENABLED = os.environ.get("GRAB_V15_FP16_DEEP", "1").lower() not in {"0", "false", "no"}
V16_FP16_EMB_ENABLED = os.environ.get("GRAB_V16_FP16_EMB", "1").lower() not in {"0", "false", "no"}
V24_SPARSE_EMB_ENABLED = os.environ.get("GRAB_V24_SPARSE_EMB", "1").lower() not in {"0", "false", "no"}
V27_NO_AUXLOSS_ENABLED = os.environ.get("GRAB_V27_NO_AUXLOSS", "1").lower() not in {"0", "false", "no"}
V29_NO_MOELOSS_RETURN_ENABLED = os.environ.get("GRAB_V29_NO_MOELOSS_RETURN", "1").lower() not in {"0", "false", "no"}
V30_CPU_METADATA_ENABLED = os.environ.get("GRAB_V30_CPU_METADATA", "1").lower() not in {"0", "false", "no"}
V32_CHUNKED_COLLECT_ENABLED = os.environ.get("GRAB_V32_CHUNKED_COLLECT", "1").lower() not in {"0", "false", "no"}
V32_COLLECT_CHUNK_PREDS = int(os.environ.get("GRAB_V32_COLLECT_CHUNK_PREDS", "65536"))
V38_ADAPT_PRED_INDICES_ENABLED = os.environ.get("GRAB_V38_ADAPT_PRED_INDICES", "1").lower() not in {"0", "false", "no"}
V38_PRED_INDICES_MAX_DENSITY = float(os.environ.get("GRAB_V38_PRED_INDICES_MAX_DENSITY", "0.125"))
V42_SILENT_RUNNER_ENABLED = os.environ.get("GRAB_V42_SILENT_RUNNER", "1").lower() not in {"0", "false", "no"}
V103_PER_USER_CAUSAL_SDPA = os.environ.get("GRAB_V103_PER_USER_CAUSAL_SDPA", "1").lower() not in {"0", "false", "no"}
V109_DENSE_SMOE = os.environ.get("GRAB_V109_DENSE_SMOE", "1").lower() not in {"0", "false", "no"}
V110_EINSUM_FUSION = os.environ.get("GRAB_V110_EINSUM_FUSION", "1").lower() not in {"0", "false", "no"}
V110_REMOVE_DEAD_FFN = os.environ.get("GRAB_V110_REMOVE_DEAD_FFN", "1").lower() not in {"0", "false", "no"}
V110_OFFSETS_STAY_CPU = os.environ.get("GRAB_V110_OFFSETS_STAY_CPU", "1").lower() not in {"0", "false", "no"}
V110_FLASH_ATTN_INSTALL = os.environ.get("GRAB_V110_FLASH_ATTN_INSTALL", "1").lower() not in {"0", "false", "no"}
VERBOSE_INFER_ENABLED = os.environ.get("GRAB_VERBOSE_INFER", "0").lower() not in {"0", "false", "no"}


def _info(message, always=False):
    if always or VERBOSE_INFER_ENABLED or not V42_SILENT_RUNNER_ENABLED:
        print(message)


def _warn(message):
    print(message)


def _progress(iterable, **kwargs):
    if V42_SILENT_RUNNER_ENABLED and not VERBOSE_INFER_ENABLED:
        return iterable
    return tqdm(iterable, **kwargs)


def configure_runtime():
    """Enable safe CUDA runtime fast paths without changing model semantics."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_device(requested_device=None, require_cuda=True):
    """Resolve runtime device and fail early on unsupported CPU fallback paths."""
    if requested_device:
        dev = torch.device(requested_device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device '{requested_device}', but CUDA is unavailable."
            )
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if require_cuda:
        raise RuntimeError(
            "CUDA is unavailable. This submission is intended for the AI Studio GPU runtime. "
            "Running the full checkpoint on CPU is a known unstable path and may be killed by the platform."
        )
    return torch.device("cpu")


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
    _info(f'[INFO] loading {len(sample_files)} files: {[str(f) for f in sample_files]}')

    item_dict = {}
    user_logs = defaultdict(list)

    for sample_file in _progress(sample_files, desc='Loading sample files'):
        has_clk = _detect_has_clk(sample_file)
        min_parts = 5 if has_clk else 4
        _info(f'  {sample_file.name}: has_clk={has_clk}')

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

    _info(f'[INFO] loaded {len(item_dict)} records, {len(user_seq)} users')
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

        self.user_ids = sorted(self.user_items.keys())
        self.num_users = len(self.user_ids)
        self.total_samples = len(item_dict)

        all_signs = set()
        for rec in item_dict.values():
            all_signs.update(rec['signs'].tolist())
        self.max_slot_id = 28
        self.max_sign_id = max(all_signs) if all_signs else 0

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
    """Compatibility wrapper for the evaluator's expected dataset class name/signature."""

    def __init__(
        self,
        test_logids_ordered,
        item_dict,
        user_seq,
        max_feasign_per_slot=None,
        max_ctx_len=None,
    ):
        pred_logids = set(test_logids_ordered)
        super().__init__(
            item_dict=item_dict,
            user_seq=user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=pred_logids,
        )
        self.test_logids_ordered = list(test_logids_ordered)
        self.max_ctx_len = max_ctx_len


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


# ============================================================
# 模型定义（来自 main.py）
# ============================================================

def move_batch_to_device(batch, device, non_blocking=False):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device, non_blocking=non_blocking) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [move_batch_to_device(x, device, non_blocking=non_blocking) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    else:
        return batch


def move_model_inputs_to_device(batch, device, non_blocking=False):
    # V110: user_offsets stays on CPU — per_user_causal_sdpa only needs CPU offsets
    # Avoids GPU→CPU round-trip in CTRModel.forward
    if V110_OFFSETS_STAY_CPU:
        moved = {
            "user_offsets": batch["user_offsets"],  # stays CPU
        }
    else:
        moved = {
            "user_offsets": batch["user_offsets"].to(device, non_blocking=non_blocking),
        }
    for key, value in batch.items():
        if isinstance(key, int):
            values, offsets = value
            moved[key] = (
                values.to(device, non_blocking=non_blocking),
                offsets.to(device, non_blocking=non_blocking),
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
        self.input_norm = nn.LayerNorm(slot_num * emb_dim)
        self.linear = nn.Linear(in_features=slot_num * emb_dim, out_features=d_model)
        self.v15_fp16_linear_active = False
        self.v16_fp16_embedding_active = False
        self.v24_sparse_embedding_active = False
        self.v24_sparse_embedding_failed = False

    def forward(self, batch):
        if self.v24_sparse_embedding_active:
            try:
                return self.forward_sparse_embedding_bag(batch)
            except Exception as exc:
                self.v24_sparse_embedding_active = False
                self.v24_sparse_embedding_failed = True
                _warn(f"[WARNING] V24 sparse embedding path failed once; fallback to V16 segment_reduce path: {type(exc).__name__}: {exc}")
        return self.forward_segment_reduce(batch)

    def forward_segment_reduce(self, batch):
        pooled_embs = []
        max_idx = self.emb.num_embeddings - 1
        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            offsets = offsets.to(values.device)
            values = values.clamp(0, max_idx)  # 超出 vocab_size 的 sign id 截断，避免越界
            sign_emb = self.emb(values)
            if self.v16_fp16_embedding_active:
                sign_emb = sign_emb.float()
            res = torch.segment_reduce(sign_emb, reduce='sum', offsets=offsets, initial=0)
            pooled_embs.append(res)
        fused_embs = torch.cat(pooled_embs, dim=1)
        norm_emb = self.input_norm(fused_embs)
        if self.v15_fp16_linear_active:
            norm_emb = norm_emb.half()
        rep_emb = self.linear(norm_emb)
        if self.v15_fp16_linear_active:
            rep_emb = rep_emb.float()
        return rep_emb

    def forward_sparse_embedding_bag(self, batch):
        # Merge all slot bags into one embedding_bag call, then reshape back to
        # the original [token, slot * dim] layout before LayerNorm/Linear.
        merged_values = []
        merged_offsets = []
        max_idx = self.emb.num_embeddings - 1
        cursor = 0
        first_values, first_offsets = batch[1]
        device = first_values.device

        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            values = values.clamp(0, max_idx)
            offsets = offsets.to(device=values.device)
            merged_values.append(values)
            merged_offsets.append(offsets[:-1] + cursor)
            cursor += values.numel()

        if merged_values:
            merged_values = torch.cat(merged_values, dim=0)
        else:
            merged_values = torch.empty(0, dtype=torch.long, device=device)

        merged_offsets.append(
            torch.tensor([cursor], dtype=torch.long, device=device)
        )
        merged_offsets = torch.cat(merged_offsets, dim=0)

        fused_embs = F.embedding_bag(
            merged_values,
            self.emb.weight,
            merged_offsets,
            mode="sum",
            include_last_offset=True,
        )
        fused_embs = fused_embs.view(self.slot_num, -1, self.emb_dim).transpose(0, 1)
        fused_embs = fused_embs.reshape(fused_embs.size(0), self.slot_num * self.emb_dim)
        if self.v16_fp16_embedding_active:
            fused_embs = fused_embs.float()

        norm_emb = self.input_norm(fused_embs)
        if self.v15_fp16_linear_active:
            norm_emb = norm_emb.half()
        rep_emb = self.linear(norm_emb)
        if self.v15_fp16_linear_active:
            rep_emb = rep_emb.float()
        return rep_emb


def scaled_dot_product(q, k, v, extension):
    if hasattr(F, "scaled_dot_product_attention"):
        attn_mask = None
        if extension is not None and "mask" in extension:
            attn_mask = extension["mask"]
        return F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if extension is not None and "mask" in extension:
        mask = extension["mask"]
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def per_user_causal_sdpa(q, k, v, user_offsets_cpu):
    """Per-user causal SDPA - eliminates padding and mask construction.

    Args:
        q, k, v: [1, n_heads, total_tokens, head_dim]
        user_offsets_cpu: [num_users+1] CPU LongTensor with cumulative token counts

    Returns:
        attn_out: [1, n_heads, total_tokens, head_dim]
    """
    num_users = user_offsets_cpu.size(0) - 1
    offsets = user_offsets_cpu.tolist()

    # Path 1: flash_attn_varlen_func (single kernel, zero padding)
    if _FLASH_ATTN_AVAILABLE:
        # flash_attn expects [total_seq_len, num_heads, head_dim] in FP16
        q_fa = q.squeeze(0).permute(1, 0, 2).half().contiguous()
        k_fa = k.squeeze(0).permute(1, 0, 2).half().contiguous()
        v_fa = v.squeeze(0).permute(1, 0, 2).half().contiguous()
        max_seqlen = int((user_offsets_cpu[1:] - user_offsets_cpu[:-1]).max())
        cu_seqlens = user_offsets_cpu.to(torch.int32)
        out_fa = _flash_attn_varlen_func(
            q_fa, k_fa, v_fa,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        # [S, H, D] -> [1, H, S, D]
        return out_fa.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

    # Path 2: per-user SDPA with is_causal=True (no mask construction)
    outputs = []
    for i in range(num_users):
        start = offsets[i]
        end = offsets[i + 1]
        if start == end:
            continue
        q_i = q[:, :, start:end, :]
        k_i = k[:, :, start:end, :]
        v_i = v[:, :, start:end, :]
        attn_out_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        outputs.append(attn_out_i)

    return torch.cat(outputs, dim=2)


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

    def forward(self, x):
        # x: [B,S,D]
        logits = self.w_g(x)  # [B,S,E]

        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * 0.1

        probs = torch.softmax(logits, dim=-1)  # [B,S,E]

        topk_score, topk_idx = torch.topk(probs, self.k, dim=-1)  # [B,S,k]

        return topk_idx, topk_score, probs

class SMoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            Expert(d_model, dim_ff) for _ in range(num_experts)
        ])

        self.gate = TopKGate(d_model, num_experts, k=k)
        self._v109_dense_weights_ready = False

    def _v109_prepare_dense_weights(self):
        """Pre-stack expert weights for dense batched GEMM (called once after model.to(device))."""
        if self._v109_dense_weights_ready:
            return
        # fc1.weight: [dim_ff, D] → transpose to [D, dim_ff] for matmul
        self._all_fc1_w = torch.stack([e.fc1.weight.t() for e in self.experts])  # [E, D, dim_ff]
        self._all_fc1_b = torch.stack([e.fc1.bias for e in self.experts])        # [E, dim_ff]
        # fc2.weight: [D, dim_ff] → transpose to [dim_ff, D] for matmul
        self._all_fc2_w = torch.stack([e.fc2.weight.t() for e in self.experts])  # [E, dim_ff, D]
        self._all_fc2_b = torch.stack([e.fc2.bias for e in self.experts])        # [E, D]
        self._v109_dense_weights_ready = True

    def forward(self, x):
        # x: [B,S,D]
        B, S, D = x.shape

        topk_idx, topk_score, probs = self.gate(x)

        if V109_DENSE_SMOE and not self.training and self._v109_dense_weights_ready:
            # V109: Dense batched SMoE — all tokens through all experts, gate weights zero out non-topk
            x_flat = x.reshape(-1, D)  # [B*S, D]
            NS = B * S

            if V110_EINSUM_FUSION:
                # V110: Build gate_weights and use einsum for fused weighted sum
                # Avoids allocating [E, NS, D] intermediate product tensor
                gate_weights = x_flat.new_zeros(NS, self.num_experts)
                gate_weights.scatter_(1, topk_idx.reshape(NS, self.k), topk_score.reshape(NS, self.k))

                # Batched expert forward: [1, NS, D] @ [E, D, dim_ff] → [E, NS, dim_ff]
                hidden = torch.matmul(x_flat.unsqueeze(0), self._all_fc1_w) + self._all_fc1_b.unsqueeze(1)
                hidden = F.relu(hidden)
                # [E, NS, dim_ff] @ [E, dim_ff, D] → [E, NS, D]
                out_all = torch.matmul(hidden, self._all_fc2_w) + self._all_fc2_b.unsqueeze(1)

                # Fused weighted sum via einsum: [E, NS, D] × [E, NS] → [NS, D]
                # This contracts over expert dimension in a single GEMM-like operation
                # instead of element-wise multiply + sum which allocates [E, NS, D] temp
                out = torch.einsum('end,en->nd', out_all, gate_weights.t())
                out = out.reshape(B, S, D)
            else:
                # V109 original path
                gate_weights = x_flat.new_zeros(NS, self.num_experts)
                gate_weights.scatter_(1, topk_idx.reshape(NS, self.k), topk_score.reshape(NS, self.k))

                hidden = torch.matmul(x_flat.unsqueeze(0), self._all_fc1_w) + self._all_fc1_b.unsqueeze(1)
                hidden = F.relu(hidden)
                out_all = torch.matmul(hidden, self._all_fc2_w) + self._all_fc2_b.unsqueeze(1)

                out = (out_all * gate_weights.t().unsqueeze(-1)).sum(dim=0)
                out = out.reshape(B, S, D)
        else:
            # Original sparse path
            out = torch.zeros_like(x)
            out_flat = out.reshape(-1, D)
            x_flat = x.reshape(-1, D)
            token_idx_flat = (
                torch.arange(B * S, device=x.device)
                .unsqueeze(1)
                .expand(-1, self.k)
                .reshape(-1)
            )
            expert_idx_flat = topk_idx.reshape(-1)
            expert_score_flat = topk_score.reshape(-1, 1)

            order = torch.argsort(expert_idx_flat)
            expert_idx_sorted = expert_idx_flat[order]
            token_idx_sorted = token_idx_flat[order]
            expert_score_sorted = expert_score_flat[order]

            counts = torch.bincount(expert_idx_sorted, minlength=self.num_experts)
            starts = torch.cumsum(counts, dim=0) - counts

            for i in range(self.num_experts):
                count = int(counts[i].item())
                if count == 0:
                    continue
                start = int(starts[i].item())
                end = start + count
                cur_token_idx = token_idx_sorted[start:end]
                cur_weight = expert_score_sorted[start:end]
                selected_x = x_flat.index_select(0, cur_token_idx)
                expert_out = self.experts[i](selected_x)
                out_flat.index_add_(0, cur_token_idx, expert_out * cur_weight)

        if V29_NO_MOELOSS_RETURN_ENABLED and not self.training:
            return out, None
        if V27_NO_AUXLOSS_ENABLED and not self.training:
            moe_loss = x.new_zeros(())
        else:
            importance = probs.sum(dim=(0,1))  # [E]
            moe_loss = (importance.std() / (importance.mean() + 1e-6))

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
        if V110_REMOVE_DEAD_FFN:
            # V110: ffn1/ffn2 are never used when SMoE is active — skip to save memory & load time
            self.ffn1 = nn.ModuleList()
            self.ffn2 = nn.ModuleList()
        else:
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
        self.v14_fp16_weights_active = False

    def forward(self, x, extension=None, user_offsets_cpu=None):
        x = x.unsqueeze(0)
        if self.v14_fp16_weights_active:
            x = x.half()
        B, S, D = x.shape

        return_moe_loss = not (V29_NO_MOELOSS_RETURN_ENABLED and not self.training)
        moe_loss_total = 0.0 if return_moe_loss else None
        for i in range(self.num_layers):
            residual = x
            x = self.norm1[i](x)
            qkv = self.qkv_proj[i](x)
            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            if V103_PER_USER_CAUSAL_SDPA and user_offsets_cpu is not None:
                attn_out = per_user_causal_sdpa(q, k, v, user_offsets_cpu)
            else:
                attn_out = self.attention_fn(q, k, v, extension)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            x = residual + self.out_proj[i](attn_out)
            residual = x
            x = self.norm2[i](x)

            moe_result = self.moe[i](x)
            if return_moe_loss:
                moe_out, moe_loss = moe_result
            else:
                moe_out = moe_result[0] if isinstance(moe_result, tuple) else moe_result

            x = residual + moe_out

            if return_moe_loss:
                moe_loss_total = moe_loss_total + moe_loss

        if return_moe_loss:
            return x, moe_loss_total
        return x, x.new_zeros(())


class CTRModel(nn.Module):
    def __init__(self, rep_encoder, seq_encoder, d_model):
        super().__init__()
        self.rep_encoder = rep_encoder
        self.seq_encoder = seq_encoder
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def get_sequence_causal_mask(self, seq_info):
        lengths = seq_info[1:] - seq_info[:-1]
        user_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=seq_info.device),
            lengths.view(-1),
        )
        same_user = user_ids.view(1, -1).eq(user_ids.view(-1, 1))
        causal = torch.ones(
            (user_ids.numel(), user_ids.numel()),
            device=seq_info.device,
            dtype=torch.bool,
        ).tril()
        return same_user & causal

    def forward(self, batch):
        seq_input = self.rep_encoder(batch)
        if V103_PER_USER_CAUSAL_SDPA:
            # V110: if offsets stayed CPU, no need for .cpu() call
            if V110_OFFSETS_STAY_CPU:
                user_offsets_cpu = batch["user_offsets"]  # already CPU
            else:
                user_offsets_cpu = batch["user_offsets"].cpu()
            encoder_result = self.seq_encoder(
                x=seq_input,
                user_offsets_cpu=user_offsets_cpu,
            )
        else:
            seq_mask = self.get_sequence_causal_mask(batch["user_offsets"])
            encoder_result = self.seq_encoder(
                x=seq_input,
                extension={"mask": seq_mask.unsqueeze(0).unsqueeze(0)},
            )
        if isinstance(encoder_result, tuple):
            encoder_output, moe_loss = encoder_result
        else:
            encoder_output = encoder_result
            moe_loss = None
        encoder_output_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
        encoder_output = encoder_output.float()
        pred = self.linear(encoder_output)
        pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
        if moe_loss is None:
            moe_loss = pred_logits.new_zeros(())
        return pred_logits, moe_loss


def apply_v14_fp16_weights(model):
    """Convert the compute-heavy Transformer/SMoE block to FP16.

    RepEncoder and final output projection stay FP32 to reduce CTR calibration risk.
    """
    converted = 0
    if not V14_FP16_WEIGHTS_ENABLED:
        _info("[INFO] V14 FP16 weight path disabled by GRAB_V14_FP16_WEIGHTS")
        return converted
    for module in model.seq_encoder.modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.half()
            converted += 1
    model.seq_encoder.v14_fp16_weights_active = converted > 0
    _info(f"[INFO] V14 FP16 weight path enabled: converted {converted} seq_encoder Linear/LayerNorm modules")
    return converted


def apply_v15_fp16_deep(model):
    """Extend V14 by converting RepEncoder's projection Linear to FP16."""
    if not V15_FP16_DEEP_ENABLED:
        _info("[INFO] V15 FP16 deep path disabled by GRAB_V15_FP16_DEEP")
        return 0
    model.rep_encoder.linear.half()
    model.rep_encoder.v15_fp16_linear_active = True
    _info("[INFO] V15 FP16 deep path enabled: converted RepEncoder.linear")
    return 1


def apply_v16_fp16_embedding(model):
    """Convert the large RepEncoder embedding table to FP16 while preserving FP32 outputs.

    Embedding lookup is the largest memory-bandwidth consumer. The lookup table is
    stored as FP16, then gathered embeddings are cast back to FP32 before pooling
    and LayerNorm to keep the RepEncoder interface aligned with V14/V15.
    """
    if not V16_FP16_EMB_ENABLED:
        _info("[INFO] V16 FP16 embedding path disabled by GRAB_V16_FP16_EMB")
        return 0
    model.rep_encoder.emb.half()
    model.rep_encoder.v16_fp16_embedding_active = True
    _info("[INFO] V16 FP16 embedding path enabled: converted RepEncoder.emb")
    return 1


def apply_v24_sparse_embedding(model):
    """Replace per-slot segment_reduce with a single embedding_bag aggregation."""
    if not V24_SPARSE_EMB_ENABLED:
        _info("[INFO] V24 sparse embedding path disabled by GRAB_V24_SPARSE_EMB")
        return 0
    if not hasattr(F, "embedding_bag"):
        _warn("[WARNING] V24 sparse embedding path unavailable: torch.nn.functional.embedding_bag missing")
        return 0
    model.rep_encoder.v24_sparse_embedding_active = True
    _info("[INFO] V24 sparse embedding path enabled: fused slot aggregation with embedding_bag")
    return 1


# ============================================================
# 模型加载入口
# ============================================================

def load_model(ckpt_path=None, device='cuda:0'):
    """加载模型并返回，供 evaluation.py 调用。

    Args:
        device: 推理设备（默认 'cuda:0'）
        ckpt_path: checkpoint 文件路径，默认使用 infer.py 同目录下的 ckpt.pt

    Returns:
        (model, device) 元组
    """
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

    dev = resolve_device(device, require_cuda=False)

    # 加载 checkpoint
    # 若需要加载自定义修改的权重，请修改 479-488行逻辑，强制使用你文件夹中的权重
    # 测评系统默认使用原始官方权重
    if ckpt_path is None:
        ckpt_path = Path(__file__).parent / 'ckpt.pt'
    else:
        ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        import gc
        import inspect

        load_kwargs = {
            'map_location': 'cpu',
            'weights_only': False,
        }
        try:
            if 'mmap' in inspect.signature(torch.load).parameters:
                load_kwargs['mmap'] = True
        except (TypeError, ValueError):
            pass

        _info(f"[INFO] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, **load_kwargs)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt
        epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'

        # V110: If dead FFN removed, filter out ffn1/ffn2 keys from state_dict
        if V110_REMOVE_DEAD_FFN:
            ffn_keys = [k for k in state_dict if 'ffn1.' in k or 'ffn2.' in k]
            for k in ffn_keys:
                del state_dict[k]
            if ffn_keys:
                _info(f"[INFO] V110: Removed {len(ffn_keys)} dead FFN keys from state_dict")

        try:
            if 'assign' in inspect.signature(model.load_state_dict).parameters:
                model.load_state_dict(state_dict, assign=True)
            else:
                model.load_state_dict(state_dict)
        except (TypeError, ValueError):
            model.load_state_dict(state_dict)

        del state_dict
        del ckpt
        gc.collect()
        _info(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={epoch})")
    else:
        _warn(f"[WARNING] Checkpoint {ckpt_path} not found, using random weights")

    apply_v14_fp16_weights(model)
    apply_v15_fp16_deep(model)
    apply_v16_fp16_embedding(model)
    apply_v24_sparse_embedding(model)
    model.to(dev)
    model.eval()

    # V109: Pre-stack expert weights for dense batched SMoE (must be after .to(dev) and FP16 conversion)
    if V109_DENSE_SMOE:
        for moe in model.seq_encoder.moe:
            moe._v109_prepare_dense_weights()
        _info("[INFO] V109 dense SMoE: expert weights pre-stacked for batched GEMM")

    _info(f"[INFO] Model ready. Device: {dev}")

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
        _warn('[WARNING] only one class present in labels, AUC is not defined, returning 0.5')
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
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint 文件路径，默认使用同目录下的 ckpt.pt')
    parser.add_argument('--device', type=str, default=None, help='推理设备，默认自动选择 cuda:0')
    parser.add_argument('--allow-cpu', action='store_true', help='允许在无 CUDA 环境下退回 CPU（已知不稳定，仅调试用）')
    args = parser.parse_args()

    configure_runtime()
    runtime_device = resolve_device(args.device, require_cuda=not args.allow_cpu)

    cur_path = Path(__file__).parent.absolute()
    ref_dir = cur_path / 'dataset'
    history_dir = ref_dir / 'history'
    input_file = ref_dir / 'test.csv'
    output_file = Path('predict.txt')
    label_file = ref_dir / 'label_data.txt'

    # ----- 数据加载，优先从缓存读取 -----
    batches_cache_dir = ref_dir / 'cached_batches'
    shard_files = []
    test_loader = None

    if batches_cache_dir.exists() and any(batches_cache_dir.glob('shard_*.pt')):
        _info(f'[INFO] loading cached batch shards from {batches_cache_dir}')
        shard_files = sorted(
            batches_cache_dir.glob('shard_*.pt'),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        _info(f'[INFO] found {len(shard_files)} cached batch shards')
    else:
        _info('[INFO] start loading data from CSV')
        history_files = sorted(history_dir.glob('*.csv')) if history_dir.exists() else []
        all_files = history_files + [input_file]

        item_dict, user_seq = load_sample_files(sample_files_list=all_files)
        test_pred_logids = load_logids_from_file(input_file)
        _info(f'[INFO] Test pred logids count: {len(test_pred_logids)}')

        max_feasign_per_slot = {1: 2}
        test_dataset = CTRUserDataset(
            item_dict, user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=test_pred_logids,
        )
        _info(f'[INFO] num_users={test_dataset.num_users}, '
              f'total_samples={test_dataset.total_samples}, '
              f'pred_samples={len(test_pred_logids)}, '
              f'max_sign_id={test_dataset.max_sign_id}')

        test_loader = DataLoader(
            test_dataset,
            batch_size=50,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(test_dataset.max_slot_id),
        )
        _info('[INFO] using streaming DataLoader batches (no on-disk cache)')

    _info('[INFO] data loading done')

    # ----- 加载模型 -----
    model, dev = load_model(ckpt_path=args.ckpt, device=str(runtime_device))

    # ----- 推理 -----
    _info('*' * 20 + ' start inference ' + '*' * 20)
    collector = ResultCollector(
        enabled=V32_CHUNKED_COLLECT_ENABLED,
        chunk_preds=V32_COLLECT_CHUNK_PREDS,
    )
    time_sum = 0.0
    use_cuda = dev.type == "cuda"
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_cuda and hasattr(torch.cuda, "amp")
        else nullcontext()
    )

    def run_one_batch(batch):
        nonlocal time_sum
        if V30_CPU_METADATA_ENABLED:
            pred_mask_cpu = batch["pred_mask"].bool()
            model_batch = move_model_inputs_to_device(batch, dev, non_blocking=use_cuda)
            cpu_logids = batch["logid"].cpu() if torch.is_tensor(batch["logid"]) else batch["logid"]
            pred_count = int(pred_mask_cpu.sum().item())
            pred_total = int(pred_mask_cpu.numel())
            use_pred_indices = (
                V38_ADAPT_PRED_INDICES_ENABLED
                and (pred_count == 0 or (pred_total > 0 and (pred_count / pred_total) < V38_PRED_INDICES_MAX_DENSITY))
            )
            if use_pred_indices:
                pred_indices_cpu = pred_mask_cpu.nonzero(as_tuple=False).flatten()
                pred_indices_device = pred_indices_cpu.to(dev, non_blocking=use_cuda)
                pred_mask_device = None
            else:
                pred_indices_cpu = None
                pred_indices_device = None
                pred_mask_device = pred_mask_cpu.to(dev, non_blocking=use_cuda)
        else:
            batch = move_batch_to_device(batch, dev, non_blocking=use_cuda)
            pred_mask_device = batch["pred_mask"].bool()
            pred_mask_cpu = pred_mask_device.cpu()
            pred_indices_cpu = None
            pred_indices_device = None
            model_batch = batch
            cpu_logids = batch["logid"]

        t_start = time.time()
        model_out = model(model_batch)
        logits = model_out[0] if isinstance(model_out, tuple) else model_out
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        time_sum += time.time() - t_start

        if pred_indices_cpu is not None and V30_CPU_METADATA_ENABLED:
            collector.add_indices(cpu_logids, probs, pred_indices_cpu, pred_indices_device)
        else:
            collector.add(cpu_logids, probs, pred_mask_device, logid_mask=pred_mask_cpu)

    with torch.inference_mode(), autocast_ctx:
        if shard_files:
            shard_idx = 0
            for sf in shard_files:
                shard_batches = torch.load(sf, map_location='cpu', weights_only=False)
                _info(f'[INFO] loaded {len(shard_batches)} batches from {sf.name}')
                for batch in _progress(shard_batches, desc=f"Inference {sf.name}", leave=False):
                    run_one_batch(batch)
                del shard_batches
                shard_idx += 1
                if use_cuda and shard_idx % 3 == 0:
                    torch.cuda.empty_cache()
            _info(f'[INFO] inference consumed {len(shard_files)} cached shards')
        else:
            for batch in _progress(test_loader, desc="Inference"):
                run_one_batch(batch)

    _info(f'[INFO] inference time: {round(time_sum, 4)}s', always=True)
    _info('*' * 20 + ' end inference ' + '*' * 20)

    # ----- 按 test.csv 顺序写预测文件 -----
    all_logids, all_probs = collector.result()
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
    _info(f'[INFO] predictions written to {output_file}, total: {len(test_logids_in_order)}', always=True)

    # ----- 打分 -----
    if label_file.exists():
        result = _cal_score(output_file, label_file, default_latency=time_sum)
        _info(f'[INFO] AUC:            {result["auc"]:.6f}', always=True)
        _info(f'[INFO] PCOC:           {result["pcoc"]:.6f}', always=True)
        _info(f'[INFO] Latency:        {result["latency"]:.4f}s', always=True)
        _info(f'[INFO] score_latency:  {result["score_latency"]:.6f}', always=True)
        _info(f'[INFO] score_model:    {result["score_model"]:.6f}', always=True)
        _info(f'[INFO] score_all:      {result["score_all"]:.6f}', always=True)
        return result
    else:
        _warn(f'[WARNING] label file {label_file} not found, skipping scoring')
        return None


if __name__ == '__main__':
    main()
