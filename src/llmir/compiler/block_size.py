"""Block size analysis (paper Algorithm 1) for KV cache micro-pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BlockSizeAnalysisResult:
    """Mirrors ``mlir::llm::BlockSizeAnalysisResult``."""

    optimal_block_size: int
    fragmentation_score: float
    gpu_utilization: float
    memory_alignment_score: float
    combined_score: float


def _fragmentation(seq_lengths: Sequence[int], block_size: int) -> float:
    if not seq_lengths:
        return 0.0
    total = 0.0
    for seq_len in seq_lengths:
        num_blocks = (seq_len + block_size - 1) // block_size
        allocated = num_blocks * block_size
        total += (allocated - seq_len) / allocated
    return total / len(seq_lengths)


def _gpu_utilization(block_size: int, *, warp_size: int = 32) -> float:
    warp_util = min(1.0, block_size / warp_size)
    occupancy = 1.0
    if block_size > 128:
        occupancy = 0.9
    if block_size > 256:
        occupancy = 0.75
    bank = 1.0 if block_size % 32 == 0 else 0.8
    return warp_util * occupancy * bank


def _memory_alignment(
    block_size: int, *, cache_line: int = 128, element_size: int = 2
) -> float:
    bytes_per_block = block_size * element_size
    if bytes_per_block % cache_line == 0:
        return 1.0
    if cache_line % bytes_per_block == 0 or bytes_per_block % (cache_line // 2) == 0:
        return 0.9
    return 0.7


def analyze_block_size(
    seq_lengths: Iterable[int],
    *,
    candidates: Sequence[int] = (16, 32, 64, 128, 256),
) -> BlockSizeAnalysisResult:
    """Return the highest-scoring block size for observed sequence lengths."""
    lengths = [int(s) for s in seq_lengths if int(s) > 0]
    if not lengths:
        return BlockSizeAnalysisResult(128, 0.15, 0.95, 1.0, 0.85)

    best = BlockSizeAnalysisResult(128, 1.0, 0.0, 0.0, -1.0)
    for block_size in candidates:
        frag = _fragmentation(lengths, block_size)
        gpu = _gpu_utilization(block_size)
        align = _memory_alignment(block_size)
        score = (1.0 - frag) * 0.4 + gpu * 0.35 + align * 0.25
        row = BlockSizeAnalysisResult(block_size, frag, gpu, align, score)
        if row.combined_score > best.combined_score:
            best = row
    return best


def optimize_block_size_attr(current: int, seq_lengths: Iterable[int]) -> int:
    """Pick optimal block size; leave ``current`` when already optimal."""
    optimal = analyze_block_size(seq_lengths).optimal_block_size
    return optimal
