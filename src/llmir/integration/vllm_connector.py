"""
vLLM V1 KV connector that offloads prefix K/V through :class:`LLMIRKVStorage`.

Usage::

    from llmir.integration.vllm_connector import register_llmir_vllm_connector

    register_llmir_vllm_connector()
    # KVTransferConfig(kv_connector="LLMIRConnector",
    #                  kv_connector_module_path="llmir.integration.vllm_connector", ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Type

from llmir.integration.vllm_kv_storage import (
    LLMIRKVStorage,
    LLMIRKVStorageConfig,
    align_to_block_size,
)

logger = logging.getLogger(__name__)

CONNECTOR_NAME = "LLMIRConnector"
CONNECTOR_MODULE_PATH = "llmir.integration.vllm_connector"

__all__ = [
    "CONNECTOR_MODULE_PATH",
    "CONNECTOR_NAME",
    "LLMIRConnector",
    "build_kv_transfer_extra_config",
    "is_vllm_connector_available",
    "register_llmir_vllm_connector",
]


def is_vllm_connector_available() -> bool:
    """Return True when vLLM V1 KV connector base classes are importable."""
    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.base  # noqa: F401

        return True
    except ImportError:
        return False


def build_kv_transfer_extra_config(
    storage_path: str = "/tmp/llmir_vllm_kv",
    min_prefix_length: int = 4,
) -> dict[str, Any]:
    """Extra config for ``KVTransferConfig.kv_connector_extra_config``."""
    return {
        "shared_storage_path": storage_path,
        "min_prefix_length": min_prefix_length,
    }


def register_llmir_vllm_connector() -> bool:
    """Register :data:`CONNECTOR_NAME` with vLLM's connector factory."""
    if LLMIRConnector is None:
        logger.warning(
            "Cannot register %s: vLLM KV connector V1 API not available",
            CONNECTOR_NAME,
        )
        return False
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        KVConnectorFactory.register_connector(
            CONNECTOR_NAME,
            CONNECTOR_MODULE_PATH,
            CONNECTOR_NAME,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to register %s: %s", CONNECTOR_NAME, exc)
        return False


def _define_connector_class() -> Optional[Type[Any]]:
    try:
        import torch
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            KVConnectorMetadata,
            KVConnectorRole,
        )
        from vllm.v1.core.sched.output import SchedulerOutput
    except ImportError:
        return None

    if TYPE_CHECKING:
        from vllm.config import VllmConfig
        from vllm.forward_context import ForwardContext
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        from vllm.v1.kv_cache_interface import KVCacheConfig
        from vllm.v1.request import Request

    @dataclass
    class LLMIRReqMeta:
        token_ids: torch.Tensor
        slot_mapping: torch.Tensor
        is_store: bool

        @staticmethod
        def make_meta(
            token_ids: List[int],
            block_ids: List[int],
            block_size: int,
            is_store: bool,
        ) -> "LLMIRReqMeta":
            valid = align_to_block_size(len(token_ids), block_size)
            token_ids_tensor = torch.tensor(token_ids)[:valid]
            block_ids_tensor = torch.tensor(block_ids)
            num_blocks = block_ids_tensor.shape[0]
            offsets = torch.arange(0, block_size)
            slot_mapping = (
                offsets.reshape((1, block_size))
                + block_ids_tensor.reshape((num_blocks, 1)) * block_size
            )
            slot_mapping = slot_mapping.flatten()[:valid]
            return LLMIRReqMeta(
                token_ids=token_ids_tensor,
                slot_mapping=slot_mapping,
                is_store=is_store,
            )

    @dataclass
    class LLMIRConnectorMetadata(KVConnectorMetadata):
        requests: List[LLMIRReqMeta] = field(default_factory=list)

        def add_request(
            self,
            token_ids: List[int],
            block_ids: List[int],
            block_size: int,
            is_store: bool,
        ) -> None:
            self.requests.append(
                LLMIRReqMeta.make_meta(token_ids, block_ids, block_size, is_store)
            )

    class LLMIRConnector(KVConnectorBase_V1):
        """vLLM V1 connector backed by :class:`LLMIRKVStorage`."""

        def __init__(
            self,
            vllm_config: "VllmConfig",
            role: KVConnectorRole,
            kv_cache_config: "KVCacheConfig",
        ):
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
            self._block_size = vllm_config.cache_config.block_size
            extra = self._kv_transfer_config.get_from_extra_config
            self._storage = LLMIRKVStorage(
                LLMIRKVStorageConfig(
                    storage_path=extra("shared_storage_path", "/tmp/llmir_vllm_kv"),
                    min_prefix_length=int(extra("min_prefix_length", 4)),
                )
            )
            self._requests_need_load: dict[str, Any] = {}

        def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
        ) -> tuple[int | None, bool]:
            token_ids = list(request.prompt_token_ids or [])
            matched = self._storage.longest_cached_prefix_length(
                token_ids, block_size=self._block_size
            )
            min_len = self._storage.config.min_prefix_length
            if matched < min_len:
                return 0, False
            num_tokens_to_check = align_to_block_size(
                len(token_ids) - 1, self._block_size
            )
            if matched < num_tokens_to_check:
                return 0, False
            external = num_tokens_to_check - num_computed_tokens
            return max(external, 0), False

        def update_state_after_alloc(
            self,
            request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int,
        ) -> None:
            if num_external_tokens > 0:
                self._requests_need_load[request.request_id] = request

        def build_connector_meta(
            self, scheduler_output: SchedulerOutput
        ) -> LLMIRConnectorMetadata:
            meta = LLMIRConnectorMetadata()
            for new_req in scheduler_output.scheduled_new_reqs:
                token_ids = list(new_req.prompt_token_ids or [])
                if new_req.req_id in self._requests_need_load:
                    meta.add_request(
                        token_ids,
                        new_req.block_ids[0],
                        self._block_size,
                        is_store=False,
                    )
                elif (
                    self._storage.longest_cached_prefix_length(
                        token_ids, block_size=self._block_size
                    )
                    < self._storage.config.min_prefix_length
                ):
                    meta.add_request(
                        token_ids,
                        new_req.block_ids[0],
                        self._block_size,
                        is_store=True,
                    )
            self._requests_need_load.clear()
            return meta

        def start_load_kv(
            self, forward_context: "ForwardContext", **kwargs: Any
        ) -> None:
            import torch

            metadata = self._get_connector_metadata()
            if not isinstance(metadata, LLMIRConnectorMetadata):
                return
            if forward_context.attn_metadata is None:
                return

            for request in metadata.requests:
                if request.is_store:
                    continue
                token_list = [int(t) for t in request.token_ids.tolist()]
                for layer_name, layer in forward_context.no_compile_layers.items():
                    kv_cache_layer = getattr(layer, "kv_cache", None)
                    if kv_cache_layer is None:
                        continue
                    kv_np = self._storage.load_layer_kv(token_list, layer_name)
                    kv_t = torch.from_numpy(kv_np).to(
                        device=kv_cache_layer.device, dtype=kv_cache_layer.dtype
                    )
                    _inject_kv_slice(
                        kv_cache_layer,
                        kv_t,
                        request.slot_mapping,
                        self._block_size,
                    )

        def wait_for_layer_load(self, layer_name: str) -> None:
            return

        def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: Any,
            attn_metadata: Any,
            **kwargs: Any,
        ) -> None:
            metadata = self._get_connector_metadata()
            if not isinstance(metadata, LLMIRConnectorMetadata):
                return
            for request in metadata.requests:
                if not request.is_store:
                    continue
                extracted = _extract_kv_slice(
                    kv_layer, request.slot_mapping, self._block_size
                )
                token_list = [int(t) for t in request.token_ids.tolist()]
                self._storage.store_layer_kv(
                    token_list,
                    layer_name,
                    extracted.detach().cpu().numpy(),
                )

        def wait_for_save(self) -> None:
            return

    return LLMIRConnector


def _inject_kv_slice(dst: Any, src: Any, slot_mapping: Any, block_size: int) -> None:
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    if dst.dim() == 3:
        dst.reshape(-1, dst.shape[-1])[slot_mapping, ...] = src
    else:
        dst[block_idxs, :, offsets] = src


def _extract_kv_slice(kv_layer: Any, slot_mapping: Any, block_size: int) -> Any:
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    if kv_layer.dim() == 3:
        return kv_layer.reshape(-1, kv_layer.shape[-1])[slot_mapping, ...]
    return kv_layer[block_idxs, :, offsets]


LLMIRConnector = _define_connector_class()
