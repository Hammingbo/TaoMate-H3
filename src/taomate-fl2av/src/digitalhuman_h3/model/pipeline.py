# SPDX-License-Identifier: Apache-2.0
"""End-to-end in-process MiniMax H3 T2VA/FL2VA and image Ref2VA pipeline."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import time
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlparse

import torch

from digitalhuman_h3.distributed import ParallelContext, use_parallel_context

from .components import partition_root, read_component_config
from .condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
from .denoise import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .dit import MiniMaxH3DiT
from ..denoise_schedule import select_time_shift_sigmas
from .geometry import MiniMaxH3Geometry, resolve_geometry, resolve_reference_image_size
from .output import (
    H3LatentDecoder,
    save_h3_mp4,
    save_h3_video_mp4,
    save_h3_video_mp4_from_rgb24_shards,
    write_h3_rgb24_shard,
)
from .packed_sequence import minimax_h3_packed_sequence, minimax_h3_packed_sequence_ref2va
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import (
    minimax_h3_multi_image_presentation,
    minimax_h3_text_only_ids,
)
from .progressive_denoise import minimax_h3_progressive_denoise_loop
from .task_profiles import MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES
from .text_encoder import MiniMaxH3TextEncoder, load_minimax_h3_text_weights
from .weight_loading import load_minimax_h3_dit_weights
from ..video_vae_receipt import (
    build_parallel_video_vae_temporal_output_receipt,
    require_single_host_temporal_window_dp,
)
from ..progressive_resolution import (
    ProgressiveResolutionPlan,
    coupled_progressive_video_noise_rows,
    lift_video_state_rows,
    resize_video_state_rows,
)


NATIVE_EXECUTED_STAGES = (
    "NativeRequestValidation",
    "NativeTextEncoding",
    "NativeVisualEncoding",
    "NativeLatentPreparation",
    "NativeTimestepPreparation",
    "NativeDenoising",
)

H3ModelForward = Callable[
    [Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]
]


class H3DenoiseObserver(Protocol):
    def begin(self, call: Mapping[str, Any]) -> None: ...

    def on_step(
        self, step: int, video_rows: torch.Tensor, audio_rows: torch.Tensor
    ) -> None: ...


@dataclass
class H3NativeGenerationResult:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    last_video_condition_rows: torch.Tensor
    geometry: MiniMaxH3Geometry
    output_path: Path | None
    executed_stages: tuple[str, ...]
    progressive_resolution: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class H3S2VPairedTrajectoryResult:
    low_geometry: MiniMaxH3Geometry
    high_geometry: MiniMaxH3Geometry
    executed_stages: tuple[str, ...]
    paired_resolution: Mapping[str, Any]


def _local_image_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"} or parsed.netloc not in {"", "localhost"}:
        raise ValueError("native H3 currently requires a worker-local image path")
    value = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"H3 image must be a regular local file: {path}")
    return path


def _cover_crop(image: Any, *, width: int, height: int) -> Any:
    from PIL import Image

    scale = max(width / image.width, height / image.height)
    resized_w = max(width, int(round(image.width * scale)))
    resized_h = max(height, int(round(image.height * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _prepare_keyframes(
    conditions: Sequence[Mapping[str, Any]],
    *,
    geometry: MiniMaxH3Geometry,
) -> tuple[list[Any], tuple[int, ...]]:
    from PIL import Image, ImageOps

    signature: list[int] = []
    images: list[Any] = []
    for index, condition in enumerate(conditions):
        if condition.get("type") != "image" or condition.get("role") != "keyframe":
            raise ValueError("FL2VA conditions must be image/keyframe entries")
        frame_index = condition.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("FL2VA keyframe frame_index must be an integer")
        uri = condition.get("uri", condition.get("path"))
        if not isinstance(uri, str) or not uri:
            raise ValueError("FL2VA keyframe requires uri or path")
        with Image.open(_local_image_path(uri)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if index == 0:
            image = image.resize(
                (geometry.width, geometry.height), Image.Resampling.LANCZOS
            )
        else:
            image = _cover_crop(
                image, width=geometry.width, height=geometry.height
            )
        signature.append(frame_index)
        images.append(image)
    signature_tuple = tuple(signature)
    if signature_tuple not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "FL2VA frame_index signature must be [0], [-1], or [0,-1]"
        )
    return images, signature_tuple


def _normalize_keyframe_latent(
    latent: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> torch.Tensor:
    latent = latent.cpu().float()
    if latent.ndim == 4:
        latent = latent[None]
    if latent.ndim != 5 or int(latent.shape[1]) != 24:
        raise ValueError(f"unexpected H3 keyframe latent shape {tuple(latent.shape)}")
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32).view(
        1, 24, 1, 1, 1
    )
    std = torch.tensor(config["latents_std"], dtype=torch.float32).view(
        1, 24, 1, 1, 1
    )
    return minimax_h3_patchify_video_latent(
        (latent - mean) / std, patch_size=(1, 2, 2)
    ).float()


def _prepare_reference_images(
    conditions: Sequence[Mapping[str, Any]],
) -> list[Any]:
    from PIL import Image, ImageOps

    if not 1 <= len(conditions) <= 9:
        raise ValueError("image Ref2VA requires 1 to 9 image/reference conditions")
    images = []
    for condition in conditions:
        if condition.get("type") != "image" or condition.get("role") != "reference":
            raise ValueError(
                "native Ref2VA currently supports image/reference conditions only; "
                "audio/video references and hybrid keyframes are not implemented"
            )
        if condition.get("frame_index") is not None:
            raise ValueError("Ref2VA reference images must not have frame_index")
        uri = condition.get("uri", condition.get("path"))
        if not isinstance(uri, str) or not uri:
            raise ValueError("Ref2VA reference image requires uri or path")
        with Image.open(_local_image_path(uri)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        size = resolve_reference_image_size(*image.size)
        images.append(image.resize(size, Image.Resampling.LANCZOS))
    return images


class MiniMaxH3NativePipeline:
    """One positive CFG-distilled H3 branch over explicit TP/Ulysses groups."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        variant: str,
        device: torch.device,
        parallel_context: ParallelContext,
        rank: int,
        world_size: int,
        defer_transformer: bool = False,
        defer_text_encoder: bool = False,
        transformer_load_w8a8_policy: str | None = None,
    ) -> None:
        if variant not in {"fl2va", "ref2va"}:
            raise ValueError("variant must be fl2va or ref2va")
        self.model_root = Path(model_root).resolve(strict=True)
        self.partition_root = partition_root(self.model_root, variant)
        self.variant = variant
        self.device = device
        self.parallel_context = parallel_context
        self.rank = rank
        self.world_size = world_size
        self.transformer_load_w8a8_policy = transformer_load_w8a8_policy
        self.tokenizer = self._load_tokenizer()
        self.processor = self._load_processor()
        with use_parallel_context(parallel_context):
            self.text_encoder: MiniMaxH3TextEncoder | None = None
            self.text_weight_receipt: Any | None = None
            if not defer_text_encoder:
                self.load_text_encoder()
            self.transformer: MiniMaxH3DiT | None = None
            self.dit_weight_receipt: Any | None = None
            if not defer_transformer:
                self.load_transformer()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        self.video_vae: Any | None = None
        self.audio_vae: Any | None = None
        self.output_decoder: H3LatentDecoder | None = None
        self._output_decoder_lock = threading.Lock()
        self.last_video_decode_receipt: dict[str, Any] | None = None
        self._component_timer: Any | None = None
        self._video_vae_inference = {
            "fp8_linears": False,
            "fp8_attention": False,
            "exact_fusions": False,
            "batched_tiles": False,
            "balanced_cpu_output": False,
            "temporal_window_dp": False,
            "temporal_window_microbatch_size": 1,
            "gpu_rgb24_output": False,
            "keep_on_device_after_decode": False,
            "decoder_tile_size": None,
            "decoder_tile_overlap": None,
        }

    def configure_inference_video_vae(
        self,
        *,
        fp8_linears: bool = False,
        fp8_attention: bool = False,
        exact_fusions: bool = False,
        batched_tiles: bool = False,
        balanced_cpu_output: bool = False,
        temporal_window_dp: bool = False,
        temporal_window_microbatch_size: int = 1,
        gpu_rgb24_output: bool = False,
        keep_on_device_after_decode: bool = False,
        decoder_tile_size: int | None = None,
        decoder_tile_overlap: int | None = None,
    ) -> None:
        if self.output_decoder is not None:
            raise RuntimeError("Video VAE inference options must be set before decode")
        if temporal_window_dp and not balanced_cpu_output:
            raise ValueError(
                "Video VAE temporal-window DP requires balanced CPU output"
            )
        if temporal_window_dp:
            require_single_host_temporal_window_dp(world_size=self.world_size)
        if temporal_window_microbatch_size < 1:
            raise ValueError(
                "Video VAE temporal-window microbatch size must be positive"
            )
        if temporal_window_microbatch_size > 1 and not temporal_window_dp:
            raise ValueError(
                "Video VAE temporal-window microbatching requires temporal-window DP"
            )
        if gpu_rgb24_output and not temporal_window_dp:
            raise ValueError(
                "Video VAE GPU RGB24 output requires temporal-window DP"
            )
        self._video_vae_inference = {
            "fp8_linears": bool(fp8_linears),
            "fp8_attention": bool(fp8_attention),
            "exact_fusions": bool(exact_fusions),
            "batched_tiles": bool(batched_tiles),
            "balanced_cpu_output": bool(balanced_cpu_output),
            "temporal_window_dp": bool(temporal_window_dp),
            "temporal_window_microbatch_size": int(
                temporal_window_microbatch_size
            ),
            "gpu_rgb24_output": bool(gpu_rgb24_output),
            "keep_on_device_after_decode": bool(keep_on_device_after_decode),
            "decoder_tile_size": decoder_tile_size,
            "decoder_tile_overlap": decoder_tile_overlap,
        }

    def set_component_timer(self, timer: Any | None) -> None:
        self._component_timer = timer

    def load_transformer(self) -> MiniMaxH3DiT:
        """Materialize the DiT after prompt encoding when GPU memory is tight."""

        if self.transformer is None:
            with use_parallel_context(self.parallel_context):
                transformer = MiniMaxH3DiT.allocate(
                    self.device,
                    parallel_context=self.parallel_context,
                    load_time_w8a8_policy=self.transformer_load_w8a8_policy,
                )
                receipt = load_minimax_h3_dit_weights(
                    transformer, self.partition_root / "transformer"
                )
            transformer.eval()
            self.transformer = transformer
            self.dit_weight_receipt = receipt
        return self.transformer

    def load_text_encoder(self) -> MiniMaxH3TextEncoder:
        if self.text_encoder is None:
            with use_parallel_context(self.parallel_context):
                self.text_encoder = MiniMaxH3TextEncoder.allocate(
                    self.partition_root / "text_encoder",
                    device=self.device,
                    context=self.parallel_context,
                )
                self.text_weight_receipt = load_minimax_h3_text_weights(
                    self.text_encoder, self.partition_root / "text_encoder"
                )
            self.text_encoder.eval()
        return self.text_encoder

    @property
    def modules(self) -> dict[str, Any]:
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "video_vae": self.video_vae,
            "audio_vae": self.audio_vae,
        }

    def _load_tokenizer(self) -> Any:
        transformers = __import__("transformers", fromlist=["AutoTokenizer"])
        return transformers.AutoTokenizer.from_pretrained(
            str(self.partition_root / "tokenizer"), local_files_only=True
        )

    def _load_processor(self) -> Any:
        transformers = __import__("transformers", fromlist=["AutoProcessor"])
        return transformers.AutoProcessor.from_pretrained(
            str(self.partition_root / "processor"), local_files_only=True
        )

    def _move_text_encoder(self, device: torch.device | str) -> None:
        if self.text_encoder is None:
            raise RuntimeError("native H3 text encoder was released")
        if self.text_encoder.device != torch.device(device):
            self.text_encoder.to(device)

    def _encode_text(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._move_text_encoder(self.device)
        if not images:
            ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
            tags = torch.ones(int(ids.shape[0]), dtype=torch.long)
            hidden = self.text_encoder.encode_ids(ids)
            return hidden, tags
        vision = self.processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values = vision["pixel_values"]
        image_grid_thw = vision["image_grid_thw"]
        merge = int(self.processor.image_processor.merge_size) ** 2
        counts = [int(row.prod().item()) // merge for row in image_grid_thw]
        ids, tags = minimax_h3_multi_image_presentation(
            self.tokenizer, prompt=prompt, image_token_counts=counts
        )
        hidden = self.text_encoder.encode_ids(
            ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return hidden, tags

    def _encode_keyframe_rows(
        self,
        images: Sequence[Any],
        *,
        geometry: MiniMaxH3Geometry,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if not images:
            return None
        shapes = [
            (1, geometry.video_latent_h, geometry.video_latent_w) for _ in images
        ]
        rows = self._encode_image_rows(images, condition_shapes=shapes)
        return minimax_h3_imgvid_cond_noise_aug_rows(
            rows,
            condition_shapes=shapes,
            target_latent_t=geometry.video_latent_t,
            imgvid_cond_num_frames=len(images),
            generator=generator,
            noise_aug=0.999,
        )

    def _encode_reference_rows(
        self,
        images: Sequence[Any],
        *,
        geometry: MiniMaxH3Geometry,
        seed: int,
    ) -> torch.Tensor:
        shapes = [(1, image.height // 16, image.width // 16) for image in images]
        clean_rows = self._encode_image_rows(images, condition_shapes=shapes).cpu()
        # The official reference recipe restarts the same CPU RNG for EACH
        # material; it must not advance the target video/audio RNG streams.
        augmented = []
        offset = 0
        for shape in shapes:
            count = (shape[1] // 2) * (shape[2] // 2)
            augmented.append(minimax_h3_imgvid_cond_noise_aug_rows(
                clean_rows[offset : offset + count],
                condition_shapes=[shape],
                target_latent_t=geometry.video_latent_t,
                imgvid_cond_num_frames=len(images),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                noise_aug=0.999,
            ))
            offset += count
        return torch.cat(augmented, dim=0)

    def _encode_image_rows(
        self,
        images: Sequence[Any],
        *,
        condition_shapes: Sequence[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Shared seed-42 sampled VAE encode; no target-canvas assumption."""
        total_rows = sum(t * (h // 2) * (w // 2) for t, h, w in condition_shapes)
        rows = torch.empty(total_rows, 96, dtype=torch.float32, device=self.device)
        error: str | None = None
        try:
            from .components import configure_bundled_video_vae_tile_parallel

            # The bundled VAE stores its tile-parallel topology in module-global
            # state. Every rank must enter encode_images together; otherwise rank
            # zero waits in the VAE barrier while its peers enter the pipeline
            # error collective below.
            decoder = self._load_output_decoder()
            self.video_vae = decoder.video_vae
            configure_bundled_video_vae_tile_parallel(
                self.video_vae,
                group=(
                    torch.distributed.group.WORLD if self.world_size > 1 else None
                ),
                rank=self.rank,
                world_size=self.world_size,
            )
            self.video_vae.to(self.device, dtype=torch.float32).eval()
            config = read_component_config(self.partition_root / "video_vae")
            encoded: list[torch.Tensor] = []
            devices = [self.device] if self.device.type == "cuda" else []
            for image in images:
                with torch.random.fork_rng(devices=devices):
                    torch.default_generator.manual_seed(42)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed(42)
                    latent = self.video_vae.encode_images(
                        image, use_fp16_latent=True
                    )[0]
                if self.rank == 0:
                    encoded.append(
                        _normalize_keyframe_latent(latent, config=config)
                    )
            if self.rank == 0:
                host_rows = torch.cat(encoded, dim=0)
                if tuple(host_rows.shape) != (total_rows, 96):
                    raise ValueError(
                        f"image condition rows {tuple(host_rows.shape)} != {(total_rows, 96)}"
                    )
                rows.copy_(host_rows.to(self.device))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.video_vae is not None:
                self.video_vae.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if self.world_size > 1:
            errors: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(errors, error)
            failure = next((item for item in errors if item), None)
            if failure:
                raise RuntimeError(f"native H3 image encode failed: {failure}")
            torch.distributed.broadcast(rows, src=0)
        elif error:
            raise RuntimeError(f"native H3 image encode failed: {error}")
        return rows

    @staticmethod
    def _initial_video_noise(
        geometry: MiniMaxH3Geometry,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raw_video = torch.randn(
            1,
            24,
            geometry.video_latent_t,
            geometry.video_latent_h,
            geometry.video_latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        return minimax_h3_patchify_video_latent(
            raw_video, patch_size=(1, 2, 2)
        )

    @staticmethod
    def _initial_noise(
        geometry: MiniMaxH3Geometry,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = MiniMaxH3NativePipeline._initial_video_noise(
            geometry, generator=generator
        )
        audio = MiniMaxH3NativePipeline._initial_audio_noise(
            geometry,
            generator=generator,
        )
        return video, audio

    @staticmethod
    def _initial_audio_noise(
        geometry: MiniMaxH3Geometry,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            geometry.audio_latent_t * 2,
            32,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

    def _offload_text(self) -> None:
        if self.text_encoder is None or self.text_encoder.device.type == "cpu":
            return
        self.text_encoder.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _collective_decode_error(self, error: str | None) -> str | None:
        if self.world_size == 1:
            return error
        errors: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(errors, error)
        return next((str(item) for item in errors if item is not None), None)

    def _load_output_decoder(self) -> H3LatentDecoder:
        with self._output_decoder_lock:
            if self.output_decoder is None:
                self.output_decoder = H3LatentDecoder(
                    video_vae_root=self.partition_root / "video_vae",
                    audio_vae_root=self.partition_root / "audio_vae",
                    video_parallel_group=(
                        torch.distributed.group.WORLD
                        if self.world_size > 1
                        else None
                    ),
                    video_parallel_rank=self.rank,
                    video_parallel_world_size=self.world_size,
                    load_audio=self.rank == 0,
                )
                options = self._video_vae_inference
                if options["batched_tiles"]:
                    self.output_decoder.video_vae.model.stack_tiling = True
                if options["decoder_tile_size"] is not None:
                    self.output_decoder.video_vae.model.decoder_tile_size = int(
                        options["decoder_tile_size"]
                    )
                if options["decoder_tile_overlap"] is not None:
                    self.output_decoder.video_vae.model.decoder_tile_overlap_min = int(
                        options["decoder_tile_overlap"]
                    )
                if options["fp8_linears"]:
                    self.output_decoder.enable_video_vae_fp8_linears()
                if options["fp8_attention"]:
                    self.output_decoder.enable_video_vae_fp8_attention()
                if options["exact_fusions"]:
                    self.output_decoder.enable_video_vae_exact_fusions()
                if options["temporal_window_dp"]:
                    self.output_decoder.enable_video_vae_temporal_window_dp(
                        microbatch_size=options[
                            "temporal_window_microbatch_size"
                        ],
                        gpu_rgb24_output=options["gpu_rgb24_output"],
                    )
                self.output_decoder.enable_video_vae_cpu_temporal_output(
                    balanced_across_ranks=options["balanced_cpu_output"]
                )
            return self.output_decoder

    def _save_balanced_video_shards(
        self,
        *,
        frames: torch.Tensor,
        rank_receipts: Sequence[Mapping[str, Any]],
        output_path: Path,
        fps: int,
        video_filter: str | None,
        published_video_frames: int | None,
        video_crf: int,
        verify_video_decode: bool,
    ) -> Path:
        root_payload: list[Any] = [None]
        if self.rank == 0:
            try:
                shared_root = Path("/dev/shm")
                if not shared_root.is_dir():
                    raise RuntimeError(
                        "balanced Video VAE output requires local /dev/shm"
                    )
                root_payload[0] = {
                    "path": tempfile.mkdtemp(
                        prefix="digitalhuman-h3-balanced-vae-",
                        dir=shared_root,
                    ),
                    "error": None,
                }
            except Exception as exc:
                root_payload[0] = {
                    "path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(root_payload, src=0)
        root_result = root_payload[0]
        if not isinstance(root_result, dict) or root_result.get("error"):
            detail = (
                root_result.get("error")
                if isinstance(root_result, dict)
                else "invalid shared temporary directory receipt"
            )
            raise RuntimeError(str(detail))
        shard_root = Path(str(root_result["path"]))

        local_shard = None
        local_error = None
        try:
            local_receipt = rank_receipts[self.rank]
            local_shard = write_h3_rgb24_shard(
                shard_root / f"rank_{self.rank:05d}.rgb24",
                frames=frames,
                frame_start=int(local_receipt["retained_frame_start"]),
            )
        except Exception as exc:
            local_error = (
                f"rank {self.rank} Video VAE RGB24 shard: "
                f"{type(exc).__name__}: {exc}"
            )
        shard_error = self._collective_decode_error(local_error)

        gathered_shards: list[Any] = [None] * self.world_size
        if shard_error is None:
            if self.world_size > 1:
                torch.distributed.all_gather_object(gathered_shards, local_shard)
            else:
                gathered_shards[0] = local_shard

        saved = None
        publication_error = shard_error
        if self.rank == 0:
            try:
                if publication_error is None:
                    saved = save_h3_video_mp4_from_rgb24_shards(
                        output_path,
                        shards=gathered_shards,
                        fps=fps,
                        video_filter=video_filter,
                        published_frames=published_video_frames,
                        crf=video_crf,
                        verify_decode=verify_video_decode,
                    )
            except Exception as exc:
                publication_error = (
                    "rank 0 balanced Video VAE publication: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                shutil.rmtree(shard_root, ignore_errors=True)

        payload: list[Any] = [
            {
                "error": publication_error,
                "path": str(saved) if saved is not None else None,
            }
            if self.rank == 0
            else None
        ]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[0]["error"] is not None:
            raise RuntimeError(str(payload[0]["error"]))
        return Path(str(payload[0]["path"]))

    def _decode_and_save_parallel(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor | None,
        output_path: Path | None,
        lossless_audio_path: Path | None = None,
        fps: int,
        error_prefix: str,
        decode_audio: bool = True,
        video_filter: str | None = None,
        published_video_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        balanced_cpu_output = bool(
            self._video_vae_inference["balanced_cpu_output"]
        )
        if balanced_cpu_output and decode_audio:
            raise ValueError(
                "balanced Video VAE CPU output is available for video-only "
                "one-shot publication"
            )
        if balanced_cpu_output and output_path is None:
            raise ValueError("balanced Video VAE CPU output requires output_path")
        self.last_video_decode_receipt = None
        decoder = None
        local_error = None
        try:
            decoder = self._load_output_decoder()
        except Exception as exc:
            local_error = (
                f"rank {self.rank} VAE initialization: "
                f"{type(exc).__name__}: {exc}"
            )
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"{error_prefix}: {decode_error}")
        assert decoder is not None

        frames = None
        local_error = None
        try:
            timer_context = (
                contextlib.nullcontext()
                if self._component_timer is None
                else self._component_timer.region("video_vae")
            )
            with timer_context:
                frames = decoder.decode_video(
                    video_latents=video_latents,
                    device=self.device,
                    offload_after_decode=not bool(
                        self._video_vae_inference[
                            "keep_on_device_after_decode"
                        ]
                    ),
                )
        except Exception as exc:
            local_error = (
                f"rank {self.rank} video VAE: {type(exc).__name__}: {exc}"
            )
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"{error_prefix}: {decode_error}")
        assert frames is not None

        local_receipt = decoder.last_video_decode_receipt
        if self.world_size > 1:
            rank_receipts: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(rank_receipts, local_receipt)
        else:
            rank_receipts = [local_receipt]
        self.last_video_decode_receipt = (
            build_parallel_video_vae_temporal_output_receipt(rank_receipts)
        )

        if balanced_cpu_output:
            assert output_path is not None
            return self._save_balanced_video_shards(
                frames=frames,
                rank_receipts=rank_receipts,
                output_path=output_path,
                fps=fps,
                video_filter=video_filter,
                published_video_frames=published_video_frames,
                video_crf=video_crf,
                verify_video_decode=verify_video_decode,
            )

        saved = None
        local_error = None
        if self.rank == 0:
            try:
                if output_path is None:
                    raise ValueError("decoded inference requires output_path")
                if decode_audio:
                    if audio_latents is None:
                        raise ValueError(
                            "audio latents are required when decode_audio is enabled"
                        )
                    timer_context = (
                        contextlib.nullcontext()
                        if self._component_timer is None
                        else self._component_timer.region("audio_vae")
                    )
                    with timer_context:
                        waveform, sample_rate = decoder.decode_audio(
                            audio_latents=audio_latents,
                            device=self.device,
                            offload_after_decode=True,
                        )
                    saved = save_h3_mp4(
                        output_path,
                        frames=frames,
                        audio=waveform,
                        fps=fps,
                        sample_rate=sample_rate,
                        lossless_audio_path=lossless_audio_path,
                    )
                else:
                    saved = save_h3_video_mp4(
                        output_path,
                        frames=frames,
                        fps=fps,
                        video_filter=video_filter,
                        published_frames=published_video_frames,
                        crf=video_crf,
                        verify_decode=verify_video_decode,
                    )
            except Exception as exc:
                local_error = (
                    "rank 0 AV publication: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            del frames

        payload: list[Any] = [
            {"error": local_error, "path": str(saved) if saved else None}
            if self.rank == 0
            else None
        ]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        publication_error = payload[0]["error"]
        if publication_error is not None:
            raise RuntimeError(f"{error_prefix}: {publication_error}")
        return Path(payload[0]["path"])

    def release_text_encoder(self) -> None:
        """Release the prompt encoder after all training prompts are cached."""

        self.text_encoder = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_denoise_branch(
        self,
        *,
        geometry: MiniMaxH3Geometry,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        keyframe_rows: torch.Tensor | None,
        keyframe_signature: tuple[int, ...] | None,
        refined_prompt_embeddings: torch.Tensor | None = None,
        defer_prompt_refiner_to_model: bool = False,
        audio_is_condition: bool = False,
    ) -> tuple[MiniMaxH3DenoiseBranch, torch.Tensor | None]:
        if self.transformer is None:
            raise RuntimeError("native H3 transformer is not loaded")
        packed = minimax_h3_packed_sequence(
            text_len=int(text_embeddings.shape[0]),
            latent_t=geometry.video_latent_t,
            latent_h=geometry.video_latent_h,
            latent_w=geometry.video_latent_w,
            audio_t=geometry.audio_latent_t,
            audio_is_condition=audio_is_condition,
            include_keyframe_cond=keyframe_rows is not None,
            keyframe_frame_indices=(
                list(keyframe_signature) if keyframe_signature is not None else None
            ),
            frame_count=(geometry.frame_count if keyframe_rows is not None else None),
        )
        token_tags = packed["token_tags"]
        token_tags[packed["text_pos"].view(-1)] = text_tags.view(-1)
        positive = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=token_tags,
            device=self.device,
            parallel_context=self.parallel_context,
        )
        positive._digitalhuman_h3_raw_text_embeddings = text_embeddings
        positive._digitalhuman_h3_raw_text_tags = text_tags
        refined = refined_prompt_embeddings
        if defer_prompt_refiner_to_model and refined is not None:
            raise ValueError(
                "deferred prompt refinement cannot reuse external refined embeddings"
            )
        if refined is None and not defer_prompt_refiner_to_model:
            refined = self.transformer.refine_prompt_embeds(
                positive.static_kwargs["prompt_embeds"],
                positive.static_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"],
                device=self.device,
            )
        if refined is not None:
            positive.static_kwargs["prompt_embeds"] = refined
            positive.static_kwargs["refined_prompt_embeds_length"] = int(
                refined.shape[0]
            )
        positive.static_kwargs["rope_cache"] = self.transformer.build_rope_cache(
            positive.static_kwargs["img_position_ids"], device=self.device
        )
        return positive, refined

    @torch.no_grad()
    def generate_s2v_paired_trajectory(
        self,
        *,
        prompt: str,
        seed: int,
        conditions: Sequence[Mapping[str, Any]],
        duration_seconds: float,
        audio_condition_rows: torch.Tensor,
        plan: ProgressiveResolutionPlan,
        observer: Any,
        num_inference_steps: int = 50,
        flow_shift: float = 12.0,
    ) -> H3S2VPairedTrajectoryResult:
        """Extract one fixed-480 and one progressive-768 S2V teacher pair.

        The full 480p branch is shared by both artifacts. The 768p endpoint
        still comes from its complete native coupled-noise trajectory; the
        lifted low state is retained only as the progressive transition input.
        Ground-truth audio is a fixed condition at every model forward.
        """

        if self.transformer is None:
            raise RuntimeError("native H3 transformer is not loaded")
        if num_inference_steps != 50:
            raise ValueError("S2V paired ODE extraction requires 50 sigma states")
        if len(conditions) != 1:
            raise ValueError("S2V paired ODE extraction requires one first frame")

        low_geometry = resolve_geometry(
            duration_seconds=duration_seconds,
            width=plan.low.width,
            height=plan.low.height,
        )
        high_geometry = resolve_geometry(
            duration_seconds=duration_seconds,
            width=plan.high.width,
            height=plan.high.height,
        )
        if (
            low_geometry.frame_count != high_geometry.frame_count
            or low_geometry.video_latent_t != high_geometry.video_latent_t
            or low_geometry.audio_latent_t != high_geometry.audio_latent_t
        ):
            raise RuntimeError("S2V paired low/high temporal geometry differs")
        expected_audio_shape = (high_geometry.audio_latent_t * 2, 32)
        if tuple(audio_condition_rows.shape) != expected_audio_shape:
            raise ValueError(
                "S2V audio condition rows differ: "
                f"expected {expected_audio_shape}, got {tuple(audio_condition_rows.shape)}"
            )

        low_images, keyframe_signature = _prepare_keyframes(
            conditions,
            geometry=low_geometry,
        )
        high_images, high_signature = _prepare_keyframes(
            conditions,
            geometry=high_geometry,
        )
        if keyframe_signature != (0,) or high_signature != keyframe_signature:
            raise RuntimeError("S2V paired keyframe signature differs")

        with use_parallel_context(self.parallel_context):
            condition_generator = torch.Generator(device="cpu").manual_seed(seed)
            text_embeddings, text_tags = self._encode_text(prompt=prompt, images=())
            low_keyframe_rows = self._encode_keyframe_rows(
                low_images,
                geometry=low_geometry,
                generator=condition_generator,
            )
            high_keyframe_rows = self._encode_keyframe_rows(
                high_images,
                geometry=high_geometry,
                generator=condition_generator,
            )
            self._offload_text()

            low_positive, refined = self._build_denoise_branch(
                geometry=low_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                keyframe_rows=low_keyframe_rows,
                keyframe_signature=keyframe_signature,
                audio_is_condition=True,
            )
            high_positive, _ = self._build_denoise_branch(
                geometry=high_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                keyframe_rows=high_keyframe_rows,
                keyframe_signature=keyframe_signature,
                refined_prompt_embeddings=refined,
                audio_is_condition=True,
            )
            (
                initial_low_target,
                initial_high_target,
                noise_coupling,
            ) = coupled_progressive_video_noise_rows(
                temporal_length=low_geometry.video_latent_t,
                source_canvas=plan.low,
                target_canvas=plan.high,
                noise_seed=seed,
            )
            initial_low_video = self._initial_video_rows(
                low_positive,
                initial_low_target,
                keyframe_rows=low_keyframe_rows,
            )
            initial_high_video = self._initial_video_rows(
                high_positive,
                initial_high_target,
                keyframe_rows=high_keyframe_rows,
            )
            audio_condition = audio_condition_rows.detach().to(
                device="cpu",
                dtype=torch.float32,
            ).contiguous()
            video_sigmas = select_time_shift_sigmas(
                num_steps=num_inference_steps,
                shift_scale=flow_shift,
                state_indices=None,
            )

            observer.begin(
                {
                    "low_positive": low_positive,
                    "high_positive": high_positive,
                    "raw_text_embeddings": text_embeddings,
                    "text_tags": text_tags,
                    "initial_low_video_rows": initial_low_video,
                    "initial_high_video_rows": initial_high_video,
                    "audio_condition_rows": audio_condition,
                    "low_keyframe_cond_rows": low_keyframe_rows,
                    "high_keyframe_cond_rows": high_keyframe_rows,
                    "keyframe_signature": keyframe_signature,
                    "target_video_latent_t": low_geometry.video_latent_t,
                    "video_sigmas": video_sigmas,
                    "progressive_resolution": plan.as_dict(),
                }
            )

            low_state_033: torch.Tensor | None = None

            def low_step_callback(
                step: int,
                video_rows: torch.Tensor,
                current_audio_rows: torch.Tensor,
            ) -> None:
                nonlocal low_state_033
                if step + 1 == plan.switch_state_index:
                    low_state_033 = (
                        video_rows[low_positive.video_target_slice]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                observer.on_low_step(step, video_rows, current_audio_rows)

            minimax_h3_denoise_loop(
                model=self.transformer,
                positive=low_positive,
                initial_video_rows=initial_low_video,
                initial_audio_rows=audio_condition,
                keyframe_cond_rows=low_keyframe_rows,
                sigmas_video=video_sigmas,
                sigmas_audio=video_sigmas,
                device=self.device,
                audio_ref_rows=audio_condition,
                on_step=low_step_callback,
            )
            if low_state_033 is None:
                raise RuntimeError("S2V paired low state 33 was not captured")
            lifted_state_033, lift_receipt = lift_video_state_rows(
                low_state_033,
                temporal_length=low_geometry.video_latent_t,
                source_canvas=plan.low,
                target_canvas=plan.high,
                video_sigma=video_sigmas[plan.switch_state_index],
                noise_seed=seed,
            )
            observer.on_resolution_lift(
                plan.switch_state_index,
                lifted_state_033,
                audio_condition,
                lift_receipt,
            )
            minimax_h3_denoise_loop(
                model=self.transformer,
                positive=high_positive,
                initial_video_rows=initial_high_video,
                initial_audio_rows=audio_condition,
                keyframe_cond_rows=high_keyframe_rows,
                sigmas_video=video_sigmas,
                sigmas_audio=video_sigmas,
                device=self.device,
                audio_ref_rows=audio_condition,
                on_step=observer.on_high_step,
            )

        return H3S2VPairedTrajectoryResult(
            low_geometry=low_geometry,
            high_geometry=high_geometry,
            executed_stages=NATIVE_EXECUTED_STAGES,
            paired_resolution={
                **plan.as_dict(),
                "lift": lift_receipt,
                "initial_noise_coupling": noise_coupling,
                "fixed_480_model_forwards": num_inference_steps - 1,
                "progressive_low_model_forwards_reused": plan.switch_state_index,
                "progressive_high_model_forwards": num_inference_steps - 1,
                "total_model_forwards": 2 * (num_inference_steps - 1),
                "separate_run_model_forwards": (
                    (num_inference_steps - 1)
                    + plan.switch_state_index
                    + (num_inference_steps - 1)
                ),
                "high_target_source": (
                    "native_high_full_trajectory_from_coupled_iid_noise"
                ),
            },
        )

    @staticmethod
    def _initial_video_rows(
        positive: MiniMaxH3DenoiseBranch,
        target_rows: torch.Tensor,
        *,
        keyframe_rows: torch.Tensor | None,
    ) -> torch.Tensor:
        if keyframe_rows is None:
            return target_rows
        expanded = torch.zeros(
            int(positive.img_pos.numel()),
            96,
            dtype=torch.float32,
            device=target_rows.device,
        )
        expanded[positive.update_mask] = target_rows
        return expanded

    @torch.no_grad()
    def _generate_progressive(
        self,
        *,
        prompt: str,
        seed: int,
        video_noise_seed: int | None,
        audio_noise_seed: int | None,
        task: str,
        conditions: Sequence[Mapping[str, Any]],
        duration_seconds: float,
        num_inference_steps: int,
        sigma_state_indices: tuple[int, ...] | None,
        flow_shift: float,
        audio_flow_shift: float,
        trajectory_only: bool,
        output_path: Path | None,
        lossless_audio_path: Path | None,
        observer: H3DenoiseObserver | None,
        precomputed_text: tuple[torch.Tensor, torch.Tensor] | None,
        clean_keyframe_rows: torch.Tensor | None,
        plan: ProgressiveResolutionPlan,
        defer_prompt_refiner_to_model: bool,
        decode_video: bool,
        decode_audio: bool,
        model_forward: H3ModelForward | None = None,
    ) -> H3NativeGenerationResult:
        if self.transformer is None:
            raise RuntimeError("native H3 transformer is not loaded")
        low_geometry = resolve_geometry(
            duration_seconds=duration_seconds,
            width=plan.low.width,
            height=plan.low.height,
        )
        high_geometry = resolve_geometry(
            duration_seconds=duration_seconds,
            width=plan.high.width,
            height=plan.high.height,
        )
        if (
            low_geometry.frame_count != high_geometry.frame_count
            or low_geometry.video_latent_t != high_geometry.video_latent_t
            or low_geometry.audio_latent_t != high_geometry.audio_latent_t
        ):
            raise RuntimeError("progressive low/high temporal geometry differs")

        low_images: list[Any] = []
        high_images: list[Any] = []
        keyframe_signature: tuple[int, ...] | None = None
        if task == "fl2va":
            if conditions:
                low_images, keyframe_signature = _prepare_keyframes(
                    conditions, geometry=low_geometry
                )
                high_images, high_signature = _prepare_keyframes(
                    conditions, geometry=high_geometry
                )
                if high_signature != keyframe_signature:
                    raise RuntimeError("progressive keyframe signatures differ")
            else:
                keyframe_signature = (0,)
        # Distilled conditioned inference keeps Qwen text-only while the same
        # media conditions independently provide VAE keyframe rows.

        with use_parallel_context(self.parallel_context):
            condition_generator = torch.Generator(device="cpu").manual_seed(seed)
            if precomputed_text is None:
                text_embeddings, text_tags = self._encode_text(
                    prompt=prompt,
                    images=high_images,
                )
            else:
                text_embeddings, text_tags = precomputed_text
                if (
                    text_embeddings.ndim != 2
                    or int(text_embeddings.shape[1]) != 5120
                    or text_tags.ndim != 1
                    or int(text_tags.shape[0]) != int(text_embeddings.shape[0])
                ):
                    raise ValueError("precomputed H3 text feature shape differs")
            low_keyframe_rows = self._encode_keyframe_rows(
                low_images,
                geometry=low_geometry,
                generator=condition_generator,
            )
            high_keyframe_rows = self._encode_keyframe_rows(
                high_images,
                geometry=high_geometry,
                generator=condition_generator,
            )
            if clean_keyframe_rows is not None:
                expected_high_rows = plan.high.spatial_tokens_per_video_latent
                if tuple(clean_keyframe_rows.shape) != (expected_high_rows, 96):
                    raise ValueError(
                        "direct progressive keyframe rows must use the high canvas"
                    )
                high_clean_rows = clean_keyframe_rows.detach().to(
                    device="cpu", dtype=torch.float32
                )
                low_clean_rows = resize_video_state_rows(
                    high_clean_rows,
                    temporal_length=1,
                    source_canvas=plan.high,
                    target_canvas=plan.low,
                )
                low_keyframe_rows = minimax_h3_imgvid_cond_noise_aug_rows(
                    low_clean_rows,
                    condition_shapes=[
                        (1, low_geometry.video_latent_h, low_geometry.video_latent_w)
                    ],
                    target_latent_t=low_geometry.video_latent_t,
                    imgvid_cond_num_frames=1,
                    generator=condition_generator,
                    noise_aug=0.999,
                )
                high_keyframe_rows = minimax_h3_imgvid_cond_noise_aug_rows(
                    high_clean_rows,
                    condition_shapes=[
                        (1, high_geometry.video_latent_h, high_geometry.video_latent_w)
                    ],
                    target_latent_t=high_geometry.video_latent_t,
                    imgvid_cond_num_frames=1,
                    generator=condition_generator,
                    noise_aug=0.999,
                )
            self._offload_text()

            low_positive, refined = self._build_denoise_branch(
                geometry=low_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                keyframe_rows=low_keyframe_rows,
                keyframe_signature=keyframe_signature,
                defer_prompt_refiner_to_model=defer_prompt_refiner_to_model,
            )
            (
                initial_video_target,
                progressive_high_initial_video_target,
                initial_noise_coupling,
            ) = coupled_progressive_video_noise_rows(
                temporal_length=low_geometry.video_latent_t,
                source_canvas=plan.low,
                target_canvas=plan.high,
                noise_seed=(seed if video_noise_seed is None else video_noise_seed),
            )
            # Progressive video noise is indexed per temporal latent so each
            # streaming block can reproduce the full-clip random source. Use
            # the next unused temporal substream for audio instead of
            # restarting at the video seed.
            if audio_noise_seed is None:
                audio_generator = torch.Generator(device="cpu").manual_seed(
                    seed + low_geometry.video_latent_t * 1_000_003
                )
            else:
                # Base10 audio follows the high-resolution video RNG draw.
                audio_generator = torch.Generator(device="cpu").manual_seed(audio_noise_seed)
                self._initial_video_noise(high_geometry, generator=audio_generator)
            initial_audio = self._initial_audio_noise(
                low_geometry,
                generator=audio_generator,
            )
            initial_video = self._initial_video_rows(
                low_positive,
                initial_video_target,
                keyframe_rows=low_keyframe_rows,
            )
            sigmas_video = select_time_shift_sigmas(
                num_steps=num_inference_steps,
                shift_scale=flow_shift,
                state_indices=sigma_state_indices,
            )
            sigmas_audio = select_time_shift_sigmas(
                num_steps=num_inference_steps,
                shift_scale=audio_flow_shift,
                state_indices=sigma_state_indices,
            )
            retained_indices = (
                tuple(range(num_inference_steps))
                if sigma_state_indices is None
                else tuple(sigma_state_indices)
            )
            try:
                switch_position = retained_indices.index(plan.switch_state_index)
            except ValueError as exc:
                raise ValueError(
                    "progressive sigma schedule must contain state 33"
                ) from exc
            if switch_position <= 0 or switch_position >= len(retained_indices) - 1:
                raise ValueError("progressive state 33 must be an interior sigma point")

            if observer is not None:
                observer.begin(
                    {
                        "positive": low_positive,
                        "raw_text_embeddings": text_embeddings,
                        "text_tags": text_tags,
                        "initial_video_rows": initial_video,
                        "initial_audio_rows": initial_audio,
                        "keyframe_cond_rows": low_keyframe_rows,
                        "high_keyframe_cond_rows": high_keyframe_rows,
                        "keyframe_signature": keyframe_signature,
                        "target_video_latent_t": low_geometry.video_latent_t,
                        "audio_ref_rows": None,
                        "sigmas_video": sigmas_video,
                        "sigmas_audio": sigmas_audio,
                        "progressive_resolution": plan.as_dict(),
                    }
                )

            lift_seed = seed
            high_positive, _ = self._build_denoise_branch(
                geometry=high_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                keyframe_rows=high_keyframe_rows,
                keyframe_signature=keyframe_signature,
                refined_prompt_embeddings=refined,
                defer_prompt_refiner_to_model=defer_prompt_refiner_to_model,
            )
            native_high_initial_video_target: torch.Tensor | None = None
            high_positive._digitalhuman_h3_coupled_initial_video_target_rows = (
                progressive_high_initial_video_target
            )
            native_high_initial_audio: torch.Tensor | None = None
            if trajectory_only and observer is not None:
                native_high_initial_video_target = (
                    progressive_high_initial_video_target
                )
                native_high_initial_audio = initial_audio.clone()

            def resolution_lift_callback(
                lifted_target: torch.Tensor,
                current_audio_rows: torch.Tensor,
                lift_receipt: dict[str, Any],
            ) -> None:
                if observer is None:
                    return
                lift_callback = getattr(observer, "on_resolution_lift", None)
                if not callable(lift_callback):
                    raise RuntimeError(
                        "progressive ODE observer has no resolution-lift callback"
                    )
                lift_callback(
                    plan.switch_state_index,
                    lifted_target,
                    current_audio_rows,
                    lift_receipt,
                )

            def high_step_callback(
                local_step: int,
                video_rows: torch.Tensor,
                current_audio_rows: torch.Tensor,
            ) -> None:
                if observer is not None:
                    observer.on_step(
                        plan.switch_state_index + local_step,
                        video_rows,
                        current_audio_rows,
                    )

            progressive_result = minimax_h3_progressive_denoise_loop(
                model=self.transformer,
                model_forward=model_forward,
                low_positive=low_positive,
                high_positive=high_positive,
                initial_low_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                low_keyframe_cond_rows=low_keyframe_rows,
                high_keyframe_cond_rows=high_keyframe_rows,
                sigmas_video=sigmas_video,
                sigmas_audio=sigmas_audio,
                switch_position=switch_position,
                plan=plan,
                device=self.device,
                lift_noise_seed=lift_seed,
                native_high_initial_video_target_rows=(
                    native_high_initial_video_target
                ),
                native_high_initial_audio_rows=native_high_initial_audio,
                on_low_step=(None if observer is None else observer.on_step),
                on_resolution_lift=(
                    None if observer is None else resolution_lift_callback
                ),
                on_high_step=(None if observer is None else high_step_callback),
            )
            high_video_rows = progressive_result.video_rows
            audio_rows = progressive_result.audio_rows
            lift_receipt = progressive_result.lift_receipt
            target_video = high_video_rows[high_positive.video_target_slice]
            video_latents = minimax_h3_unpatchify_video_tokens(
                target_video,
                latent_shape=(
                    high_geometry.video_latent_t,
                    high_geometry.video_latent_h // 2,
                    high_geometry.video_latent_w // 2,
                    24,
                ),
                patch_size=(1, 2, 2),
            )
            target_audio = audio_rows[high_positive.audio_target_slice]
            audio_latents = minimax_h3_unpack_audio_tokens(
                target_audio,
                audio_t=high_geometry.audio_latent_t * 2,
                audio_channel=2,
            )

        saved = (
            None
            if trajectory_only or not decode_video
            else self._decode_and_save_parallel(
                video_latents=video_latents,
                audio_latents=audio_latents,
                output_path=output_path,
                lossless_audio_path=lossless_audio_path,
                fps=high_geometry.fps,
                error_prefix="native H3 progressive decode failed",
                decode_audio=decode_audio,
            )
        )
        return H3NativeGenerationResult(
            video_latents=video_latents,
            audio_latents=audio_latents,
            last_video_condition_rows=(
                target_video[
                    -(
                        high_geometry.video_latent_h
                        // 2
                        * (high_geometry.video_latent_w // 2)
                    ) :
                ]
                .detach()
                .cpu()
                .contiguous()
            ),
            geometry=high_geometry,
            output_path=saved,
            executed_stages=NATIVE_EXECUTED_STAGES,
            progressive_resolution={
                **plan.as_dict(),
                "lift": lift_receipt,
                "high_target_source": progressive_result.high_target_source,
                "low_model_forwards": progressive_result.low_model_forwards,
                "high_model_forwards": progressive_result.high_model_forwards,
                "initial_noise_coupling": initial_noise_coupling,
            },
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        task: str,
        conditions: Sequence[Mapping[str, Any]],
        duration_seconds: float,
        aspect_ratio: str,
        width: int | None,
        height: int | None,
        num_inference_steps: int,
        sigma_state_indices: tuple[int, ...] | None,
        flow_shift: float,
        audio_flow_shift: float,
        trajectory_only: bool,
        output_path: Path | None,
        observer: H3DenoiseObserver | None = None,
        precomputed_text: tuple[torch.Tensor, torch.Tensor] | None = None,
        clean_keyframe_rows: torch.Tensor | None = None,
        progressive_resolution: ProgressiveResolutionPlan | None = None,
        defer_prompt_refiner_to_model: bool = False,
        model_forward: H3ModelForward | None = None,
        lossless_audio_path: Path | None = None,
        decode_audio: bool = True,
        consistency_renoise: bool = False,
        decode_video: bool = True,
        video_noise_seed: int | None = None,
        audio_noise_seed: int | None = None,
    ) -> H3NativeGenerationResult:
        if self.transformer is None:
            raise RuntimeError("native H3 transformer is not loaded")
        if task not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("native H3 supports t2va, fl2va, and image ref2va")
        if task == "ref2va":
            if self.variant != "ref2va":
                raise ValueError("ref2va requires the released Ref2VA partition")
            if clean_keyframe_rows is not None or progressive_resolution is not None:
                raise ValueError("image Ref2VA does not accept direct keyframes or progressive resolution")
        elif self.variant != "fl2va":
            raise ValueError("t2va and fl2va require the FL2VA partition")
        if task == "t2va" and conditions:
            raise ValueError("t2va does not accept conditions")
        if task == "fl2va" and not conditions and clean_keyframe_rows is None:
            raise ValueError("fl2va requires one or two keyframes")
        if conditions and clean_keyframe_rows is not None:
            raise ValueError("choose media conditions or direct clean keyframe rows")
        if consistency_renoise and (
            num_inference_steps != 4
            or sigma_state_indices is not None
            or flow_shift != 5.0
            or audio_flow_shift != 5.0
        ):
            raise ValueError(
                "Stage2b consistency inference requires four model forwards, "
                "no retained-state indices, and shared AV flow shift 5"
            )
        if progressive_resolution is not None:
            if consistency_renoise:
                raise ValueError(
                    "Stage2b consistency re-noise is not defined for progressive inference"
                )
            return self._generate_progressive(
                prompt=prompt,
                seed=seed,
                video_noise_seed=video_noise_seed,
                audio_noise_seed=audio_noise_seed,
                task=task,
                conditions=conditions,
                duration_seconds=duration_seconds,
                num_inference_steps=num_inference_steps,
                sigma_state_indices=sigma_state_indices,
                flow_shift=flow_shift,
                audio_flow_shift=audio_flow_shift,
                trajectory_only=trajectory_only,
                output_path=output_path,
                lossless_audio_path=lossless_audio_path,
                observer=observer,
                precomputed_text=precomputed_text,
                clean_keyframe_rows=clean_keyframe_rows,
                plan=progressive_resolution,
                defer_prompt_refiner_to_model=defer_prompt_refiner_to_model,
                model_forward=model_forward,
                decode_video=decode_video,
                decode_audio=decode_audio,
            )
        geometry = resolve_geometry(
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
        )
        images: list[Any] = []
        keyframe_signature: tuple[int, ...] | None = None
        if task == "ref2va":
            images = _prepare_reference_images(conditions)
            if self.rank == 0:
                print(f"Ref2VA prepared references: {[image.size for image in images]}; "
                      f"target={geometry.width}x{geometry.height}, frames={geometry.frame_count}", flush=True)
        if task == "fl2va":
            if conditions:
                images, keyframe_signature = _prepare_keyframes(
                    conditions, geometry=geometry
                )
            else:
                keyframe_signature = (0,)
        # Distilled conditioned inference keeps Qwen text-only while the same
        # media conditions independently provide VAE keyframe rows.
        with use_parallel_context(self.parallel_context):
            request_generator = torch.Generator(device="cpu").manual_seed(seed)
            if precomputed_text is None:
                text_embeddings, text_tags = self._encode_text(
                    prompt=prompt, images=images
                )
            else:
                text_embeddings, text_tags = precomputed_text
                if (
                    text_embeddings.ndim != 2
                    or int(text_embeddings.shape[1]) != 5120
                    or text_tags.ndim != 1
                    or int(text_tags.shape[0]) != int(text_embeddings.shape[0])
                ):
                    raise ValueError("precomputed H3 text feature shape differs")
            if task == "ref2va":
                # Qwen and VAE receive the identical independently resized images.
                # Release Qwen's device residency before the large reference VAE encode.
                self._offload_text()
                if self.rank == 0:
                    print(f"Ref2VA Qwen encoding complete: {text_embeddings.shape[0]} tokens; encoding image latents", flush=True)
                keyframe_rows = self._encode_reference_rows(
                    images, geometry=geometry, seed=seed
                )
            else:
                keyframe_rows = self._encode_keyframe_rows(
                    images, geometry=geometry, generator=request_generator
                )
            if clean_keyframe_rows is not None:
                expected_rows = (geometry.video_latent_h // 2) * (
                    geometry.video_latent_w // 2
                )
                if tuple(clean_keyframe_rows.shape) != (expected_rows, 96):
                    raise ValueError(
                        "direct H3 keyframe rows do not match request geometry"
                    )
                keyframe_rows = minimax_h3_imgvid_cond_noise_aug_rows(
                    clean_keyframe_rows.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    condition_shapes=[
                        (
                            1,
                            geometry.video_latent_h,
                            geometry.video_latent_w,
                        )
                    ],
                    target_latent_t=geometry.video_latent_t,
                    imgvid_cond_num_frames=1,
                    generator=request_generator,
                    noise_aug=0.999,
                )
            self._offload_text()
            packed_args = dict(
                text_len=int(text_embeddings.shape[0]),
                latent_t=geometry.video_latent_t,
                latent_h=geometry.video_latent_h,
                latent_w=geometry.video_latent_w,
                audio_t=geometry.audio_latent_t,
            )
            if task == "ref2va":
                packed = minimax_h3_packed_sequence_ref2va(
                    **packed_args,
                    ref_blocks=[
                        {"kind": "image", "latent_h": image.height // 16,
                         "latent_w": image.width // 16} for image in images
                    ],
                )
            else:
                packed = minimax_h3_packed_sequence(
                    **packed_args,
                    include_keyframe_cond=keyframe_rows is not None,
                    keyframe_frame_indices=(
                        list(keyframe_signature) if keyframe_signature is not None else None
                    ),
                    frame_count=(geometry.frame_count if keyframe_rows is not None else None),
                )
            token_tags = packed["token_tags"]
            token_tags[packed["text_pos"].view(-1)] = text_tags.view(-1)
            positive = MiniMaxH3DenoiseBranch(
                packed=packed,
                text_embeddings=text_embeddings,
                token_tags=token_tags,
                device=self.device,
                parallel_context=self.parallel_context,
            )
            positive._digitalhuman_h3_raw_text_embeddings = text_embeddings
            positive._digitalhuman_h3_raw_text_tags = text_tags
            if not defer_prompt_refiner_to_model:
                refined = self.transformer.refine_prompt_embeds(
                    positive.static_kwargs["prompt_embeds"],
                    positive.static_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"],
                    device=self.device,
                )
                positive.static_kwargs["prompt_embeds"] = refined
                positive.static_kwargs["refined_prompt_embeds_length"] = int(
                    refined.shape[0]
                )
            positive.static_kwargs["rope_cache"] = self.transformer.build_rope_cache(
                positive.static_kwargs["img_position_ids"], device=self.device
            )
            initial_video, initial_audio = self._initial_noise(
                geometry, generator=request_generator
            )
            if task == "ref2va":
                # Released Ref2VA uses independent, identically seeded AV streams.
                initial_audio = self._initial_audio_noise(
                    geometry, generator=torch.Generator(device="cpu").manual_seed(
                        seed if audio_noise_seed is None else audio_noise_seed
                    )
                )
            elif audio_noise_seed is not None and (
                audio_noise_seed != seed or keyframe_rows is not None
            ):
                # Image noise augmentation advances request_generator. An
                # explicit teacher audio seed must not depend on image count.
                audio_generator = torch.Generator(device="cpu").manual_seed(
                    audio_noise_seed
                )
                self._initial_video_noise(geometry, generator=audio_generator)
                initial_audio = self._initial_audio_noise(
                    geometry, generator=audio_generator
                )
            if video_noise_seed is not None and video_noise_seed != seed:
                video_generator = torch.Generator(device="cpu").manual_seed(
                    video_noise_seed
                )
                initial_video = self._initial_video_noise(
                    geometry, generator=video_generator
                )
            if keyframe_rows is not None:
                expanded = torch.zeros(
                    int(positive.img_pos.numel()), 96, dtype=torch.float32
                )
                expanded[positive.update_mask] = initial_video
                initial_video = expanded
            schedule_points = (
                num_inference_steps + 1
                if consistency_renoise
                else num_inference_steps
            )
            sigmas_video = select_time_shift_sigmas(
                num_steps=schedule_points,
                shift_scale=flow_shift,
                state_indices=sigma_state_indices,
            )
            sigmas_audio = select_time_shift_sigmas(
                num_steps=schedule_points,
                shift_scale=audio_flow_shift,
                state_indices=sigma_state_indices,
            )
            if observer is not None:
                observer.begin(
                    {
                        "positive": positive,
                        "raw_text_embeddings": text_embeddings,
                        "text_tags": text_tags,
                        "initial_video_rows": initial_video,
                        "initial_audio_rows": initial_audio,
                        "keyframe_cond_rows": keyframe_rows,
                        "keyframe_signature": keyframe_signature,
                        "target_video_latent_t": geometry.video_latent_t,
                        "audio_ref_rows": None,
                        "sigmas_video": sigmas_video,
                        "sigmas_audio": sigmas_audio,
                    }
                )
            denoise_started = time.monotonic()

            def on_denoise_step(step, video, audio):
                if observer is not None:
                    observer.on_step(step, video, audio)
                if task == "ref2va" and self.rank == 0 and (
                    step == 0 or (step + 1) % 5 == 0 or step + 1 == len(sigmas_video) - 1
                ):
                    print(f"Ref2VA denoise {step + 1}/{len(sigmas_video) - 1}, "
                          f"elapsed={time.monotonic() - denoise_started:.1f}s", flush=True)

            video_rows, audio_rows = minimax_h3_denoise_loop(
                model=self.transformer,
                model_forward=model_forward,
                positive=positive,
                initial_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                keyframe_cond_rows=keyframe_rows,
                sigmas_video=sigmas_video,
                sigmas_audio=sigmas_audio,
                device=self.device,
                on_step=(on_denoise_step if task == "ref2va" else (None if observer is None else observer.on_step)),
                consistency_renoise=consistency_renoise,
                renoise_generator=(
                    request_generator if consistency_renoise else None
                ),
            )
            target_video = video_rows[positive.video_target_slice]
            video_latents = minimax_h3_unpatchify_video_tokens(
                target_video,
                latent_shape=(
                    geometry.video_latent_t,
                    geometry.video_latent_h // 2,
                    geometry.video_latent_w // 2,
                    24,
                ),
                patch_size=(1, 2, 2),
            )
            target_audio = audio_rows[positive.audio_target_slice]
            audio_latents = minimax_h3_unpack_audio_tokens(
                target_audio,
                audio_t=geometry.audio_latent_t * 2,
                audio_channel=2,
            )
        saved = (
            None
            if trajectory_only or not decode_video
            else self._decode_and_save_parallel(
                video_latents=video_latents,
                audio_latents=audio_latents,
                output_path=output_path,
                lossless_audio_path=lossless_audio_path,
                fps=geometry.fps,
                error_prefix="native H3 decode failed",
                decode_audio=decode_audio,
            )
        )
        return H3NativeGenerationResult(
            video_latents=video_latents,
            audio_latents=audio_latents,
            last_video_condition_rows=(
                target_video[
                    -(
                        (geometry.video_latent_h // 2)
                        * (geometry.video_latent_w // 2)
                    ) :
                ]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            ),
            geometry=geometry,
            output_path=saved,
            executed_stages=NATIVE_EXECUTED_STAGES,
        )


__all__ = [
    "H3DenoiseObserver",
    "H3NativeGenerationResult",
    "H3S2VPairedTrajectoryResult",
    "MiniMaxH3NativePipeline",
    "NATIVE_EXECUTED_STAGES",
]
