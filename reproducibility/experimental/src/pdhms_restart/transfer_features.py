"""Label-independent ImageNet transfer features; no PaHaW backbone fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18

from .lpq import preprocess


IMAGE_SIZE = 224
FEATURE_DIMENSION = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
OFFICIAL_HASH_PREFIX = "f37072fd"


def configure_determinism(seed: int = 20260908, cpu_threads: int = 4) -> None:
    """Configure repeatable float32 inference on this recorded software/device stack."""
    # Set before any CUDA context or cuBLAS operation is initialized.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def array_sha256(values: np.ndarray) -> str:
    """Hash dtype, dimensions and C-order bytes, including integer state buffers."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": values.dtype.str, "shape": list(values.shape)},
                             sort_keys=True).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Stable content digest of every parameter and persistent buffer."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array_sha256(tensor.detach().cpu().numpy()).encode("ascii"))
    return digest.hexdigest()


def normalized_imagenet_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Apply fixed ImageNet RGB statistics, without estimating cohort statistics."""
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected a preprocessed 224 by 224 RGB uint8 image.")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(torch.float32) / 255.0
    mean = tensor.new_tensor(IMAGENET_MEAN).reshape(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).reshape(3, 1, 1)
    return (tensor - mean) / std


def prepared_image(path: Path) -> tuple[torch.Tensor, dict]:
    """Decode and hash the same bytes; reuse the original foreground/square crop."""
    content = Path(path).read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    prepared = preprocess(rgb, output_size=IMAGE_SIZE)
    tensor = normalized_imagenet_tensor(prepared)
    return tensor, {
        "image_path": str(Path(path).resolve()),
        "image_sha256": hashlib.sha256(content).hexdigest(),
        "source_height": int(rgb.shape[0]), "source_width": int(rgb.shape[1]),
        "prepared_rgb_sha256": array_sha256(prepared),
        "normalized_tensor_sha256": array_sha256(tensor.numpy()),
    }


class FrozenResNet18(nn.Module):
    """ResNet-18 through global average pooling, permanently in inference mode."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not isinstance(backbone.fc, nn.Linear) or backbone.fc.in_features != FEATURE_DIMENSION:
            raise ValueError("Expected the unmodified torchvision ResNet-18 classifier.")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenResNet18:
        """A surrounding pipeline cannot enable BatchNorm updates on this extractor."""
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if any(module.training for module in self.modules()):
            raise RuntimeError("Every backbone module must remain in eval mode.")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("Every backbone parameter must remain frozen.")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected a batch of 3 by 224 by 224 images.")
        if images.dtype != torch.float32:
            raise ValueError("Frozen feature extraction requires float32 input.")
        features = self.backbone(images)
        if features.shape != (len(images), FEATURE_DIMENSION) or not torch.isfinite(features).all():
            raise RuntimeError("Invalid ResNet-18 global-average-pooling features.")
        return features


def load_frozen_resnet18(checkpoint: Path, expected_sha256: str) -> FrozenResNet18:
    """Load a hash-verified local checkpoint without downloading or adapting weights."""
    content = Path(checkpoint).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError("The checkpoint SHA-256 differs from the frozen configuration.")
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(io.BytesIO(content), map_location="cpu", weights_only=True), strict=True)
    return FrozenResNet18(backbone)


@dataclass(frozen=True)
class ExtractionResult:
    features: np.ndarray
    image_inputs: list[dict]
    first_batch_bitwise_repeatable: bool
    model_state_sha256_before: str
    model_state_sha256_after: str


def extract_features(
    model: FrozenResNet18,
    paths: Sequence[Path],
    *,
    device: str = "cpu",
    batch_size: int = 32,
    expected_image_hashes: Sequence[str] | None = None,
) -> ExtractionResult:
    """Extract in input order; labels never enter this function or its preprocessing."""
    if batch_size < 1 or not paths:
        raise ValueError("Provide a positive batch size and at least one image.")
    if expected_image_hashes is not None and len(expected_image_hashes) != len(paths):
        raise ValueError("Expected exactly one image hash per input.")
    model.to(device)
    before = model_state_sha256(model)
    chunks, image_inputs = [], []
    repeatable = False
    for start in range(0, len(paths), batch_size):
        prepared = [prepared_image(path) for path in paths[start:start + batch_size]]
        for offset, (_, record) in enumerate(prepared):
            if expected_image_hashes is not None and record["image_sha256"] != expected_image_hashes[start + offset]:
                raise ValueError(f"Input image hash changed: {paths[start + offset]}")
            image_inputs.append({"row_index": start + offset, **record})
        batch = torch.stack([tensor for tensor, _ in prepared]).to(device)
        features = model(batch)
        if start == 0:
            repeatable = bool(torch.equal(features, model(batch)))
            if not repeatable:
                raise RuntimeError("Repeated inference on the same batch was not bitwise equal.")
        chunks.append(features.cpu().numpy().copy())
    after = model_state_sha256(model)
    if before != after:
        raise RuntimeError("Feature extraction changed model parameters or BatchNorm buffers.")
    return ExtractionResult(np.concatenate(chunks), image_inputs, repeatable, before, after)
