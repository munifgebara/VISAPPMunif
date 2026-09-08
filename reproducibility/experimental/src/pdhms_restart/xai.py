"""Spatial explanations and perturbation checks for binary CNN predictions."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


def predicted_class_confidence(probability_pd: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return confidence in each sample's declared predicted class."""

    return torch.where(predicted == 1, probability_pd, 1.0 - probability_pd)


def gradcam(
    model: nn.Module,
    images: torch.Tensor,
    predicted: torch.Tensor,
    target_layer: nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Grad-CAM maps for the predicted binary class."""

    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output: torch.Tensor) -> None:
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)
        targets = torch.where(predicted == 1, logits, -logits)
        targets.sum().backward()
        activation = captured["activation"]
        gradient = activation.grad
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * activation).sum(dim=1))
        maps = functional.interpolate(
            maps[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        flat = maps.flatten(1)
        maximum = flat.max(dim=1).values
        normalized = torch.where(
            maximum[:, None, None] > 0,
            maps / maximum[:, None, None].clamp_min(1e-12),
            torch.zeros_like(maps),
        )
        probabilities = torch.sigmoid(logits)
        return normalized.detach().cpu().numpy(), probabilities.detach().cpu().numpy()
    finally:
        handle.remove()


def channel_ablation_drops(
    model: nn.Module,
    images: torch.Tensor,
    predicted: torch.Tensor,
) -> np.ndarray:
    """Measure predicted-class confidence loss after replacing each channel by white."""

    model.eval()
    with torch.no_grad():
        baseline_probability = torch.sigmoid(model(images))
        baseline_confidence = predicted_class_confidence(baseline_probability, predicted)
        drops = []
        for channel in range(3):
            ablated = images.clone()
            ablated[:, channel] = 1.0
            probability = torch.sigmoid(model(ablated))
            confidence = predicted_class_confidence(probability, predicted)
            drops.append(baseline_confidence - confidence)
    return torch.stack(drops, dim=1).cpu().numpy()


def deletion_drops(
    model: nn.Module,
    images: torch.Tensor,
    predicted: torch.Tensor,
    maps: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare deletion of top Grad-CAM pixels with equally sized random deletion."""

    if not 0 < fraction < 1:
        raise ValueError("Deletion fraction must be between zero and one.")
    batch, _, height, width = images.shape
    pixel_count = height * width
    deleted_count = max(1, int(round(pixel_count * fraction)))
    flat_maps = maps.reshape(batch, pixel_count)
    top_indices = np.argpartition(flat_maps, -deleted_count, axis=1)[:, -deleted_count:]
    rng = np.random.default_rng(seed)
    random_indices = np.vstack(
        [rng.choice(pixel_count, size=deleted_count, replace=False) for _ in range(batch)]
    )

    def delete(indices: np.ndarray) -> torch.Tensor:
        result = images.clone()
        for sample_index, selected in enumerate(indices):
            ys, xs = np.divmod(selected, width)
            result[
                sample_index,
                :,
                torch.as_tensor(ys, device=result.device),
                torch.as_tensor(xs, device=result.device),
            ] = 1.0
        return result

    model.eval()
    with torch.no_grad():
        baseline_probability = torch.sigmoid(model(images))
        baseline_confidence = predicted_class_confidence(baseline_probability, predicted)
        targeted_probability = torch.sigmoid(model(delete(top_indices)))
        random_probability = torch.sigmoid(model(delete(random_indices)))
        targeted_confidence = predicted_class_confidence(targeted_probability, predicted)
        random_confidence = predicted_class_confidence(random_probability, predicted)
    return (
        (baseline_confidence - targeted_confidence).cpu().numpy(),
        (baseline_confidence - random_confidence).cpu().numpy(),
    )


def occlusion_map(
    model: nn.Module,
    image: torch.Tensor,
    predicted: int,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    """Map positive confidence loss caused by white square occlusion."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one channel-first image.")
    _, height, width = image.shape
    if patch_size > height or patch_size > width or stride < 1:
        raise ValueError("Invalid occlusion patch geometry.")
    positions = [
        (y, x)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]
    device = image.device
    predicted_tensor = torch.tensor([predicted], dtype=torch.int64, device=device)
    model.eval()
    with torch.no_grad():
        baseline_probability = torch.sigmoid(model(image[None]))
        baseline_confidence = float(
            predicted_class_confidence(baseline_probability, predicted_tensor).item()
        )
    losses = []
    for start in range(0, len(positions), batch_size):
        current = positions[start : start + batch_size]
        batch = image[None].repeat(len(current), 1, 1, 1)
        for index, (y, x) in enumerate(current):
            batch[index, :, y : y + patch_size, x : x + patch_size] = 1.0
        labels = torch.full((len(current),), predicted, dtype=torch.int64, device=device)
        with torch.no_grad():
            probability = torch.sigmoid(model(batch))
            confidence = predicted_class_confidence(probability, labels)
        losses.extend(torch.relu(baseline_confidence - confidence).cpu().numpy().tolist())
    importance = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for (y, x), loss in zip(positions, losses, strict=True):
        importance[y : y + patch_size, x : x + patch_size] += float(loss)
        counts[y : y + patch_size, x : x + patch_size] += 1.0
    importance = np.divide(importance, counts, out=np.zeros_like(importance), where=counts > 0)
    maximum = float(importance.max())
    return importance / maximum if maximum > 0 else importance


def top_fraction_iou(first: np.ndarray, second: np.ndarray, fraction: float = 0.10) -> float:
    """Intersection over union between equal-size top-attribution pixel sets."""

    if first.shape != second.shape or not 0 < fraction < 1:
        raise ValueError("Attribution maps or fraction are invalid.")
    count = max(1, int(round(first.size * fraction)))
    first_top = set(np.argpartition(first.ravel(), -count)[-count:].tolist())
    second_top = set(np.argpartition(second.ravel(), -count)[-count:].tolist())
    return len(first_top & second_top) / len(first_top | second_top)
