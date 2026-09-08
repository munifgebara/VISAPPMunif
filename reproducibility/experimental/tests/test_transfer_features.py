from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from pdhms_restart.transfer_features import (
    FrozenResNet18, IMAGENET_MEAN, IMAGENET_STD, configure_determinism,
    extract_features, load_frozen_resnet18, model_state_sha256,
    normalized_imagenet_tensor, prepared_image,
)


@pytest.fixture
def frozen_model() -> FrozenResNet18:
    # Unit tests do not download weights or depend on a private dataset.
    configure_determinism(seed=41, cpu_threads=2)
    return FrozenResNet18(resnet18(weights=None))


def save_drawing(path: Path, channel: int = 0) -> None:
    rgb = np.full((80, 240, 3), 255, dtype=np.uint8)
    rgb[25:55, 0:12, :] = 0
    rgb[25:55, 228:240, :] = 0
    rgb[25:55, 0:12, channel] = 255
    rgb[25:55, 228:240, (channel + 2) % 3] = 255
    Image.fromarray(rgb).save(path)


def test_whole_drawing_survives_crop_padding_and_imagenet_normalization(tmp_path: Path) -> None:
    path = tmp_path / "drawing.png"
    save_drawing(path)
    tensor, record = prepared_image(path)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    mean = torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(3, 1, 1)
    rgb = ((tensor * std + mean) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    # Markers touch opposite source boundaries. A centre crop would discard them.
    assert np.any((rgb[:, :, 0] > 240) & (rgb[:, :, 1] < 10) & (rgb[:, :, 2] < 10))
    assert np.any((rgb[:, :, 2] > 240) & (rgb[:, :, 0] < 10) & (rgb[:, :, 1] < 10))
    assert np.all(rgb[0] == 255) and np.all(rgb[-1] == 255)
    assert record["image_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    again, repeated_record = prepared_image(path)
    assert torch.equal(tensor, again) and record == repeated_record


def test_fixed_statistics_are_not_fitted_to_image_or_dataset() -> None:
    white = np.full((224, 224, 3), 255, dtype=np.uint8)
    tensor = normalized_imagenet_tensor(white)
    expected = (torch.ones(3) - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(tensor[:, 0, 0], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="uint8"):
        normalized_imagenet_tensor(white.astype(np.float32))


def test_eval_is_permanent_and_features_are_exact_global_average_pool(frozen_model: FrozenResNet18) -> None:
    frozen_model.train(True)
    assert not any(module.training for module in frozen_model.modules())
    assert not any(parameter.requires_grad for parameter in frozen_model.parameters())
    pooled = []
    handle = frozen_model.backbone.avgpool.register_forward_hook(lambda module, inputs, output: pooled.append(output))
    try:
        output = frozen_model(torch.rand(2, 3, 224, 224, requires_grad=True))
    finally:
        handle.remove()
    assert output.shape == (2, 512) and not output.requires_grad
    torch.testing.assert_close(output, pooled[0].flatten(1), rtol=0, atol=0)


def test_direct_batchnorm_mode_tampering_is_rejected(frozen_model: FrozenResNet18) -> None:
    batchnorm = next(module for module in frozen_model.modules() if isinstance(module, nn.BatchNorm2d))
    batchnorm.train(True)
    with pytest.raises(RuntimeError, match="eval mode"):
        frozen_model(torch.zeros(1, 3, 224, 224))


def test_extraction_is_repeatable_keeps_order_and_never_updates_bn(
    frozen_model: FrozenResNet18, tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    save_drawing(first, 0)
    save_drawing(second, 1)
    paths = [second, first]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    state_before = {name: tensor.clone() for name, tensor in frozen_model.state_dict().items()}
    result = extract_features(frozen_model, paths, batch_size=2, expected_image_hashes=hashes)
    repeated = extract_features(frozen_model, paths, batch_size=2, expected_image_hashes=hashes)
    assert result.features.shape == (2, 512) and result.features.dtype == np.float32
    assert np.isfinite(result.features).all() and not np.array_equal(result.features[0], result.features[1])
    np.testing.assert_array_equal(result.features, repeated.features)
    assert result.first_batch_bitwise_repeatable and repeated.first_batch_bitwise_repeatable
    assert [record["image_sha256"] for record in result.image_inputs] == hashes
    assert [record["row_index"] for record in result.image_inputs] == [0, 1]
    assert result.model_state_sha256_before == result.model_state_sha256_after
    for name, tensor in frozen_model.state_dict().items():
        torch.testing.assert_close(tensor, state_before[name], rtol=0, atol=0)


def test_changed_image_is_rejected_before_inference(frozen_model: FrozenResNet18, tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    save_drawing(path)
    with pytest.raises(ValueError, match="Input image hash changed"):
        extract_features(frozen_model, [path], expected_image_hashes=["0" * 64])


def test_local_checkpoint_requires_exact_hash_and_retains_weights(tmp_path: Path) -> None:
    configure_determinism(seed=18, cpu_threads=2)
    original = resnet18(weights=None)
    path = tmp_path / "local-test-checkpoint.pth"
    torch.save(original.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_frozen_resnet18(path, "0" * 64)
    loaded = load_frozen_resnet18(path, digest)
    assert isinstance(loaded.backbone.fc, nn.Identity)
    torch.testing.assert_close(loaded.backbone.conv1.weight, original.conv1.weight, rtol=0, atol=0)


def test_state_hash_includes_batchnorm_buffers(frozen_model: FrozenResNet18) -> None:
    before = model_state_sha256(frozen_model)
    frozen_model.backbone.bn1.running_mean[0] += 1
    assert model_state_sha256(frozen_model) != before


def test_non_float32_or_wrong_size_input_is_rejected(frozen_model: FrozenResNet18) -> None:
    with pytest.raises(ValueError, match="224"):
        frozen_model(torch.zeros(1, 3, 128, 128))
    with pytest.raises(ValueError, match="float32"):
        frozen_model(torch.zeros(1, 3, 224, 224, dtype=torch.float16))
