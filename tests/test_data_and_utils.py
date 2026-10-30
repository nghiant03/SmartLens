import numpy as np
import pytest
import torch

from smartlens.data.datasets import infinite, make_loader, resize_mask
from smartlens.data.perturbations import PERTURBATIONS, apply_perturbation
from smartlens.data.splits import split_category
from smartlens.models.base import gaussian_blur, topk_mean


def _records(n_train=40, defects=(("good", 20), ("crack", 10), ("hole", 7))):
    train = [{"image": f"tr/{i}.png", "mask": None, "label": 0, "defect_type": "good"} for i in range(n_train)]
    test = []
    for d, n in defects:
        for i in range(n):
            test.append({"image": f"te/{d}/{i}.png", "mask": None if d == "good" else f"gt/{d}/{i}.png",
                         "label": int(d != "good"), "defect_type": d})
    return train, test


def test_split_disjoint_stratified_deterministic():
    train, test = _records()
    a = split_category(train, test, "bottle", seed=42)
    b = split_category(train, test, "bottle", seed=42)
    assert a == b
    all_imgs = [r["image"] for s in a.values() for r in s]
    assert len(all_imgs) == len(set(all_imgs)) == len(train) + len(test)
    assert len(a["val_normal"]) == 6 and len(a["train"]) == 34
    for d, n in (("good", 20), ("crack", 10), ("hole", 7)):
        assert sum(r["defect_type"] == d for r in a["test_val"]) == round(0.2 * n) or n < 5
        assert any(r["defect_type"] == d for r in a["test"])
    assert all(r["label"] == 0 for r in a["train"] + a["val_normal"])


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbations_deterministic(name):
    rng = np.random.default_rng(0)
    img = (rng.random((64, 80, 3)) * 255).astype(np.uint8)
    mask = np.zeros((64, 80), np.uint8); mask[20:30, 20:40] = 255
    o1, m1 = apply_perturbation(name, img, mask, "a.png")
    o2, m2 = apply_perturbation(name, img, mask, "a.png")
    assert o1.shape == img.shape and o1.dtype == np.uint8
    assert np.array_equal(o1, o2) and np.array_equal(m1, m2)
    assert not np.array_equal(o1, img)
    if name != "affine":
        assert np.array_equal(m1, mask)


def test_blur_and_topk():
    x = torch.zeros(2, 33, 33); x[:, 16, 16] = 1.0
    y = gaussian_blur(x, 2.0)
    assert y.sum().item() == pytest.approx(2.0, rel=1e-3)
    assert y[0].argmax().item() == 16 * 33 + 16
    assert topk_mean(torch.arange(100.0).view(1, 10, 10), 0.05).item() == pytest.approx(97.0)


def test_thin_mask_resampling_protocol_is_explicit():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 2] = 255
    assert not resize_mask(mask, (2, 2), "area").any()
    assert resize_mask(mask, (2, 2), "nearest").sum() == 2
    assert set(np.unique(resize_mask(mask, (16, 16), "nearest"))) <= {0, 1}
    # Even nearest-neighbour can lose off-grid defects: this is not native evaluation.
    assert not resize_mask(np.roll(mask, 1, axis=1), (2, 2), "nearest").any()


def test_empty_drop_last_loader_fails_instead_of_hanging():
    loader = make_loader(torch.arange(3), batch_size=4, shuffle=False, num_workers=0, drop_last=True)
    with pytest.raises(ValueError, match="Empty training loader"):
        next(infinite(loader))
