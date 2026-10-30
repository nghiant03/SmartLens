import torch
import torch.nn.functional as F

from smartlens.config import PROJECT_ROOT
from smartlens.losses.ssim import ms_ssim_loss, ssim_map
from smartlens.models.base import gaussian_kernel1d
from smartlens.models.baselines.dinomaly import LinearAttention, global_cosine_hm_percent
from smartlens.models.baselines.patchcore import chunked_nn, greedy_coreset
from smartlens.models.baselines.ssim_cae import SSIMCAE


def _ref_ssim_2d(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    c = x.shape[1]
    k1d = gaussian_kernel1d(sigma, window, x.device).to(x.dtype)
    k2d = (k1d[:, None] @ k1d[None, :]).expand(c, 1, window, window).contiguous()
    pad = window // 2
    filt = lambda t: F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), k2d, groups=c)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = filt(x), filt(y)
    sxx, syy, sxy = (filt(x * x) - mx * mx).clamp_min(0), (filt(y * y) - my * my).clamp_min(0), filt(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def test_ssim_identity_and_separable_equivalence():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim_map(x, x), torch.ones_like(x), atol=1e-4)
    assert ms_ssim_loss(x, x).item() < 1e-4
    assert torch.allclose(ssim_map(x, y), _ref_ssim_2d(x, y), atol=1e-5)


def test_ssim_cae_shapes_and_save_load(tmp_path):
    m = SSIMCAE({"name": "ssim_cae", "input_size": 64, "latent_dim": 16, "base_channels": 8, "amp": False}).eval()
    out = m.predict(torch.rand(2, 3, 64, 64))
    assert out["anomaly_map"].shape == (2, 64, 64) and out["image_score"].shape == (2,)
    m.save(tmp_path / "m.pt")
    m2 = SSIMCAE(m.cfg).load(tmp_path / "m.pt").eval()
    assert torch.allclose(m2.predict(torch.ones(1, 3, 64, 64))["image_score"],
                          m.predict(torch.ones(1, 3, 64, 64))["image_score"])


def test_coreset_and_chunked_nn():
    torch.manual_seed(0)
    bank = torch.randn(200, 16)
    idx = greedy_coreset(bank, 0.1, proj_dim=8, seed=0)
    assert idx.shape == (20,) and len(set(idx.tolist())) == 20
    q = torch.randn(15, 16)
    d, i = chunked_nn(q, bank, k=3, chunk=4)
    ref_d, ref_i = torch.cdist(q, bank).topk(3, dim=1, largest=False)
    assert torch.allclose(d, ref_d, atol=1e-4)
    assert torch.equal(i, ref_i)


def test_linear_attention_and_hm_loss():
    att = LinearAttention(32, 4)
    x = torch.randn(2, 10, 32)
    assert att(x).shape == x.shape
    en = [torch.randn(2, 8, 4, 4)]
    de = [torch.randn(2, 8, 4, 4, requires_grad=True)]
    loss = global_cosine_hm_percent(en, de, p=0.9, factor=0.0)
    loss.backward()
    nz = (de[0].grad.abs().sum(1) > 0).float().mean().item()
    assert 0 < nz <= 0.2


def test_no_dataset_download_scripts_in_codebase():
    forbidden = ("mydrive.ch", "mvtec_anomaly_detection.tar", "VisA_20220922.tar", "amazon-visual-anomaly")
    for folder in ("smartlens", "scripts", "configs"):
        for p in (PROJECT_ROOT / folder).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".yml"}:
                txt = p.read_text(encoding="utf-8")
                for token in forbidden:
                    assert token not in txt, f"Found dataset download reference {token!r} in {p}"
