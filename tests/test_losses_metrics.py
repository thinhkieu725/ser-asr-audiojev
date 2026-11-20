import numpy as np
import torch

from sersys.calibration import fit_temperature
from sersys.losses import SmallLossFilter, class_weights, per_sample_loss, reduce_loss
from sersys.metrics import ece, ser_metrics


def test_class_weights_inv_sqrt():
    w = class_weights(np.array([100, 25, 400]), "inv_sqrt_freq")
    assert torch.allclose(w.mean(), torch.tensor(1.0))
    assert w[1] > w[0] > w[2]
    assert class_weights(np.array([1, 2]), "none") is None


def test_losses():
    logits = torch.randn(5, 7)
    y = torch.randint(0, 7, (5,))
    ce = per_sample_loss(logits, y, "ce")
    assert ce.shape == (5,)
    onehot = torch.nn.functional.one_hot(y, 7).float()
    assert torch.allclose(per_sample_loss(logits, y, "soft_kl", soft_targets=onehot), ce, atol=1e-5)
    keep = torch.tensor([1.0, 0, 1, 1, 0])
    assert torch.allclose(reduce_loss(ce, y, None, keep), ce[keep.bool()].mean())


def test_small_loss_filter_drops_high_losses_per_class():
    f = SmallLossFilter(2, drop_frac=0.2, warmup_epochs=0, min_history=10)
    rng = np.random.default_rng(0)
    for _ in range(50):
        f(torch.tensor(rng.random(4), dtype=torch.float32), torch.tensor([0, 1, 0, 1]), epoch=1.0)
    keep = f(torch.tensor([0.01, 5.0]), torch.tensor([0, 1]), epoch=1.0)
    assert keep.tolist() == [1.0, 0.0]


def test_metrics_and_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=1.5, size=(5000, 7))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(7, p=row) for row in p])  # labels calibrated at T = 1
    m = ser_metrics(logits * 4, y, 7)  # overconfident by a factor 4
    assert 0 <= m["ua"] <= 1 and np.array(m["confusion"]).sum() == 5000
    t = fit_temperature(logits * 4, y)
    assert 3.5 < t < 4.5
    after = ser_metrics(logits * 4, y, 7, temperature=t)
    assert after["ece"] < m["ece"] and after["ua"] == m["ua"]
    assert ece(np.eye(3), np.array([0, 1, 2])) == 0.0
