"""Candidate permutation bookkeeping (PLAN 1.4)."""
import torch

from sersys.scoring import random_perms, tta_perms, unpermute


def test_unpermute_maps_positions_back_to_classes():
    torch.manual_seed(0)
    B, C = 16, 7
    perm = random_perms(B, C)
    # The logit at displayed position j encodes the class shown there.
    logits = unpermute(perm.float(), perm)
    assert torch.equal(logits, torch.arange(C).float().expand(B, C))


def test_unpermute_with_class_specific_scores():
    torch.manual_seed(1)
    B, C = 8, 7
    canonical = torch.randn(B, C)
    perm = random_perms(B, C)
    shown = torch.gather(canonical, 1, perm)  # what the model scores at each displayed position
    assert torch.allclose(unpermute(shown, perm), canonical)


def test_identity_and_gradients():
    C = 7
    perm = torch.arange(C).expand(3, C)
    x = torch.randn(3, C, requires_grad=True)
    y = unpermute(x, perm)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_tta_perms_are_cyclic_shifts_of_reference():
    ref = [3, 0, 6, 1, 5, 2, 4]
    shifts = {tuple(ref[s:] + ref[:s]) for s in range(7)}
    for k in (1, 3, 7):
        perms = tta_perms(ref, k)
        assert perms.shape == (k, 7)
        assert perms[0].tolist() == ref
        assert len({tuple(p) for p in perms.tolist()}) == k
        assert all(tuple(p) in shifts for p in perms.tolist())
