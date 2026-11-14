"""Запуск: python -m pytest -q tests/  (сначала на CPU, затем на CUDA при наличии)."""
import numpy as np, torch, pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ref_numpy as R, scan_torch as S

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

def make(lens, K, seed=0, dtype=torch.float64, device="cpu"):
    g = np.random.default_rng(seed); T = sum(lens)
    m = g.random((T, K)) * 0.9 + 0.05; c = g.random((T, K)) * 0.1
    seg = np.repeat(np.arange(len(lens)), lens); k0 = g.random((len(lens), K))
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    t = lambda a: torch.tensor(a, dtype=dtype, device=device)
    return m, c, seg, k0, starts, t(m), t(c), torch.tensor(seg, device=device), t(k0)

@pytest.mark.parametrize("device", DEVICES)
def test_forward_matches_reference(device):
    m, c, seg, k0, starts, tm, tc, ts, tk0 = make([7, 1, 13, 4, 20, 2, 9, 64, 3], K=5, device=device)
    ref = R.forward_seq(m, c, starts, k0)
    k = S.trajectories(tm, tc, tk0, ts).cpu().numpy()
    assert np.abs(k - ref).max() < 1e-13

@pytest.mark.parametrize("device", DEVICES)
def test_backward_matches_autograd(device):
    m, c, seg, k0, starts, tm, tc, ts, tk0 = make([5, 17, 1, 30, 8], K=3, seed=1, device=device)
    w = torch.rand_like(tm)
    grads = []
    for eff in (True, False):
        a, b, z = (x.clone().requires_grad_() for x in (tm, tc, tk0))
        L = (w * S.trajectories(a, b, z, ts, memory_efficient=eff) ** 2).sum(); L.backward()
        grads.append((a.grad, b.grad, z.grad))
    for g1, g2 in zip(*grads):
        assert (g1 - g2).abs().max().item() < 1e-12

def test_adjoint_matches_reference():
    m, c, seg, k0, starts, tm, tc, ts, tk0 = make([6, 11, 2], K=2, seed=2)
    g = np.random.default_rng(3).random(m.shape)
    lam_ref = R.adjoint_seq(m, g, starts)
    # lambda из обратного прохода = градиент по c при L = sum g * k
    b = tc.clone().requires_grad_()
    (torch.tensor(g) * S.trajectories(tm, b, tk0, ts)).sum().backward()
    assert np.abs(b.grad.numpy() - lam_ref).max() < 1e-13


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("B", [1, 3, 16, 64, 4096])
def test_chunked_forward(device, B):
    m, c, seg, k0, starts, tm, tc, ts, tk0 = make([7, 1, 13, 4, 20, 2, 9, 64, 3, 150], K=4, seed=7, device=device)
    ref = R.forward_seq(m, c, starts, k0)
    k = S.trajectories(tm, tc, tk0, ts, method="chunked", B=B).cpu().numpy()
    assert np.abs(k - ref).max() < 1e-13

@pytest.mark.parametrize("device", DEVICES)
def test_chunked_backward(device):
    m, c, seg, k0, starts, tm, tc, ts, tk0 = make([5, 17, 1, 30, 8, 77], K=3, seed=8, device=device)
    w = torch.rand_like(tm); grads = []
    for meth in ("hillis", "chunked"):
        a, b, z = (x.clone().requires_grad_() for x in (tm, tc, tk0))
        (w * S.trajectories(a, b, z, ts, method=meth, B=16) ** 2).sum().backward()
        grads.append((a.grad, b.grad, z.grad))
    for g1, g2 in zip(*grads):
        assert (g1 - g2).abs().max().item() < 1e-12
