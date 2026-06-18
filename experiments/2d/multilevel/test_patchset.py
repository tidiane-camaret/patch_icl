import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_pfn import PatchSetPFN, FourierPositionalEncoding

def test_fourier_pe_shape_and_resolution_generalizes():
    pe = FourierPositionalEncoding(e=16, num_bands=6)
    ij = torch.randint(0, 32, (2, 10, 2))
    out32 = pe(ij, grid_res=32)
    assert out32.shape == (2, 10, 16)
    # same module runs at a different grid resolution (generalization) without error
    ij64 = torch.randint(0, 64, (2, 10, 2))
    out64 = pe(ij64, grid_res=64)
    assert out64.shape == (2, 10, 16)

def _mk(mask_prior="scalar", mask_patch_size=1):
    return PatchSetPFN(feature_dim=32, e=32, h=64, l=2, a=4, thinking_rows=2,
                       fourier_bands=6, mask_prior=mask_prior, mask_patch_size=mask_patch_size)

def test_forward_shapes_and_query_grad_only():
    B, S, Q, Fd, R = 2, 12, 8, 32, 32
    m = _mk(mask_prior="scalar")
    sup_feat = torch.randn(B, S, Fd); sup_label = torch.rand(B, S); sup_ij = torch.randint(0, R, (B, S, 2))
    qry_feat = torch.randn(B, Q, Fd); qry_prior = torch.rand(B, Q); qry_ij = torch.randint(0, R, (B, Q, 2))
    logits = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij, grid_res=R)
    assert logits.shape == (B, Q)
    logits.sum().backward()
    assert m.decoder[0].weight.grad is not None        # learns
    assert m.img_embed.weight.grad is not None

def test_mask_prior_false_runs():
    B, S, Q, Fd, R = 2, 12, 8, 32, 32
    m = _mk(mask_prior="false")
    out = m(torch.randn(B,S,Fd), torch.rand(B,S), torch.randint(0,R,(B,S,2)),
            torch.randn(B,Q,Fd), torch.rand(B,Q), torch.randint(0,R,(B,Q,2)), grid_res=R)
    assert out.shape == (B, Q)

def test_mask_prior_patch_runs():
    # patch mode: mask-token input is a p×p tile; mask_embed input dim = p².
    B, S, Q, Fd, R, p = 2, 12, 8, 32, 32, 4
    m = _mk(mask_prior="patch", mask_patch_size=p)
    assert m.mask_embed.in_features == p * p
    sup_label = torch.rand(B, S, p * p)
    qry_prior = torch.rand(B, Q, p * p)
    out = m(torch.randn(B,S,Fd), sup_label, torch.randint(0,R,(B,S,2)),
            torch.randn(B,Q,Fd), qry_prior, torch.randint(0,R,(B,Q,2)), grid_res=R)
    assert out.shape == (B, Q)
    out.sum().backward()
    assert m.mask_embed.weight.grad is not None

def test_stage1_thinking_memory():
    B, S, Q, Fd, R, T1, e1 = 2, 12, 8, 32, 32, 8, 48
    m = PatchSetPFN(feature_dim=Fd, e=32, h=64, l=2, a=4, thinking_rows=2,
                    fourier_bands=6, mask_prior="scalar", stage1_dim=e1)
    args = (torch.randn(B,S,Fd), torch.rand(B,S), torch.randint(0,R,(B,S,2)),
            torch.randn(B,Q,Fd), torch.rand(B,Q), torch.randint(0,R,(B,Q,2)))
    s1 = torch.randn(B, T1, e1)
    out = m(*args, grid_res=R, stage1_think=s1)
    assert out.shape == (B, Q)
    out.sum().backward()
    assert m.stage1_proj.weight.grad is not None      # memory projection learns
    # still runs when stage1_dim is set but no memory is passed
    out2 = m(*args, grid_res=R)
    assert out2.shape == (B, Q)

def test_query_self_attn_couples_queries():
    # Perturbing query 0's feature should change query 1's output ONLY when query
    # patches attend to each other (query_self_attn=True).
    B, S, Q, Fd, R = 1, 8, 6, 16, 32
    torch.manual_seed(0)
    sf, sl, sij = torch.randn(B,S,Fd), torch.rand(B,S), torch.randint(0,R,(B,S,2))
    qf, qp, qij = torch.randn(B,Q,Fd), torch.rand(B,Q), torch.randint(0,R,(B,Q,2))
    qf2 = qf.clone(); qf2[:, 0] += 5.0                       # perturb only query 0
    for qsa, expect_change in [(True, True), (False, False)]:
        torch.manual_seed(1)
        m = PatchSetPFN(feature_dim=Fd, e=16, h=32, l=2, a=4, thinking_rows=2,
                        fourier_bands=4, mask_prior="scalar", query_self_attn=qsa).eval()
        with torch.no_grad():
            o1 = m(sf, sl, sij, qf,  qp, qij, grid_res=R)
            o2 = m(sf, sl, sij, qf2, qp, qij, grid_res=R)
        changed = (o1[:, 1] - o2[:, 1]).abs().max().item() > 1e-6
        assert changed == expect_change, f"query_self_attn={qsa}: changed={changed}"

def test_return_thinking_shape():
    import torch
    from src.models.patchset_pfn import PatchSetPFN
    torch.manual_seed(0)
    B, S, Q, Fdim, e, nthink = 2, 12, 6, 8, 16, 4
    m = PatchSetPFN(feature_dim=Fdim, e=e, h=32, l=2, a=2, thinking_rows=nthink,
                    mask_prior="scalar", mask_patch_size=1, stage1_dim=None,
                    query_self_attn=True)
    sup_feat = torch.randn(B, S, Fdim); sup_label = torch.rand(B, S)
    sup_ij = torch.randint(0, 8, (B, S, 2))
    qry_feat = torch.randn(B, Q, Fdim); qry_prior = torch.rand(B, Q)
    qry_ij = torch.randint(0, 8, (B, Q, 2))
    logits, think = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij,
                      grid_res=8, return_thinking=True)
    assert logits.shape == (B, Q), logits.shape
    assert think.shape == (B, nthink, e), think.shape
    # default path unchanged: returns only logits
    out = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij, grid_res=8)
    assert out.shape == (B, Q), out.shape

if __name__ == "__main__":
    test_fourier_pe_shape_and_resolution_generalizes()
    test_forward_shapes_and_query_grad_only()
    test_mask_prior_false_runs()
    test_mask_prior_patch_runs()
    test_stage1_thinking_memory()
    test_query_self_attn_couples_queries()
    test_return_thinking_shape()
    print("ALL PATCHSET TESTS PASSED")
