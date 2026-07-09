"""Hand-computed correctness tests for the per-algorithm losses.

All CPU, all tiny tensors — every expected value is derivable on paper.
"""

import math

import pytest
import torch

from minirl.algos import (
    CISPOConfig,
    GRPOConfig,
    GSPOConfig,
    SFTConfig,
    aggregate_loss,
    cispo_loss,
    degenerate_group_mask,
    grpo_advantages,
    grpo_loss,
    gspo_loss,
    make_loss,
    sft_loss,
)
from minirl.rollout.types import Batch


def make_batch(policy_lp, old_lp=None, behavior_lp=None, ref_lp=None, adv=None, mask=None) -> Batch:
    """Batch sized from policy_lp; defaults keep everything neutral."""
    b, t = policy_lp.shape
    mask = mask if mask is not None else torch.ones(b, t, dtype=torch.bool)
    return Batch(
        input_ids=torch.zeros(b, t, dtype=torch.long),
        attention_mask=torch.ones(b, t, dtype=torch.bool),
        loss_mask=mask,
        behavior_logprobs=behavior_lp if behavior_lp is not None else torch.zeros(b, t),
        advantages=adv if adv is not None else torch.ones(b, t),
        rewards=torch.zeros(b),
        group_ids=torch.zeros(b, dtype=torch.long),
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
    )


# ---------------- advantages ----------------


def test_grpo_advantages_hand_computed():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = grpo_advantages(rewards, groups)
    # group 0: mean .5, unbiased std sqrt(.5) -> +-(.5/.7071) ~ +-.7071
    assert torch.allclose(adv[:2], torch.tensor([0.7071, -0.7071]), atol=1e-3)
    assert torch.allclose(adv[2:], torch.zeros(2))  # degenerate group -> 0

    adv_no_std = grpo_advantages(rewards, groups, norm_std=False)  # Dr. GRPO
    assert torch.allclose(adv_no_std, torch.tensor([0.5, -0.5, 0.0, 0.0]))

    assert degenerate_group_mask(rewards, groups).tolist() == [False, False, True, True]


# ---------------- GRPO ----------------


def test_grpo_clip_hand_computed():
    lp = torch.tensor([[math.log(2.0), 0.0]])  # ratios [2, 1]
    loss_map, m = grpo_loss(lp, make_batch(lp), GRPOConfig())
    # min(2*1, 1.2*1) = 1.2 ; min(1*1, 1*1) = 1
    assert torch.allclose(loss_map, torch.tensor([[-1.2, -1.0]]), atol=1e-6)
    assert m["clip_frac"].item() == pytest.approx(0.5)

    # negative advantage: min picks the UNclipped branch -> no clipping benefit
    loss_map, m = grpo_loss(lp, make_batch(lp, adv=-torch.ones(1, 2)), GRPOConfig())
    assert torch.allclose(loss_map, torch.tensor([[2.0, 1.0]]))
    assert m["clip_frac"].item() == 0.0


def test_grpo_kl_penalty():
    lp = torch.zeros(1, 2)
    cfg = GRPOConfig(use_kl_loss=True, kl_loss_coef=0.1)
    base, _ = grpo_loss(lp, make_batch(lp), GRPOConfig())
    at_ref, m0 = grpo_loss(lp, make_batch(lp, ref_lp=torch.zeros(1, 2)), cfg)
    away, m1 = grpo_loss(lp, make_batch(lp, ref_lp=torch.full((1, 2), -1.0)), cfg)
    assert torch.allclose(at_ref, base)  # policy == ref -> k3 = 0
    assert m0["kl_ref"].item() == pytest.approx(0.0, abs=1e-8)
    assert (away - base).abs().sum() > 0 and m1["kl_ref"].item() > 0


def test_grpo_tis_clamp_and_mask():
    # policy == old (ratio 1) but the engine reported much lower logprobs:
    # tis = exp(old - behavior) = 4, cap = 2
    lp = torch.zeros(1, 1)
    mism = dict(old_lp=torch.zeros(1, 1), behavior_lp=torch.full((1, 1), -math.log(4.0)))
    base, _ = grpo_loss(lp, make_batch(lp, **mism), GRPOConfig())
    clamped, m = grpo_loss(lp, make_batch(lp, **mism), GRPOConfig(use_tis=True))
    masked, _ = grpo_loss(lp, make_batch(lp, **mism), GRPOConfig(use_tis=True, tis_mode="mask"))
    assert torch.allclose(clamped, base * 2.0)  # truncated at cap 2.0
    assert torch.allclose(masked, torch.zeros_like(base))  # icepop: rejected
    assert m["tis_mean"].item() == pytest.approx(4.0, rel=1e-5)
    assert m["tis_clip_frac"].item() == 1.0

    # no mismatch -> TIS is the identity
    ident, _ = grpo_loss(lp, make_batch(lp), GRPOConfig(use_tis=True))
    assert torch.allclose(ident, grpo_loss(lp, make_batch(lp), GRPOConfig())[0])


def test_tis_rescales_surrogate_but_not_kl():
    """slime ordering: TIS multiplies the pg term BEFORE the KL penalty is added.

    surrogate = -1 (ratio 1, A=1), tis weight = clamp(4, max=2) = 2,
    kl(k3, d=-1) = e^-1 + 1 - 1 = e^-1  ->  loss = -1*2 + 0.1*e^-1
    (if TIS wrongly rescaled KL too, we'd get (-1 + 0.1*e^-1) * 2 instead)
    """
    lp = torch.zeros(1, 1)
    batch = make_batch(
        lp,
        old_lp=torch.zeros(1, 1),
        behavior_lp=torch.full((1, 1), -math.log(4.0)),
        ref_lp=torch.full((1, 1), -1.0),
    )
    cfg = GRPOConfig(use_tis=True, use_kl_loss=True, kl_loss_coef=0.1)
    loss_map, _ = grpo_loss(lp, batch, cfg)
    assert loss_map.item() == pytest.approx(-2.0 + 0.1 * math.exp(-1), rel=1e-6)


# ---------------- DAPO ----------------


def test_dapo_is_a_named_grpo_config():
    """DAPO = GRPO's loss body + {eps_clip_high: 0.28, loss_agg: token_mean}."""
    dapo_fn, dapo_cfg = make_loss("dapo")
    assert dapo_fn is grpo_loss  # file-vs-config rule: identical body
    assert dapo_cfg.eps_clip_high == 0.28 and dapo_cfg.loss_agg == "token_mean"
    assert dapo_cfg.use_kl_loss is False  # change #2 is GRPO's default anyway

    lp = torch.tensor([[math.log(1.25)]])
    grpo_out, _ = grpo_loss(lp, make_batch(lp), GRPOConfig())  # hi = 0.2 -> clipped at 1.2
    dapo_out, m = dapo_fn(lp, make_batch(lp), dapo_cfg)  # hi = 0.28 -> 1.25 admitted
    assert torch.allclose(grpo_out, torch.tensor([[-1.2]]), atol=1e-6)
    assert torch.allclose(dapo_out, torch.tensor([[-1.25]]), atol=1e-6)
    assert m["clip_frac"].item() == 0.0
    # the lower clip binds only for A<0 (min picks the pessimistic branch):
    # ratio 0.5, A=-1 -> min(-0.5, clip(0.5)->0.8 * -1) = -0.8 -> loss 0.8
    lp_dn = torch.tensor([[math.log(0.5)]])
    dn, _ = dapo_fn(lp_dn, make_batch(lp_dn, adv=-torch.ones(1, 1)), dapo_cfg)
    assert torch.allclose(dn, torch.tensor([[0.8]]), atol=1e-6)


# ---------------- GSPO ----------------


def test_gspo_sequence_ratio_is_uniform_geometric_mean():
    lp = torch.tensor([[0.1, 0.3]])
    wide = GSPOConfig(eps_clip=10.0, eps_clip_high=10.0)  # no clipping
    loss_map, _ = gspo_loss(lp, make_batch(lp), wide)
    expected = -math.exp(0.2)  # exp(mean log-ratio), same on every token
    assert torch.allclose(loss_map, torch.full((1, 2), expected), atol=1e-6)

    # with default tiny eps, that ratio clips to 1 + 4e-4 (whole-sequence clip)
    loss_map, m = gspo_loss(lp, make_batch(lp), GSPOConfig())
    assert torch.allclose(loss_map, torch.full((1, 2), -(1 + 4e-4)), atol=1e-6)
    assert m["clip_frac"].item() == 1.0


# ---------------- CISPO ----------------


def test_cispo_gradient_flows_through_clipped_tokens():
    lp = torch.tensor([[math.log(4.0)]], requires_grad=True)  # ratio 4, way past cap
    loss_map, m = cispo_loss(lp, make_batch(lp.detach()), CISPOConfig())
    loss_map.sum().backward()
    # d/dlp of -sg(1.28) * 1 * lp = -1.28: clipped, yet gradient flows
    assert lp.grad.abs().item() == pytest.approx(1.28, abs=1e-6)
    assert m["clip_frac"].item() == 1.0

    # contrast: GRPO's surrogate on the same token has ZERO gradient
    lp2 = torch.tensor([[math.log(4.0)]], requires_grad=True)
    loss_map, _ = grpo_loss(lp2, make_batch(lp2.detach()), GRPOConfig())
    loss_map.sum().backward()
    assert lp2.grad.abs().item() == pytest.approx(0.0, abs=1e-8)


def test_cispo_no_lower_bound_by_default():
    lp = torch.tensor([[math.log(0.01)]])  # ratio far below 1 - eps for any eps
    loss_map, m = cispo_loss(lp, make_batch(lp), CISPOConfig())
    # weight stays 0.01 (unbounded below): loss = -0.01 * 1 * ln(0.01)
    assert loss_map.item() == pytest.approx(-0.01 * math.log(0.01), rel=1e-5)
    assert m["clip_frac"].item() == 0.0


# ---------------- SFT + aggregation + wrapper ----------------


def test_sft_and_aggregation_modes():
    lp = torch.tensor([[-1.0, -1.0, 0.0, 0.0], [-2.0, -2.0, -2.0, -2.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    loss_map, m = sft_loss(lp, make_batch(lp, mask=mask), SFTConfig())
    # token: (2*1 + 4*2) / 6 ; sequence: (1 + 2) / 2
    assert aggregate_loss(loss_map, mask, "token_mean").item() == pytest.approx(10 / 6)
    assert aggregate_loss(loss_map, mask, "seq_mean").item() == pytest.approx(1.5)
    assert m["nll"].item() == pytest.approx(10 / 6)


def test_token_aggregation_microbatch_invariance():
    """Splitting a minibatch must not change the result when denom is global."""
    torch.manual_seed(0)
    loss_map = torch.rand(4, 6)
    mask = torch.rand(4, 6) > 0.4
    full = aggregate_loss(loss_map, mask, "token_mean")
    denom = mask.sum()
    split = sum(aggregate_loss(loss_map[i : i + 2], mask[i : i + 2], "token_mean", denom=denom) for i in (0, 2))
    assert full.item() == pytest.approx(split.item(), rel=1e-6)


def test_make_loss_wrapper():
    loss_fn, cfg = make_loss("dapo", eps_clip_high=0.3)
    assert loss_fn is grpo_loss and cfg.eps_clip_high == 0.3  # override beats the preset
    # dr_grpo: named variant = grpo loss body + the two bias-removal flags
    loss_fn, cfg = make_loss("dr_grpo")
    assert loss_fn is grpo_loss
    assert cfg.grpo_std_normalization is False and cfg.loss_agg == "token_mean"
    _, cfg = make_loss("dr_grpo", use_tis=True, grpo_std_normalization=True)  # overrides still win
    assert cfg.use_tis is True and cfg.grpo_std_normalization is True
    lp = torch.zeros(1, 2)
    loss_map, metrics = loss_fn(lp, make_batch(lp), cfg)
    assert loss_map.shape == (1, 2) and "clip_frac" in metrics
    with pytest.raises(KeyError):
        make_loss("nope")
