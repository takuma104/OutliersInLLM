"""Integration tests on the real checkpoints (GPU, cached weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from outliers.data import token_byte_lengths, token_categories, delimiter_token_mask
from outliers.evaluate import evaluate_lm, token_nll
from outliers.hooks import OutlierProbe
from outliers.models import NON_QUANT_KINDS, iter_linears, iter_residual_norms, layer_types, load_model, set_attn_impl
from outliers.quant import SCHEMES, fake_quantize
from outliers.stats import norm_weight_table, weight_stats_table

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.\n\n"
    "Locally nicknamed \"La dame de fer\", it was constructed from 1887 to 1889 as the centerpiece of the "
    "1889 World's Fair. Although initially criticised by some of France's leading artists and intellectuals "
    "for its design, it has since become a global cultural icon of France and one of the most recognisable "
    "structures in the world."
)

EXPECTED = {
    "qwen3.5-0.8b": {"layers": 24, "full": 6, "linears": 18 * 5 + 6 * 4 + 24 * 3},
    "qwen3-0.6b": {"layers": 28, "full": 28, "linears": 28 * 7},
}


@pytest.fixture(scope="module", params=list(EXPECTED))
def loaded(request: pytest.FixtureRequest) -> tuple[str, PreTrainedModel, PreTrainedTokenizerBase]:
    model, tok = load_model(request.param)
    yield request.param, model, tok
    del model
    torch.cuda.empty_cache()


def _ids(tok: PreTrainedTokenizerBase, n: int = 2) -> torch.Tensor:
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :96]
    return ids.repeat(n, 1).cuda()


def test_structure(loaded: tuple[str, PreTrainedModel, PreTrainedTokenizerBase]) -> None:
    name, model, _ = loaded
    exp = EXPECTED[name]
    types = layer_types(model)
    assert len(types) == exp["layers"]
    assert types.count("full_attention") == exp["full"]
    assert len(iter_residual_norms(model)) == 2 * exp["layers"] + 1
    assert len(iter_linears(model)) == exp["linears"]
    lam = norm_weight_table(model)
    if name.startswith("qwen3.5"):
        w = model.model.layers[0].input_layernorm.weight.float()
        assert torch.allclose(torch.tensor(lam[lam.norm == "model.layers.0.input_layernorm"].lambda_eff.values),
                              (1 + w).cpu())
    ws = weight_stats_table(model)
    assert len(ws) == exp["linears"] + 1  # + lm_head
    assert ws["rel_err_int8_ch"].max() < 0.05


def test_probe_is_passive_and_complete(loaded: tuple[str, PreTrainedModel, PreTrainedTokenizerBase]) -> None:
    name, model, tok = loaded
    ids = _ids(tok)
    cats = token_categories(ids.cpu(), delimiter_token_mask(tok))
    ref = token_nll(model, ids)
    with OutlierProbe(model) as probe:
        probe.set_batch(cats.cuda(), 0)
        out = token_nll(model, ids)  # probe attached: forward must be unchanged
    assert torch.equal(ref, out)
    assert len(model.model.norm._forward_pre_hooks) == 0
    with OutlierProbe(model) as probe:
        probe.run(ids, cats, batch_size=1)
        res = probe.results()

    exp = EXPECTED[name]
    n_res = 2 * exp["layers"] + 1
    assert (res.residual.kind == "residual").sum() == n_res
    assert (res.residual.kind == "block").sum() == 2 * exp["layers"]
    assert len(res.linears) == exp["linears"]
    assert res.residual.n_tokens.min() > 0
    # first residual read equals the embedding of the tokens
    emb = model.model.embed_tokens(ids).float()
    row = res.residual[res.residual.name == "model.layers.0.input_layernorm"].iloc[0]
    assert row.max_first == pytest.approx(float(emb[:, 0].abs().max()), rel=1e-6)
    quantized = res.linears[~res.linears.kind.isin(NON_QUANT_KINDS)]
    assert quantized.filter(like="sqnr_").notna().all().all()


def test_attention_probe_eager(loaded: tuple[str, PreTrainedModel, PreTrainedTokenizerBase]) -> None:
    name, model, tok = loaded
    ids = _ids(tok)
    cats = token_categories(ids.cpu(), delimiter_token_mask(tok))
    set_attn_impl(model, "eager")
    try:
        with OutlierProbe(model, residual=False, blocks=False, linears=False, attention=True) as probe:
            probe.run(ids, cats, batch_size=2)
            df = probe.results().attention
    finally:
        set_attn_impl(model, "sdpa")
    cfg = model.config
    assert len(df) == EXPECTED[name]["full"] * cfg.num_attention_heads
    assert df.attn_to_first.between(0, 1).all()
    if name.startswith("qwen3.5"):
        assert df.gate_rest.between(0, 1).all()


def test_eval_and_fake_quant(loaded: tuple[str, PreTrainedModel, PreTrainedTokenizerBase]) -> None:
    _, model, tok = loaded
    ids = _ids(tok)
    labels_loss = float(model(input_ids=ids[:1], labels=ids[:1]).loss)
    nll = token_nll(model, ids[:1])
    assert float(nll.mean()) == pytest.approx(labels_loss, rel=2e-3)

    bl = token_byte_lengths(tok)
    base = evaluate_lm(model, ids, bl)
    assert base["n_bytes"] == int(bl[ids[:, 1:].cpu()].sum())
    before = {n: p.clone() for n, p in model.named_parameters()}
    with fake_quantize(model, SCHEMES["W8A8"]):
        w8 = evaluate_lm(model, ids, bl)
    with fake_quantize(model, SCHEMES["W4A4"]):
        w4 = evaluate_lm(model, ids, bl)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
    assert abs(w8["nll"] - base["nll"]) < 0.1
    assert w4["nll"] > base["nll"]
    # sanity: chunked CE equals direct CE on a short sequence
    logits = model(input_ids=ids[:1]).logits.float()
    direct = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
    assert torch.allclose(nll[0], direct, atol=2e-3)
