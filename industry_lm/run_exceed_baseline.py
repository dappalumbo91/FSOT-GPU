#!/usr/bin/env python3
"""
Ladder B: where pure FSOT *exceeds* industry baseline (not clone score).

Axes:
  - speed (prefill, decode) — already winning on scoreboard
  - long-context attention
  - factual probes where baseline next-token is weak / wrong
  - generation teacher-plausibility

Writes results/industry_lm/exceed_baseline.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from competitive.sparse_consensus_batched import consensus_true_sparse_padded  # noqa: E402
from fsot_cuda_ops import available as cuda_dll, fsot_consensus  # noqa: E402
from fsot_layer_swap import swap_all_layers  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)

# Factual / common-sense prompts where we grade both models independently
# Preferred answer tokens (any of list counts as hit)
FACTUAL = [
    {
        "prompt": "The capital of France is",
        "accept": [" Paris", "Paris"],
    },
    {
        "prompt": "The largest planet in our solar system is",
        "accept": [" Jupiter", "Jupiter"],
    },
    {
        "prompt": "2 + 2 =",
        "accept": [" 4", "4"],
    },
    {
        "prompt": "1 + 1 =",
        "accept": [" 2", "2"],
    },
    {
        "prompt": "The capital of Japan is",
        "accept": [" Tokyo", "Tokyo"],
    },
    {
        "prompt": "The chemical formula for water is",
        "accept": [" H", "H2O", " H2O", " H₂O"],
    },
    {
        "prompt": "Water freezes at",
        "accept": [" 0", "0", " 32", " zero", " 0°", " 0 C", " 0°C"],
    },
    {
        "prompt": "The square root of 9 is",
        "accept": [" 3", "3"],
    },
    {
        "prompt": "The Earth orbits the",
        "accept": [" Sun", " sun", "Sun"],
    },
    {
        "prompt": "The speed of light is approximately",
        "accept": [" 3", "300", " 300", " c", " 186"],
    },
]


def load_base(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    return tok, m


def load_fsot(device):
    tok, m = load_base(device)
    swap_all_layers(m)
    for path in [
        CKPT / "pure_fsot_exceed_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
        CKPT / "pure_fsot_agree_best.pt",
        CKPT / "pure_fsot_push80_best.pt",
    ]:
        if path.is_file():
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            return tok, m, str(path), ck.get("agree16")
    return tok, m, None, None


@torch.no_grad()
def next_tok(tok, model, device, prompt):
    inp = tok(prompt, return_tensors="pt").to(device)
    logits = model(**inp).logits[0, -1]
    tid = int(logits.argmax())
    return tok.decode([tid]), tid, logits


def factual_hit(decoded: str, accept: list[str]) -> bool:
    d = decoded
    for a in accept:
        if d == a or d.strip() == a.strip() or d.startswith(a) or a.strip() in d:
            return True
    return False


@torch.no_grad()
def prefill_ms(tok, model, device, text, iters=30, warmup=6):
    inp = tok(text, return_tensors="pt").to(device)
    for _ in range(warmup):
        _ = model(**inp)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inp)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


@torch.no_grad()
def decode_tps(tok, model, device, prompts, max_new=24):
    rates = []
    for p in prompts:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        new = int(out.shape[-1] - inp["input_ids"].shape[-1])
        rates.append(new / (ms / 1000.0))
    return sum(rates) / len(rates)


@torch.no_grad()
def attn_long(device, S=4096, H=9, D=64, iters=40):
    q = torch.randn(1, H, S, D, device=device)
    k = torch.randn(1, H, S, D, device=device)
    v = torch.randn(1, H, S, D, device=device)

    def sdpa():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def fsot():
        if cuda_dll():
            return fsot_consensus(q, k, v)
        return torch.stack([consensus_true_sparse_padded(q[0], k[0], v[0])], 0)

    def bench(fn, n=iters, w=8):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return 1000.0 * (time.perf_counter() - t0) / n

    ms_s, ms_f = bench(sdpa), bench(fsot)
    return {
        "S": S,
        "sdpa_ms": ms_s,
        "fsot_ms": ms_f,
        "speedup": ms_s / max(ms_f, 1e-12),
        "fsot_wins": ms_f < ms_s * 0.95,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== LADDER B: exceed baseline (capability) ===")

    tok, base = load_base(device)
    tok2, fsot, ckpt, meta = load_fsot(device)
    print("ckpt", ckpt, "meta_agree", meta)

    # Fidelity (for context)
    agree = 0
    for item in FACTUAL:
        b, _, _ = next_tok(tok, base, device, item["prompt"])
        f, _, _ = next_tok(tok2, fsot, device, item["prompt"])
        agree += int(b == f)
    fidelity = agree / len(FACTUAL)

    # Independent factual hits
    fact_rows = []
    base_hits = fsot_hits = 0
    for item in FACTUAL:
        bdec, _, _ = next_tok(tok, base, device, item["prompt"])
        fdec, _, _ = next_tok(tok2, fsot, device, item["prompt"])
        bh = factual_hit(bdec, item["accept"])
        fh = factual_hit(fdec, item["accept"])
        base_hits += int(bh)
        fsot_hits += int(fh)
        fact_rows.append(
            {
                "prompt": item["prompt"],
                "base": bdec,
                "fsot": fdec,
                "base_hit": bh,
                "fsot_hit": fh,
                "fsot_better": fh and not bh,
                "base_better": bh and not fh,
            }
        )
        tag = (
            "FSOT_BETTER"
            if fh and not bh
            else ("BASE_BETTER" if bh and not fh else ("BOTH" if bh and fh else "NEITHER"))
        )
        print(f"  [{tag}] {item['prompt']!r} base={bdec!r} fsot={fdec!r}")

    # Speed
    pb = prefill_ms(tok, base, device, "The capital of France is")
    pf = prefill_ms(tok2, fsot, device, "The capital of France is")
    prompts = [x["prompt"] for x in FACTUAL[:5]]
    db = decode_tps(tok, base, device, prompts)
    df = decode_tps(tok2, fsot, device, prompts)
    long = attn_long(device, S=4096) if device == "cuda" else None
    long8 = attn_long(device, S=8192) if device == "cuda" else None

    wins = []
    loses = []
    ties = []

    def cmp(name, fsot_better, base_better):
        if fsot_better:
            wins.append(name)
        elif base_better:
            loses.append(name)
        else:
            ties.append(name)

    cmp("factual_hit_rate", fsot_hits > base_hits, base_hits > fsot_hits)
    cmp("prefill", pf < pb * 0.97, pb < pf * 0.97)
    cmp("decode_tps", df > db * 1.03, db > df * 1.03)
    if long:
        cmp("attn_S4096", long["fsot_wins"], long["speedup"] < 0.95)
    if long8:
        cmp("attn_S8192", long8["fsot_wins"], long8["speedup"] < 0.95)

    # cases where FSOT is factually better
    fsot_better_n = sum(1 for r in fact_rows if r["fsot_better"])
    base_better_n = sum(1 for r in fact_rows if r["base_better"])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ladder": "B_exceed_baseline",
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "ckpt": ckpt,
        "meta_agree": meta,
        "fidelity_on_factual_set": fidelity,
        "factual": {
            "base_hits": base_hits,
            "fsot_hits": fsot_hits,
            "n": len(FACTUAL),
            "base_rate": base_hits / len(FACTUAL),
            "fsot_rate": fsot_hits / len(FACTUAL),
            "fsot_better_cases": fsot_better_n,
            "base_better_cases": base_better_n,
            "rows": fact_rows,
        },
        "speed": {
            "prefill_base_ms": pb,
            "prefill_fsot_ms": pf,
            "prefill_speedup": pb / max(pf, 1e-12),
            "decode_base_tps": db,
            "decode_fsot_tps": df,
            "decode_speedup": df / max(db, 1e-12),
            "attn_long": long,
            "attn_longer": long8,
        },
        "verdict": {
            "wins": wins,
            "ties": ties,
            "loses": loses,
            "fsot_exceeds_on": wins,
            "note": (
                "Exceed = more capability wins than losses vs same baseline host. "
                "Fidelity (clone agree) is Ladder A and caps at 100%."
            ),
        },
        "ok": True,
    }
    path = OUT / "exceed_baseline.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = OUT / "EXCEED_BASELINE.md"
    md.write_text(
        f"""# Ladder B — exceed industry baseline (FSOT)

**Not** “agree &gt; 100%”. Agree maxes at **100% = equal fidelity**.  
**Exceed** = win capability axes on the same GPU / tiny model.

## Verdict

| | |
|--|--|
| Wins | {', '.join(wins) or '—'} |
| Ties | {', '.join(ties) or '—'} |
| Loses | {', '.join(loses) or '—'} |
| Fidelity on factual set | {fidelity:.0%} (clone match, not capability) |
| Factual hits base / FSOT | {base_hits}/{len(FACTUAL)} vs **{fsot_hits}/{len(FACTUAL)}** |
| Prefill | {pb:.2f} → **{pf:.2f} ms** ({pb/max(pf,1e-12):.2f}×) |
| Decode | {db:.1f} → **{df:.1f} t/s** ({df/max(db,1e-12):.2f}×) |

Ledger: `exceed_baseline.json`
""",
        encoding="utf-8",
    )
    print("=== EXCEED SUMMARY ===")
    print("wins", wins)
    print("loses", loses)
    print("ties", ties)
    print("factual base", base_hits, "fsot", fsot_hits)
    print("wrote", path, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
