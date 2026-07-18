# Competitive gaps — where we lag vs beating the field

**Purpose:** Explicit inventory of **lacks** relative to (1) industry same-class baselines, (2) open small-model capability norms, (3) our own SOTA standards.  
**Not:** “lose to GPT-4o on everything” — that is a different weight class.  
**Yes:** gaps we must close to **dominate open, same-class, pure-FSOT architecture SOTA** on this hardware.

Live position: [`CURRENT_STATUS.md`](CURRENT_STATUS.md) · Standards: [`SOTA_STANDARDS.md`](SOTA_STANDARDS.md)

---

## Competitor classes (who we measure against)

| Class | Who | Fair comparison? |
|-------|-----|------------------|
| **C1 Industry host** | Same SmolLM2-135M with stock SDPA attention | **Yes** — primary bar |
| **C2 Open small models** | Other ≤200M open instruct models on ARC/GSM | **Yes** — same-class capability |
| **C3 Industry attention kernels** | FlashAttention / fused SDPA at all sequence lengths | **Yes** — operator track |
| **C4 Closed frontier** | GPT-class, Claude-class, large open 7B+ | **No** as sole bar; aspirational only |
| **C5 Our standards** | Verify + overfit + held-out gates | **Yes** — process bar |

---

## Gap inventory

### G-CAP — Capability (open packs)

| # | Gap | Us (best) | Competitor bar | Severity | Blocks “beat competitors”? |
|---|-----|-----------|----------------|----------|----------------------------|
| C1 | **GSM free exact** (honest ####) | **0%** | HF ~2% same host; open small models often low but **>0** with format | **CRITICAL** | Yes — any math leaderboard |
| C2 | **GSM free first-digit / soup** | free first ~30%; mode soup | Non-collapsed gen | **CRITICAL** | Yes |
| C3 | **Digit after space** (true digit skill) | **35%** (was 30%); argmax-`1` ~80% | Uniform digit ~10% random; skilled ≫50% | **HIGH** | Yes — path to free exact |
| C4 | **ARC free letter diversity** | ~**80% D** free-gen | Balanced A–D | **HIGH** | Yes — inflated/fragile ARC |
| C5 | **ARC hold min** vs open SOTA small | **~32.5%** ≫ HF 8% | Strong small models often higher on ARC-Easy | **MEDIUM** | Partial win vs C1; lag open SOTA |
| C6 | **ARC 3-eval hold >35%** | Not held | Stable accuracy | **MEDIUM** | Yes for publish confidence |
| C7 | **MATH / multi-step** | Near-zero / untracked as primary | Needed for open math narrative | **HIGH** | Yes |
| C8 | **MMLU / broad knowledge** | Smoke only / not primary | Industry default | **MEDIUM** | Open-source narrative |
| C9 | **Long free-gen quality** | Partial; multi-token clone harsh | Fluent gen | **MEDIUM** | User-facing demos |

### G-SPEED — Attention / systems

| # | Gap | Us | Competitor | Severity |
|---|-----|-----|------------|----------|
| S1 | **Mid-S attention** (S~256–2048) | FSOT slower than fused SDPA | FlashAttention-class | **HIGH** |
| S2 | **End-to-end decode TPS** | Small win (~1.04–1.06×) | Need clearer margin | **MEDIUM** |
| S3 | **Prefill at short S** | Win in scoreboard | Defend | LOW (holding) |
| S4 | **Long-S** (S≥4096) | **WIN** | Hold and extend | LOW (won) |

### G-SCALE — Model / data

| # | Gap | Us | Competitor | Severity |
|---|-----|-----|------------|----------|
| M1 | **Parameter class** | 135M only | Open 360M–7B pure-FSOT stack not yet | **HIGH** for open SOTA narrative |
| M2 | **Training data breadth** | GSM/ARC + packs; curriculum open | Industry multi-domain | **MEDIUM** |
| M3 | **FSOT 2.1 literacy curriculum** | Partial / not primary gate | Own north star | **MEDIUM** |
| M4 | **Instruction following general** | Host-limited | Larger instruct models | **MEDIUM** |

### G-PROCESS — Measurement & ops

| # | Gap | Us | Competitor | Severity |
|---|-----|-----|------------|----------|
| P1 | **Eval noise** (ARC boot ±9%) | Mitigated by multi-rep | Larger public evals | **MEDIUM** |
| P2 | **Automated refine loop** | **Building now** | CI + continuous train elsewhere | **HIGH** (this work) |
| P3 | **Full archive cross-proof every train** | Bridge light + stamp; full suite hours | Archive has full runner | **MEDIUM** (use light always, full nightly) |
| P4 | **Public ckpt release** | Local gitignored | HuggingFace-style | **MEDIUM** |

### G-THEORY-BIND — FSOT correctness under train

| # | Gap | Us | Bar | Severity |
|---|-----|-----|-----|----------|
| T1 | Host remains pure FSOT after train | Checked (V6) | Always all-layer FSOT | LOW if V6 stays green |
| T2 | Seed θ drift under train | V3 alignment | Match archive | LOW if V3 green |
| T3 | Overfit as fake progress | `accept_update` | gen_score up | **HIGH** if ignored |

---

## What we already beat (do not regress)

| Area | Status |
|------|--------|
| Pure FSOT all-layer fidelity (agree 100%) | **Lead / equal** |
| Prefill / long-S attention | **Lead** vs SDPA on this lab |
| ARC vs same-class HF baseline | **Lead** (~4× hold min) |
| FSOT 2.1 verification bind | **Lead** (process) |
| Overfit-aware promote | **Lead** (process) |

---

## Priority order for the auto-refine loop

| Priority | Gap IDs | Lever name (loop) |
|----------|---------|-------------------|
| 1 | C2, C3 | `digit_decollapse` |
| 2 | C4, C5, C6 | `arc_letter_balance` |
| 3 | C1 | `gsm_free_exact` (after digit uncollapse) |
| 4 | S1 | `mid_s_attention` (bench + kernel; not LM CE) |
| 5 | M1–M3 | `scale_host` / `curriculum` (scheduled) |

---

## Success definition: “beating competitors”

**Same-class open win (preregistered):**

1. G1–G4 still PASS  
2. ARC min hold **>35%** with 3-eval stability and gen_score not down  
3. GSM free exact **>0%** under honest ####, then climb  
4. Digit-after-space **>50%**, argmax-`1` **<40%**  
5. Verify green (FSOT-GPU bridge + archive light cross-proof)  
6. No overfit promote (accept_update)

When 1–6 hold on public ledgers, we claim **open same-class pure-FSOT capability lead** on this hardware — not frontier-closed SOTA.
