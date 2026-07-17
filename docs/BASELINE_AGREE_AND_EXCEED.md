# What “94% vs baseline” means — and how we exceed

## The metric

**Next-token agree** on the SOTA scoreboard is **clone fidelity**, not a raw IQ score:

| Arm | Score meaning |
|-----|----------------|
| Industry baseline vs itself | **100%** by definition (same model) |
| Pure FSOT vs baseline | Fraction of probes where **argmax next-token equals** the industry host |

So **94%** = FSOT picks the same next token as SmolLM2-SDPA on **15/16** probes.  
It does **not** mean “baseline is 6% smarter.” It means “one probe still disagrees.”

### The current miss (example)

| Prompt | Baseline next token | FSOT next token |
|--------|---------------------|-----------------|
| `Water freezes at` | ` night` | ` the` |

The industry model’s greedy token here is a weak / weird completion; FSOT’s is a different path. **Matching baseline 100% is fidelity work. Exceeding baseline is capability work.**

## Two ladders (both FSOT)

### Ladder A — Equal the baseline (fidelity)

1. Drive next-token agree → **100%** on the fixed probe set (and a larger holdout).  
2. Drive KL(base ‖ fsot) down.  
3. Drive multi-token / teacher-forced top-5 up.

**STATUS: HIT** — EVAL16 agree **100%**, KL≈1.75 (`pure_fsot_agree100_best.pt`).  

**Done when:** agree = 100% on EVAL set and holdout does not collapse.

### Ladder B — Exceed the baseline (capability)

Once equal (or in parallel where structure already wins):

| Axis | How FSOT exceeds |
|------|------------------|
| **Speed** | Prefill / decode / long-context attention (already winning some) |
| **Structure** | No softmax `exp`; collapse-gated sparse consensus |
| **Factual / task probes** | Prefer correct answers when baseline is wrong (independent eval, not clone) |
| **Long context** | O(S·A) vs O(S²) attention domain |
| **Generation quality** | Plausible, coherent, task-success — not greedy clone of SDPA |

**Done when:** on a preregistered capability set, FSOT **wins more categories than it loses** vs the same baseline host, with quality gates still green.

## Standing rule

- **94% &lt; 100% on agree** → keep pushing fidelity until equal.  
- **Equal** → fidelity ladder complete; capability ladder is primary.  
- **Already exceeding** on speed/long-context is real and counted; it does not wait for 100% agree.
