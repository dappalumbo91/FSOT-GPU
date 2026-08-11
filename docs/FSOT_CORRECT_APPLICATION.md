# Correct FSOT application to the LLM (no single-axis theater)

## What we got wrong

Recent runs “improved” one barrier while **regressing the package**:

| Mistake | What actually happened |
|---------|------------------------|
| Digit-only / ARC-only phases | One skill rose, another fell — that is **not** more intelligence per parameter |
| Promoting on dig_score alone | Mode mass dropped while space-digit or ARC stayed flat/down |
| Overwriting production host | Low-ARC lab replaced a stronger package (capability density destroyed) |
| Calling sequential barrier scripts “FSOT” | FSOT is **simultaneous structure** (operators + multi-floor gates), not whack-a-mole |

If any frozen floor **regresses**, the model was **not** applied correctly to the problem we stated:

> **Low-parameter SOTA capability on constrained hardware via FSOT substrate.**

A regression is evidence we optimized a **proxy**, not the **package**.

---

## What “FSOT applied correctly” means here

### Substrate (must follow FSOT math)

- Seeds / collapse θ / consensus attention / pin bind (G-VERIFY)
- Suction–poof LR (no free Adam fishing)
- Coherence / overfit accept as **physical** gates, not afterthoughts

### Host weights (free, but guided)

- Ordinary tensors on open 135M host
- Updates only when the **joint package** does not regress

### Package (the actual solve target)

Every promote must clear **all** floors relative to the **spine baseline**:

| Floor | Meaning |
|-------|---------|
| G-VERIFY | Pin / ops green |
| Agree16 | ≥ max(90%, spine) |
| ARC min | ≥ spine − 0 (hard: no drop) |
| gen_score | ≥ spine − ε |
| Space digit | ≥ spine − ε |
| Mode mass | ≤ max(spine, 0.50) and not re-collapse to one digit |
| Overfit | `accept_update` true |

**Improve** = at least one climb axis up **and** zero floor breaks.  
**Not improve** = single-axis up with any floor break → **restore** (regression).

---

## Operating law (hard)

1. **Spine host** = highest joint package on disk (from `audit_host_package.py`), not the last script that wrote a file.  
2. **One joint train path** — every step mixes ARC + digit + teacher retention under FSOT LR.  
3. **No lab overwrite of production** unless package ≥ spine and ARC min ≥ production floor.  
4. **Sequential barrier scripts are secondary** — they may diagnose, not define success.  
5. **Report package deltas**, never a lone percent.

---

## Implementation

| Tool | Role |
|------|------|
| `industry_lm/audit_host_package.py` | Measure all hosts; rank package |
| `industry_lm/run_joint_package_climb.py` | Only legal train path for capability density |
| `docs/HARDWARE_CONSTRAINT_STRATEGY.md` | Why 135M + FSOT beats width under VRAM |

---

## One line

**FSOT is applied correctly when the pure-FSOT 135M host gains held-out package without regressing any frozen floor — not when a barrier script prints a single green number.**
