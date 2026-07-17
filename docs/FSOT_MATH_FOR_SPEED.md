# FSOT mathematics used to attack wall-clock (archive authority)

Source drive: `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full`  
Primary: `FSOT/Scalar.lean`, `vendor/fsot_compute.py`, trinary priors, kernel `lattice.rs`.

## Constants (not free knobs)

| Symbol | Formula / value | Speed use |
|--------|-----------------|-----------|
| `C_eff` | coherence efficiency (Scalar.lean) | Collapse / density of sharp trits |
| `P_var` | phase variance | With C_eff → **collapse threshold** |
| `θ_coll` | `C_eff · P_var ≈ 0.917466` | Fraction of lanes that are non-superposed |
| Gate 0.5 | kernel consensus | **Sparse key set** — skip dead keys |
| `1/√D_eff` | term1 base | Cost scales with √D; keep D honest |
| Ignition | `φ/(1+φ) / eq ≈ 0.3922` | Optional soft ignition floor (fluid priors) |
| Metatron | `3³ = 27` | Tile / word width alignment |
| `φ/(1+φ)` | complexity weight | Complexity budget motif |

## Asymptotic win (theory)

Softmax SDPA: always **Θ(H · S² · D)** with `exp`.

FSOT consensus (kernel-faithful):

1. Coherence of each key: **Θ(H · S · D)**  
2. Active keys `A = #{keys : coh > 0.5}`  
3. Trit-sim + gather only on active (causal): **Θ(H · S · A · D)**  
4. No `exp`

On N(0,1) QKV with `θ_coll ≈ 0.917`, measured A/S ≈ **0.6%–9%**.  
So wall-clock should **beat** dense SDPA when A ≪ S and the sparse path is implemented (not a full S×S materialization).

## Implementation map

| Math | Code |
|------|------|
| Sparse active keys | `competitive/sparse_consensus.py` |
| Collapse θ | `fsot_lib.seeds.COLLAPSE_THRESHOLD` |
| Trit pack density | CUDA `trinary_pack` + `fsot_lib.trinary` |
| Scalar 1/√D | informs tile choice; not a free reduce-D cheat |
