# Ladder B — exceed industry baseline (FSOT)

**Not** “agree &gt; 100%”. Agree maxes at **100% = equal fidelity**.  
**Exceed** = win capability axes on the same GPU / tiny model.

## Verdict

| | |
|--|--|
| Wins | prefill, decode_tps, attn_S4096, attn_S8192 |
| Ties | — |
| Loses | factual_hit_rate |
| Fidelity on factual set | 90% (clone match, not capability) |
| Factual hits base / FSOT | 5/10 vs **4/10** |
| Prefill | 43.32 → **38.91 ms** (1.11×) |
| Decode | 30.9 → **37.7 t/s** (1.22×) |

Ledger: `exceed_baseline.json`
