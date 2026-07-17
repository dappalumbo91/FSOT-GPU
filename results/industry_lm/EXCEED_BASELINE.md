# Ladder B — exceed industry baseline (FSOT)

**Not** “agree &gt; 100%”. Agree maxes at **100% = equal fidelity**.  
**Exceed** = win capability axes on the same GPU / tiny model.

## Verdict

| | |
|--|--|
| Wins | prefill, attn_S8192 |
| Ties | factual_hit_rate, attn_S4096 |
| Loses | decode_tps |
| Fidelity on factual set | 90% (clone match, not capability) |
| Factual hits base / FSOT | 5/10 vs **5/10** |
| Prefill | 29.84 → **26.55 ms** (1.12×) |
| Decode | 33.7 → **30.3 t/s** (0.90×) |

Ledger: `exceed_baseline.json`
