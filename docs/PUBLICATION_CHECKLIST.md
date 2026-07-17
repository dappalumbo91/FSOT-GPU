# Publication checklist (GitHub when ready)

## License (choose one)

- [ ] **Apache-2.0** (recommended for systems/GPU + patent grant)  
- [ ] **MIT** (simpler)  

Draft files: `LICENSE.Apache-2.0.txt`, `LICENSE.MIT.txt` — activate one as `LICENSE` on release.

## Content gates

- [ ] README states theory author, archive authority, what is *not* included (2.1 Instruct)  
- [ ] `docs/HOW_IT_WORKS.md` matches code  
- [ ] `docs/COMPETITIVE_POSITION.md` claims only measured facts  
- [ ] `parity/run_parity.py` green on release machine  
- [ ] No secrets, no private API keys, no third-party model weights  
- [ ] No accidental copy of FSOT-2.1-Instruct adapters/safetensors  
- [ ] Cite / link [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) as theory spine  

## Engineering gates

- [ ] `python -m fsot_lib.smoke_owned`  
- [ ] `python parity/run_parity.py`  
- [ ] `scripts/build_cuda_kernels.ps1` (where NVIDIA GPU present)  
- [ ] `lake build` in `phase1_formal_gpu/lean`  
- [ ] CI (GitHub Actions) for pure Python + Lean when public  

## Release steps (when you say go)

1. Create repo (suggested name: `FSOT-Formal-GPU` or `fsot-gpu`)  
2. Copy lab tree excluding large binaries if needed; keep small test `.exe` optional  
3. Set `LICENSE`  
4. Tag `v0.1.0-research`  
5. Announce with parity ledger JSON attached  

**Do not** create/push the public repo until you explicitly confirm license + go.
