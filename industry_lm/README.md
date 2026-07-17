# Industry LM unit — code-agnostic portable weights

**Purpose:** Bring a **small, industry-standard** Hugging Face model (SafeTensors) into the FSOT Formal-GPU lab so we can:

1. Benchmark **real LLM-shaped** workloads (not only synthetic QKV).  
2. Keep weights in a **language-neutral** bank (SafeTensors + JSON IR).  
3. Prove the path: industry model → portable schema → FSOT GPU ops / Rust / Zig / formal contracts.  
4. **Not** touch the separate FSOT-2.1-Instruct project.

## Model choice

| Model | Why |
|-------|-----|
| **HuggingFaceTB/SmolLM2-135M-Instruct** | Modern, ~135M params, fits easily on RTX 5070, SafeTensors, instruct-tuned, widely used micro-LLM baseline |

## Layout

```
industry_lm/
  models/SmolLM2-135M-Instruct/   # HF snapshot (safetensors)
  portable/                       # language-neutral export
  portable_schema.json            # IR describing tensors + graph ops
  load_bank.py                    # load safetensors → bank
  export_portable.py              # write portable bank + IR
  baseline_hf.py                  # industry path (transformers)
  fsot_bridge.py                  # same weights; FSOT ops where swapped
  run_unit.py                     # full unit: export + bench + smoke gen
```

## Code-agnostic idea

```
HuggingFace SafeTensors  →  Portable Weight Bank (names, shapes, dtype, file offsets)
                         →  Portable Graph IR (embed, rms_norm, attn, mlp, lm_head)
                         →  Backend: torch | FSOT-CUDA | (later) Rust/Zig host
```

Any language that can mmap SafeTensors and implement the IR ops can host the model. FSOT competition is about **replacing IR nodes** (attn, norm) with verified FSOT primitives while keeping the same bank.

## Run

```powershell
cd "...\gpu exparment for lean coq isabell andf star"
python industry_lm\run_unit.py
```
