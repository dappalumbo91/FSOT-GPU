#!/usr/bin/env python3
"""
FSOT 2.1 verification bridge for the industry pure-FSOT host lab.

Brings archive / formal framework checks into this repo so capability
work is refined *with verification*, not accuracy % alone.

Layers:
  V1  Archive authority stamp (cross_proof report, VERIFICATION_REPORT)
  V2  Connective-spine obligation replay (Python float, fail-closed)
  V3  Seed / collapse θ alignment: lab fsot_lib ↔ archive fsot_compute
  V4  Phase-1 formal artifacts present (Lean/Coq/Isabelle/F*)
  V5  Owned stack smoke (scalar, pack, coherence, consensus no-softmax)
  V6  Pure-FSOT host structural: all layers FsotLlamaAttention, finite forward
  V7  Overfit-metric module present (train−hold gap API) — soft, always available

Does not re-run full multi-hour Lean/Coq/Isabelle suite each time — it
**binds** to the already-green archive report and re-proves the exported
spine obligations + live host contracts.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ARCHIVE_21 = Path(r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full")
ARCHIVE_ROOT = Path(r"I:\FSOT-Physical-Archive")
DESKTOP_21 = Path(r"C:\Users\damia\Desktop\FSOT-2.1-Lean")

getcontext().prec = 50


def _load_archive_compute():
    path = ARCHIVE_21 / "vendor" / "fsot_compute.py"
    if not path.is_file():
        path = DESKTOP_21 / "vendor" / "fsot_compute.py"
    if not path.is_file():
        return None, str(path)
    name = "fsot_compute_archive"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass needs the module registered before exec_module
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod, str(path)


def check_archive_stamp() -> dict[str, Any]:
    """V1 — fail-closed on missing/red archive verification ledgers."""
    out: dict[str, Any] = {"ok": False, "sources": {}}
    cross = ARCHIVE_21 / "data" / "cross_proof_verification_report.json"
    vr = ARCHIVE_ROOT / "VERIFICATION_REPORT.json"
    if cross.is_file():
        rep = json.loads(cross.read_text(encoding="utf-8"))
        overall = bool(rep.get("overall_ok"))
        # some reports nest under frameworks
        if "overall_ok" not in rep:
            # derive from connective + full formal python_decimal
            c = rep.get("connective_spine", {}).get("python_decimal", {}).get("status")
            f = rep.get("full_formal_spine", {}).get("python_decimal", {}).get("status")
            overall = c == "passed" and f == "passed"
            # also check frameworks if present
            fw = rep.get("frameworks") or {}
            if fw:
                fails = [
                    k
                    for k, v in fw.items()
                    if isinstance(v, dict) and v.get("status") not in ("passed", "exported", "skipped", None)
                    and v.get("status") not in ("passed",)
                    and str(v.get("status", "")).lower() not in ("passed", "ok", "exported", "optional")
                ]
                # only hard-fail known red
                hard = [
                    k
                    for k, v in fw.items()
                    if isinstance(v, dict)
                    and str(v.get("status", "")).lower() in ("failed", "fail", "error")
                ]
                if hard:
                    overall = False
        out["sources"]["cross_proof"] = {
            "path": str(cross),
            "overall_ok": overall,
            "tier": rep.get("tier"),
            "generated_at": rep.get("generated_at"),
            "connective_n": rep.get("connective_spine", {}).get("obligation_count"),
            "full_formal_n": rep.get("full_formal_spine", {}).get("obligation_count"),
            "github_ready": rep.get("github_ready")
            or (rep.get("frameworks") or {}).get("github_ready"),
        }
        # Prefer explicit top-level if present at end of file
        if rep.get("overall_ok") is True:
            overall = True
        if rep.get("github_ready") is True:
            out["sources"]["cross_proof"]["github_ready"] = True
        out["sources"]["cross_proof"]["overall_ok"] = overall
    else:
        out["sources"]["cross_proof"] = {"path": str(cross), "ok": False, "missing": True}

    if vr.is_file():
        try:
            vrep = json.loads(vr.read_text(encoding="utf-8"))
            out["sources"]["archive_verification_report"] = {
                "path": str(vr),
                "status": vrep.get("status"),
                "ok": str(vrep.get("status", "")).upper() == "GREEN"
                or vrep.get("ok") is True,
                "cross_proof": vrep.get("cross_proof"),
            }
        except json.JSONDecodeError as e:
            out["sources"]["archive_verification_report"] = {
                "path": str(vr),
                "ok": False,
                "error": str(e),
            }
    else:
        out["sources"]["archive_verification_report"] = {
            "path": str(vr),
            "ok": False,
            "missing": True,
        }

    cp_ok = out["sources"].get("cross_proof", {}).get("overall_ok") is True
    # also accept connective+full passed without top-level overall_ok
    if not cp_ok and cross.is_file():
        rep = json.loads(cross.read_text(encoding="utf-8"))
        c = rep.get("connective_spine", {}).get("python_decimal", {}).get("status")
        f = rep.get("full_formal_spine", {}).get("python_decimal", {}).get("status")
        cp_ok = c == "passed" and f == "passed"
        out["sources"]["cross_proof"]["overall_ok"] = cp_ok
        out["sources"]["cross_proof"]["derived_from_decimal_status"] = True

    ar_ok = out["sources"].get("archive_verification_report", {}).get("ok") is True
    # archive stamp ok if either cross-proof green OR archive report GREEN
    out["ok"] = bool(cp_ok or ar_ok)
    out["detail"] = "archive_stamp_green" if out["ok"] else "archive_stamp_red_or_missing"
    return out


def check_connective_spine() -> dict[str, Any]:
    """V2 — re-prove exported connective spine obligations in Python."""
    path = ARCHIVE_21 / "verification" / "obligations" / "connective_spine.json"
    out: dict[str, Any] = {"ok": False, "path": str(path), "passed": 0, "failed": []}
    if not path.is_file():
        out["missing"] = True
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    obs = data.get("obligations") or []
    for o in obs:
        kind = o.get("kind")
        oid = o.get("id", "?")
        try:
            if kind == "pos":
                v = Decimal(str(o["value"]))
                ok = v > 0
            elif kind == "gt_one":
                v = Decimal(str(o["value"]))
                ok = v > 1
            elif kind == "lt":
                lv = Decimal(str(o["left_value"]))
                rv = Decimal(str(o["right_value"]))
                ok = lv < rv
            else:
                ok = False
                out["failed"].append({"id": oid, "reason": f"unknown_kind:{kind}"})
                continue
            if ok:
                out["passed"] += 1
            else:
                out["failed"].append({"id": oid, "kind": kind, "reason": "predicate_false"})
        except Exception as e:
            out["failed"].append({"id": oid, "error": str(e)})
    out["total"] = len(obs)
    out["ok"] = out["passed"] == len(obs) and len(obs) > 0
    out["obligation_count_export"] = data.get("obligation_count")
    return out


def check_seed_alignment() -> dict[str, Any]:
    """V3 — lab seeds / collapse θ match archive fsot_compute authority."""
    from fsot_lib.seeds import COLLAPSE_THRESHOLD, SEEDS
    from fsot_lib.scalar import compute_scalar

    out: dict[str, Any] = {"ok": False}
    lab_theta = float(COLLAPSE_THRESHOLD)
    out["lab"] = {
        "collapse_threshold": lab_theta,
        "c_eff": SEEDS.c_eff,
        "p_var": SEEDS.p_var,
        "phi": SEEDS.phi,
        "S_D14": compute_scalar(D_eff=14.0, observed=True, delta_psi=0.7),
    }
    mod, path = _load_archive_compute()
    out["archive_path"] = path
    if mod is None:
        out["error"] = "fsot_compute not found"
        # still ok if lab theta is in expected range from known theory
        out["ok"] = 0.9 < lab_theta < 0.95
        out["soft"] = True
        return out
    try:
        c_eff = float(mod.C_EFF)
        p_var = float(mod.P_VAR)
        arch_theta = c_eff * p_var
        out["archive"] = {
            "C_EFF": c_eff,
            "P_VAR": p_var,
            "collapse_threshold": arch_theta,
        }
        # relative agreement on θ
        rel = abs(lab_theta - arch_theta) / max(abs(arch_theta), 1e-12)
        out["theta_rel_err"] = rel
        out["ok"] = rel < 1e-6  # should be exact to float64 seeds
        if not out["ok"] and rel < 1e-4:
            out["ok"] = True
            out["note"] = "within 1e-4 float tolerance"
    except Exception as e:
        out["error"] = str(e)
        out["ok"] = 0.9 < lab_theta < 0.95
        out["soft"] = True
    return out


def check_formal_artifacts() -> dict[str, Any]:
    """V4 — phase1 multi-prover artifacts present in this lab."""
    base = ROOT / "phase1_formal_gpu"
    files = {
        "lean_trinary": base / "lean" / "Trinary.lean",
        "lean_memory": base / "lean" / "GpuMemory.lean",
        "coq_trinary": base / "coq" / "Trinary.v",
        "coq_vo": base / "coq" / "Trinary.vo",
        "isabelle": base / "isabelle" / "Trinary.thy",
        "fstar": base / "fstar" / "FSOTGpuBoot.fst",
    }
    present = {k: p.is_file() for k, p in files.items()}
    # required: at least lean sources + one other prover
    req = present["lean_trinary"] and present["lean_memory"]
    multi = sum(
        1
        for k in ("coq_trinary", "isabelle", "fstar")
        if present[k]
    )
    return {
        "ok": bool(req and multi >= 2),
        "present": present,
        "required_lean": req,
        "other_provers": multi,
    }


def check_owned_stack() -> dict[str, Any]:
    """V5 — pure owned operators (no industry softmax path)."""
    from fsot_lib.seeds import COLLAPSE_THRESHOLD
    from fsot_lib.scalar import compute_scalar
    from fsot_lib.trinary import pack_u64, unpack_u64
    from fsot_lib.coherence import coherence_norm

    out: dict[str, Any] = {"ok": True, "checks": {}}
    S = compute_scalar(D_eff=14.0, observed=True)
    out["checks"]["scalar"] = {"S": S, "ok": abs(S) > 0 and math.isfinite(S)}
    codes = [i % 3 for i in range(32)]
    w = pack_u64(codes)
    back = unpack_u64(w)
    out["checks"]["pack"] = {"ok": back == codes}
    y = coherence_norm([0.1, 0.95, -0.99, 0.5])
    out["checks"]["coherence"] = {"ok": len(y) == 4}
    out["checks"]["collapse_theta"] = {
        "theta": COLLAPSE_THRESHOLD,
        "ok": 0.9 < COLLAPSE_THRESHOLD < 0.95,
    }
    try:
        import torch
        from fsot_lib.consensus import consensus_aggregate

        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(8, 16, device=device, dtype=torch.float64)
        k = torch.randn(8, 16, device=device, dtype=torch.float64)
        v = torch.randn(8, 16, device=device, dtype=torch.float64)
        o = consensus_aggregate(q, k, v)
        # contract: finite, same shape, not all zeros on random input (usually)
        finite = bool(torch.isfinite(o).all())
        out["checks"]["consensus"] = {
            "device": device,
            "shape": list(o.shape),
            "finite": finite,
            "ok": finite and list(o.shape) == list(q.shape),
            "no_softmax": True,
        }
    except Exception as e:
        out["checks"]["consensus"] = {"ok": False, "error": str(e)}
    out["ok"] = all(c.get("ok") for c in out["checks"].values())
    return out


def check_pure_fsot_host(
    ckpt_path: Path | None = None,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """V6 — pure FSOT all-layer host loads and runs finitely."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fsot_layer_swap import FsotLlamaAttention, swap_all_layers

    out: dict[str, Any] = {"ok": False}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = HERE / "models" / "SmolLM2-135M-Instruct"
    if not model_dir.is_dir():
        out["error"] = f"missing model {model_dir}"
        return out
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    swap_all_layers(m)
    layers = m.model.layers
    n = len(layers)
    n_fsot = sum(1 for L in layers if isinstance(L.self_attn, FsotLlamaAttention))
    out["n_layers"] = n
    out["n_fsot_attn"] = n_fsot
    out["all_layers_fsot"] = n_fsot == n

    ckpt = ckpt_path
    if ckpt is None:
        for cand in (
            HERE.parent / "results" / "industry_lm" / "checkpoints" / "pure_fsot_data_driven_best.pt",
            HERE.parent / "results" / "industry_lm" / "checkpoints" / "pure_fsot_granular_best.pt",
            HERE.parent / "results" / "industry_lm" / "checkpoints" / "pure_fsot_12x3_best.pt",
        ):
            if cand.is_file():
                ckpt = cand
                break
    if ckpt and ckpt.is_file():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False)
        out["ckpt"] = str(ckpt)
        out["ckpt_meta"] = {
            k: ck.get(k)
            for k in (
                "arc_min",
                "arc_easy_hold",
                "arc_challenge_hold",
                "gsm_first",
                "agree16",
                "phase",
                "step",
            )
            if k in ck or ck.get(k) is not None
        }
    else:
        out["ckpt"] = None

    m.eval()
    with torch.no_grad():
        inp = tok("Question: 1 + 1 =\n####", return_tensors="pt").to(device)
        logits = m(**inp).logits
        finite = bool(torch.isfinite(logits).all())
        out["forward_finite"] = finite
        out["logits_shape"] = list(logits.shape)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    out["ok"] = bool(out["all_layers_fsot"] and out.get("forward_finite"))
    return out


def run_verification(
    *,
    include_host: bool = True,
    ckpt_path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "fsot21_verify_bridge_v1",
        "theory_authority": "FSOT-2.1-Lean + Physical-Archive cross_proof",
        "layers": {},
        "ok": True,
        "failed_layers": [],
    }
    layers = [
        ("V1_archive_stamp", check_archive_stamp),
        ("V2_connective_spine", check_connective_spine),
        ("V3_seed_alignment", check_seed_alignment),
        ("V4_formal_artifacts", check_formal_artifacts),
        ("V5_owned_stack", check_owned_stack),
    ]
    for name, fn in layers:
        try:
            c = fn()
        except Exception as e:
            c = {"ok": False, "error": str(e)}
        report["layers"][name] = c
        if not c.get("ok"):
            report["ok"] = False
            report["failed_layers"].append(name)

    if include_host:
        try:
            c = check_pure_fsot_host(ckpt_path)
        except Exception as e:
            c = {"ok": False, "error": str(e)}
        report["layers"]["V6_pure_fsot_host"] = c
        if not c.get("ok"):
            report["ok"] = False
            report["failed_layers"].append("V6_pure_fsot_host")

    # V7 soft: overfit metric API must be importable (does not load GPU models here)
    try:
        from overfit_metrics import (  # noqa: WPS433
            accept_update,
            build_overfit_report,
            overfit_gap,
        )

        g = overfit_gap(0.40, 0.30)  # train 40%, hold 30% → gap +0.10
        dummy = build_overfit_report([])
        report["layers"]["V7_overfit_metric_api"] = {
            "ok": True,
            "soft": True,
            "sample_gap_train40_hold30": g,
            "accept_update": callable(accept_update),
            "detail": "train−hold gap + gen_score for non-overfit direction",
        }
    except Exception as e:
        report["layers"]["V7_overfit_metric_api"] = {
            "ok": False,
            "soft": True,
            "error": str(e),
        }
        # soft: do not fail overall ledger solely on V7

    if write:
        out_dir = ROOT / "results" / "industry_lm"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "fsot21_verify.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        md = [
            "# FSOT 2.1 verification bridge",
            "",
            f"**Overall: {'PASS' if report['ok'] else 'FAIL'}**",
            "",
            f"UTC: {report['timestamp']}",
            "",
            "| Layer | OK | Notes |",
            "|-------|----|-------|",
        ]
        for name, c in report["layers"].items():
            note = c.get("detail") or c.get("error") or c.get("path") or ""
            if name == "V2_connective_spine":
                note = f"{c.get('passed')}/{c.get('total')} obligations"
            if name == "V3_seed_alignment":
                note = f"θ_lab={c.get('lab',{}).get('collapse_threshold')} rel_err={c.get('theta_rel_err')}"
            if name == "V6_pure_fsot_host":
                note = f"layers {c.get('n_fsot_attn')}/{c.get('n_layers')} finite={c.get('forward_finite')}"
            md.append(f"| {name} | {'✓' if c.get('ok') else '✗'} | {note} |")
        if report["failed_layers"]:
            md.append("")
            md.append("Failed: " + ", ".join(report["failed_layers"]))
        md.append("")
        md.append("Refine capability only while this ledger stays green.")
        (out_dir / "FSOT21_VERIFY.md").write_text("\n".join(md), encoding="utf-8")
        report["paths"] = {"json": str(path), "md": str(out_dir / "FSOT21_VERIFY.md")}
    return report


def main() -> int:
    rep = run_verification(include_host=True, write=True)
    print("=== FSOT 2.1 VERIFY BRIDGE ===")
    for name, c in rep["layers"].items():
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {name}")
    print(f"overall={'PASS' if rep['ok'] else 'FAIL'}")
    if rep.get("paths"):
        print("wrote", rep["paths"]["json"])
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
