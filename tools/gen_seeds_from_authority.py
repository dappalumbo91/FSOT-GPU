#!/usr/bin/env python3
"""Generate the seed-twin constants from the pinned FSOT authority (AEB2AD).

Fixes T-3 of FSOT-2.1-Cpp docs/TRIT_SPEC.md: C_EFF, K and the collapse threshold
Theta = C_EFF * P_VAR (plus B_IN, C_FACTOR, CHAOS, which inherit the same stale
input) were typed in at their pre-pi-identity values (0.01 in place of pi^-4).
Numbers here are never typed by hand: they are read from vendor/fsot_compute.py
of FSOT-2.1-Lean, whose SHA-256 must start with AEB2AD.

Usage
  python tools/gen_seeds_from_authority.py --authority /path/to/FSOT-2.1-Lean/vendor/fsot_compute.py
      verify the authority SHA-256, write config/fsot_authority_AEB2AD.json, rewrite every site.
  python tools/gen_seeds_from_authority.py --check
      no authority needed: fail (exit 1) if any site differs from config/fsot_authority_AEB2AD.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONST_JSON = ROOT / "config" / "fsot_authority_AEB2AD.json"
EXPECTED_SHA = "AEB2ADAD6E80F487772C5DF90A2E3DDA71624AB831A6A83B94AB471AC9AAC170"
HUB_COMMIT = "d127c07e518ae349f500d68d9e32b6236349c8f9"
AUTHORITY_NAMES = ["C_EFF", "P_VAR", "K", "B_IN", "C_FACTOR", "CHAOS"]

# (relative path, kind, symbol in file, authority key)
# kind: py = `name[: float] = lit`, zig = `const NAME: f64 = lit;`, rs = `const NAME: f64 = lit;` (underscores),
#       cuda = `#define NAME litf`, json = dotted key path.
SITES = [
    ("fsot_lib/seeds.py", "py", "c_eff", "C_EFF"),
    ("fsot_lib/seeds.py", "py", "p_var", "P_VAR"),
    ("fsot_lib/seeds.py", "py", "b_in", "B_IN"),
    ("fsot_lib/seeds.py", "py", "c_factor", "C_FACTOR"),
    ("fsot_lib/seeds.py", "py", "k", "K"),
    ("fsot_lib/seeds.py", "py", "chaos", "CHAOS"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "C_EFF", "C_EFF"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "P_VAR", "P_VAR"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "K_COUPLING", "K"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "CHAOS", "CHAOS"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "B_IN", "B_IN"),
    ("phase2_native_gpu/python/fsot_gpu_engine.py", "py", "C_FACTOR", "C_FACTOR"),
    ("phase2_native_gpu/cuda/fsot_attn_lib.cu", "cuda", "COLLAPSE_THRESHOLD", "THETA_F64"),
    ("phase2_native_gpu/cuda/fsot_beat_cuda.cu", "cuda", "COLLAPSE_THRESHOLD", "THETA_F64"),
    ("phase2_native_gpu/cuda/fsot_consensus_sparse.cu", "cuda", "COLLAPSE_THRESHOLD", "THETA_F64"),
    ("parity/zig_parity/parity.zig", "zig", "C_EFF", "C_EFF"),
    ("parity/zig_parity/parity.zig", "zig", "P_VAR", "P_VAR"),
    ("parity/zig_parity/parity.zig", "zig", "K", "K"),
    ("parity/rust_parity/src/main.rs", "rs", "C_EFF", "C_EFF"),
    ("parity/rust_parity/src/main.rs", "rs", "P_VAR", "P_VAR"),
    ("parity/rust_parity/src/main.rs", "rs", "K", "K"),
    ("parity/golden.json", "json", "collapse_threshold", "THETA_F64"),
    ("parity/golden.json", "json", "seeds.k", "K"),
    ("parity/golden.json", "json", "seeds.c_eff", "C_EFF"),
    ("parity/golden.json", "json", "seeds.p_var", "P_VAR"),
]
NUM = r"(-?[0-9][0-9_]*\.?[0-9_]*(?:[eE][-+]?[0-9]+)?)"


def _pattern(kind: str, sym: str) -> re.Pattern[str]:
    if kind == "py":
        return re.compile(rf"^(\s*{sym}\s*(?::\s*float)?\s*=\s*){NUM}", re.M)
    if kind in ("zig", "rs"):
        return re.compile(rf"^(\s*(?:pub\s+)?const\s+{sym}\s*:\s*f64\s*=\s*){NUM}", re.M)
    if kind == "cuda":
        return re.compile(rf"^(\s*#define\s+{sym}\s+){NUM}(?=f\b)", re.M)
    raise ValueError(kind)


def _fmt(kind: str, v: float) -> str:
    return repr(float(v))


def load_authority(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest().upper()
    if sha != EXPECTED_SHA:
        sys.exit(f"authority SHA-256 {sha} != expected {EXPECTED_SHA} (AEB2AD)")
    spec = importlib.util.spec_from_file_location("fsot_compute_authority", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    vals = {n: repr(float(getattr(mod, n))) for n in AUTHORITY_NAMES}
    vals["THETA_F64"] = repr(float(vals["C_EFF"]) * float(vals["P_VAR"]))
    vals["THETA_MP"] = repr(float(mod.C_EFF * mod.P_VAR))
    return vals


def _get_json(d: dict, dotted: str):
    for p in dotted.split("."):
        d = d[p]
    return d


def _set_json(d: dict, dotted: str, v) -> None:
    ps = dotted.split(".")
    for p in ps[:-1]:
        d = d[p]
    d[ps[-1]] = v


def apply_sites(vals: dict[str, str], write: bool) -> list[str]:
    bad: list[str] = []
    by_file: dict[str, list] = {}
    for rel, kind, sym, key in SITES:
        by_file.setdefault(rel, []).append((kind, sym, key))
    for rel, items in by_file.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        if items[0][0] == "json":
            # Edit number tokens in place so the file keeps its layout.
            data = json.loads(text)
            new = text
            for _, sym, key in items:
                want = float(vals[key])
                have = float(_get_json(data, sym))
                if have != want:
                    bad.append(f"{rel}:{sym} = {have!r} (authority {want!r})")
                    leaf = sym.split(".")[-1]
                    pat = re.compile(rf'("{leaf}"\s*:\s*){NUM}')
                    ms = [m for m in pat.finditer(new) if float(m.group(2)) == have]
                    if len(ms) != 1:
                        bad.append(f"{rel}:{sym}: cannot locate a unique token")
                        continue
                    m = ms[0]
                    new = new[: m.start(2)] + repr(want) + new[m.end(2):]
            if write and new != text:
                p.write_text(new, encoding="utf-8")
            continue
        new = text
        for kind, sym, key in items:
            pat = _pattern(kind, sym)
            ms = list(pat.finditer(new))
            if len(ms) != 1:
                bad.append(f"{rel}:{sym}: expected exactly 1 literal site, found {len(ms)}")
                continue
            m = ms[0]
            have = float(m.group(2).replace("_", ""))
            want = float(vals[key])
            if have != want:
                line = new.count("\n", 0, m.start()) + 1
                bad.append(f"{rel}:{line} {sym} = {have!r} (authority {want!r})")
                new = new[: m.start(2)] + _fmt(kind, want) + new[m.end(2):]
        if write and new != text:
            p.write_text(new, encoding="utf-8")
    return bad


def golden_scalar() -> float:
    """Re-derive the golden scalar with the (regenerated) fsot_lib in a fresh process."""
    code = (
        "import sys; sys.path.insert(0, %r); from fsot_lib import compute_scalar;"
        "print(repr(float(compute_scalar(D_eff=8.0, observed=True, delta_psi=0.7))))" % str(ROOT)
    )
    return float(subprocess.check_output([sys.executable, "-c", code], text=True).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.authority:
        vals = load_authority(a.authority)
        CONST_JSON.write_text(
            json.dumps(
                {
                    "description": "Seed-twin constants generated from the FSOT authority; do not edit by hand.",
                    "authority": "FSOT-2.1-Lean vendor/fsot_compute.py",
                    "authority_sha256": EXPECTED_SHA,
                    "pin_prefix": "AEB2AD",
                    "hub_commit": HUB_COMMIT,
                    "raw_url": f"https://raw.githubusercontent.com/dappalumbo91/FSOT-2.1-Lean/{HUB_COMMIT}/vendor/fsot_compute.py",
                    "generator": "tools/gen_seeds_from_authority.py",
                    "theta_note": "THETA_F64 = float(C_EFF)*float(P_VAR), the value every f64 twin computes; THETA_MP is the mpmath product rounded once.",
                    "constants": vals,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        changed = apply_sites(vals, write=True)
        for c in changed:
            print("updated", c)
        g = ROOT / "parity" / "golden.json"
        gd = json.loads(g.read_text(encoding="utf-8"))
        s = golden_scalar()
        if gd.get("scalar_observed_D8_dp0_7") != s:
            print(f"updated parity/golden.json scalar_observed_D8_dp0_7 {gd.get('scalar_observed_D8_dp0_7')!r} -> {s!r}")
            txt = g.read_text(encoding="utf-8")
            old = repr(gd["scalar_observed_D8_dp0_7"])
            assert txt.count(old) == 1
            g.write_text(txt.replace(old, repr(s)), encoding="utf-8")
        return 0
    if a.check:
        vals = json.loads(CONST_JSON.read_text(encoding="utf-8"))["constants"]
        bad = apply_sites(vals, write=False)
        g = json.loads((ROOT / "parity" / "golden.json").read_text(encoding="utf-8"))
        if g.get("scalar_observed_D8_dp0_7") != golden_scalar():
            bad.append("parity/golden.json scalar_observed_D8_dp0_7 does not match fsot_lib")
        for b in bad:
            print("MISMATCH", b)
        print("seed constants: OK (AEB2AD)" if not bad else f"seed constants: {len(bad)} mismatch(es)")
        return 1 if bad else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
