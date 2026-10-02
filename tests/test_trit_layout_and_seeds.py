"""T-2 / T-3 checks (FSOT-2.1-Cpp docs/TRIT_SPEC.md). Run: python -m pytest tests  (or python tests/test_trit_layout_and_seeds.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsot_lib.trinary import (  # noqa: E402
    canonical_word_to_t1,
    pack_u64,
    t1_word_to_canonical,
    unpack_u64_checked,
)


def test_canonical_vs_t1_vector():
    # trits -1, 0, +1, +1, -1 ; lane 0 = LSB
    t1 = 0b11_01_01_00_11          # Zig packT1: 0->00, +1->01, -1->11
    canon = 0b00_10_10_01_00       # canonical code = t + 1
    assert t1_word_to_canonical(t1, 5) == canon
    assert canonical_word_to_t1(canon, 5) == t1


def test_invalid_codes_rejected():
    for bad, fn in ((0b11, canonical_word_to_t1), (0b10, t1_word_to_canonical)):
        try:
            fn(bad, 1)
        except ValueError:
            continue
        raise AssertionError("invalid lane accepted")
    try:
        unpack_u64_checked(0b11, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("code 3 accepted")


def test_pack_u64_is_canonical():
    codes = [0, 1, 2] * 10 + [0, 1]
    assert unpack_u64_checked(pack_u64(codes)) == codes


def test_seed_constants_match_authority_json():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_seeds_from_authority.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
