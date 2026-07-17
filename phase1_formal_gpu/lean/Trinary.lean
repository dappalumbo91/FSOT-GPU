/-
  FSOT Formal-GPU — Trinary state model for GPU packing.

  States mirror SR-ITE / Zig VRAM allocator:
    Spin_Up (+1), Superposed (0), Spin_Down (-1)

  Packing target: 2 bits per state → 32 states per UInt64 (warp-friendly).
  This module is the specification authority; CUDA implements it in Phase 2.
-/

namespace FSOT.GPU

/-- Discrete trinary fluid spin used in FSOT crystal tensors. -/
inductive Trinary
  | spinDown   -- -1
  | superposed --  0
  | spinUp     -- +1
  deriving DecidableEq, Repr

/-- Map to signed integer used by Zig payload / Python. -/
def Trinary.toInt : Trinary → Int
  | .spinDown   => -1
  | .superposed => 0
  | .spinUp     => 1

/-- Encode to 2-bit payload (0,1,2) for bitfield packing. -/
def Trinary.toBits : Trinary → Nat
  | .spinDown   => 0
  | .superposed => 1
  | .spinUp     => 2

/-- Decode 2-bit payload; 3 is reserved/invalid. -/
def Trinary.ofBits? : Nat → Option Trinary
  | 0 => some .spinDown
  | 1 => some .superposed
  | 2 => some .spinUp
  | _ => none

theorem Trinary.roundtrip (t : Trinary) :
    Trinary.ofBits? t.toBits = some t := by
  cases t <;> rfl

/-- Valid packed codes are only {0,1,2}. -/
def validCode (n : Nat) : Prop := n ≤ 2

theorem toBits_valid (t : Trinary) : validCode t.toBits := by
  cases t <;> simp [validCode, Trinary.toBits]

/-- How many trinary states fit in one 64-bit word at 2 bits each. -/
def statesPerU64 : Nat := 32

theorem statesPerU64_eq : statesPerU64 = 32 := rfl

/-- Abstract: packing 32 codes into UInt64 is injective when each code is valid.
    Concrete bit-twiddling is proven / mirrored in CUDA Phase 2. -/
structure PackedTrinary32 where
  raw : UInt64
  deriving Repr

end FSOT.GPU
