/-
  FSOT Formal-GPU — Abstract VRAM / crystal memory model.

  Captures the intent of the Zig VRAM allocator (RTX 5070 ~12.8 GB boundary)
  without claiming a full CUDA driver formalization.
-/

import Trinary

namespace FSOT.GPU

/-- One FSOT voxel tensor as carried in fsot_vram_payload.json / Zig. -/
structure FsotVoxel where
  voxelId   : Nat
  x         : Float
  y         : Float
  z         : Float
  trinary   : Trinary
  resonance : Float
  deriving Repr

/-- Logical sectors of device memory (crystal layout). -/
inductive Sector
  | header
  | bootConstants
  | trinaryBanks
  | phiWorkspace
  | ltmWindows
  | interopScratch
  deriving DecidableEq, Repr

/-- Abstract device with a fixed VRAM capacity in bytes. -/
structure Device where
  name           : String
  vramBytes      : Nat
  computeMajor   : Nat
  computeMinor   : Nat
  deriving Repr

/-- RTX 5070 target from experiment config / Zig allocator. -/
def rtx5070 : Device where
  name         := "NVIDIA GeForce RTX 5070"
  vramBytes    := 12800 * 1024 * 1024  -- 12.8 GiB target boundary
  computeMajor := 12
  computeMinor := 0

/-- Allocation request must not exceed capacity (safety contract). -/
def fits (d : Device) (bytes : Nat) : Prop := bytes ≤ d.vramBytes

theorem zero_fits (d : Device) : fits d 0 := by
  simp [fits]

/-- Ownership: each exclusive sector has at most one writer (spec-level). -/
structure Ownership where
  sector : Sector
  ownerId : Nat
  exclusive : Bool
  deriving Repr

/-- Kernel contract stub: a launch may only write owned exclusive sectors. -/
structure KernelContract where
  name : String
  writes : List Sector
  reads  : List Sector
  deriving Repr

def phiTileKernel : KernelContract where
  name   := "fsot_scalar_tile"
  writes := [.phiWorkspace]
  reads  := [.bootConstants, .trinaryBanks]
  
def trinaryPackKernel : KernelContract where
  name   := "trinary_pack"
  writes := [.trinaryBanks]
  reads  := [.phiWorkspace]

end FSOT.GPU
