module FSOTGpuBoot

/// Phase 1 — F* boot scalar oracle for GPU experiment.
/// Mirrors archive verification/fstar/FSOTScalarKernel.fst canonical value.

open FStar.Real

let boot_scalar_canonical : real = 0.09928895626861721R

/// GPU tests must accept results within this relative band for f32 paths.
let gpu_f32_rel_tol : real = 0.00001R

/// Abstract trinary codes for packing contracts.
type trinary =
  | SpinDown
  | Superposed
  | SpinUp

let trinary_to_bits (t: trinary) : nat =
  match t with
  | SpinDown -> 0
  | Superposed -> 1
  | SpinUp -> 2

let trinary_of_bits (n: nat) : option trinary =
  if n = 0 then Some SpinDown
  else if n = 1 then Some Superposed
  else if n = 2 then Some SpinUp
  else None

val trinary_roundtrip: t:trinary ->
  Lemma (trinary_of_bits (trinary_to_bits t) == Some t)
let trinary_roundtrip t =
  match t with
  | SpinDown -> ()
  | Superposed -> ()
  | SpinUp -> ()

/// VRAM capacity contract for RTX 5070 experiment target (bytes).
let rtx5070_vram_target : nat = 12800 * 1024 * 1024

val alloc_fits: used:nat -> req:nat ->
  Lemma (requires used + req <= rtx5070_vram_target)
        (ensures  used + req <= rtx5070_vram_target)
let alloc_fits used req = ()
