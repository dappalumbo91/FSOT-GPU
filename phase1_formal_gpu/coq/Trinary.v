(* FSOT Formal-GPU — Coq trinary packing (parallel to Lean Trinary.lean) *)

Inductive Trinary : Type :=
  | SpinDown
  | Superposed
  | SpinUp.

Definition trinary_to_bits (t : Trinary) : nat :=
  match t with
  | SpinDown => 0
  | Superposed => 1
  | SpinUp => 2
  end.

Definition trinary_of_bits (n : nat) : option Trinary :=
  match n with
  | 0 => Some SpinDown
  | 1 => Some Superposed
  | 2 => Some SpinUp
  | _ => None
  end.

Lemma trinary_roundtrip : forall t, trinary_of_bits (trinary_to_bits t) = Some t.
Proof.
  intros t; destruct t; reflexivity.
Qed.

Definition states_per_u64 : nat := 32.
