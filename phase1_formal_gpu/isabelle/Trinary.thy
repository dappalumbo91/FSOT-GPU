theory Trinary
  imports Main
begin

datatype trinary = SpinDown | Superposed | SpinUp

fun trinary_to_bits :: "trinary ⇒ nat" where
  "trinary_to_bits SpinDown = 0" |
  "trinary_to_bits Superposed = 1" |
  "trinary_to_bits SpinUp = 2"

fun trinary_of_bits :: "nat ⇒ trinary option" where
  "trinary_of_bits 0 = Some SpinDown" |
  "trinary_of_bits 1 = Some Superposed" |
  "trinary_of_bits 2 = Some SpinUp" |
  "trinary_of_bits _ = None"

lemma trinary_roundtrip: "trinary_of_bits (trinary_to_bits t) = Some t"
  by (cases t) simp_all

definition states_per_u64 :: nat where
  "states_per_u64 = 32"

lemma warp_divides_states:
  "states_per_u64 mod 32 = 0"
  by (simp add: states_per_u64_def)

end
