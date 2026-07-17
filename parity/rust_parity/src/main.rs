//! FSOT Formal-GPU parity checker (Rust).
//! Mirrors fsot_math/consts + lab golden.json pack contract.

const PHI: f64 = 1.618_033_988_749_895;
const GAMMA: f64 = 0.577_215_664_901_532_9;
const C_EFF: f64 = 0.957_702_202_620_561_3;
const P_VAR: f64 = 0.957_987_122_672_275_7;
const K: f64 = 0.420_221_664_160_696_65;
const PSI_CON: f64 = 0.632_120_558_828_557_7;
const COLLAPSE_THRESHOLD: f64 = C_EFF * P_VAR; // ~0.9174663774653723

fn pack_u64(codes: &[u8; 32]) -> u64 {
    let mut w: u64 = 0;
    for (i, &c) in codes.iter().enumerate() {
        w |= ((c as u64) & 0x3) << (2 * i);
    }
    w
}

fn unpack_u64(w: u64) -> [u8; 32] {
    let mut codes = [0u8; 32];
    for i in 0..32 {
        codes[i] = ((w >> (2 * i)) & 0x3) as u8;
    }
    codes
}

fn main() {
    let mut codes = [0u8; 32];
    for i in 0..32 {
        codes[i] = (i % 3) as u8;
    }
    let word = pack_u64(&codes);
    let back = unpack_u64(word);
    let pack_ok = back == codes;

    // JSON-ish single line for the Python harness
    println!(
        "{{\"lang\":\"rust\",\"collapse_threshold\":{:.17},\"phi\":{:.17},\"gamma\":{:.17},\"k\":{:.17},\"c_eff\":{:.17},\"p_var\":{:.17},\"psi_con\":{:.17},\"pack_u64_word\":{},\"pack_ok\":{}}}",
        COLLAPSE_THRESHOLD,
        PHI,
        GAMMA,
        K,
        C_EFF,
        P_VAR,
        PSI_CON,
        word,
        pack_ok
    );
}
