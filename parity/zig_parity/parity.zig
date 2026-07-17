//! FSOT Formal-GPU parity checker (Zig 0.15+).
//! Pack + collapse threshold must match Python/Rust/formal golden.

const std = @import("std");

const C_EFF: f64 = 0.9577022026205613;
const P_VAR: f64 = 0.9579871226722757;
const PHI: f64 = 1.618033988749895;
const GAMMA: f64 = 0.5772156649015329;
const K: f64 = 0.42022166416069665;
const PSI_CON: f64 = 0.6321205588285577;
const COLLAPSE_THRESHOLD: f64 = C_EFF * P_VAR;

fn packU64(codes: *const [32]u8) u64 {
    var w: u64 = 0;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        w |= @as(u64, codes[i] & 0x3) << @intCast(2 * i);
    }
    return w;
}

fn unpackU64(w: u64, out: *[32]u8) void {
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        out[i] = @truncate((w >> @intCast(2 * i)) & 0x3);
    }
}

pub fn main() !void {
    var codes: [32]u8 = undefined;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        codes[i] = @intCast(i % 3);
    }
    const word = packU64(&codes);
    var back: [32]u8 = undefined;
    unpackU64(word, &back);
    var pack_ok = true;
    i = 0;
    while (i < 32) : (i += 1) {
        if (back[i] != codes[i]) pack_ok = false;
    }

    // Zig 0.15: buffered File.stdout writer
    var buf: [512]u8 = undefined;
    var aw = std.fs.File.stdout().writer(&buf);
    const w = &aw.interface;
    try w.print(
        "{{\"lang\":\"zig\",\"collapse_threshold\":{d:.17},\"phi\":{d:.17},\"gamma\":{d:.17},\"k\":{d:.17},\"c_eff\":{d:.17},\"p_var\":{d:.17},\"psi_con\":{d:.17},\"pack_u64_word\":{d},\"pack_ok\":{}}}\n",
        .{
            COLLAPSE_THRESHOLD,
            PHI,
            GAMMA,
            K,
            C_EFF,
            P_VAR,
            PSI_CON,
            word,
            pack_ok,
        },
    );
    try w.flush();
}
