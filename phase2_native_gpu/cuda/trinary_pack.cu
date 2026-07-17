// FSOT Formal-GPU Phase 2 skeleton — trinary pack/unpack
// Spec authority: phase1_formal_gpu/lean/Trinary.lean
// Codes: SpinDown=0, Superposed=1, SpinUp=2  (2 bits each, 32 per uint64_t)

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__ uint64_t pack32(const uint8_t* codes /* 32 values in {0,1,2} */) {
    uint64_t w = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        w |= (uint64_t)(codes[i] & 0x3u) << (2 * i);
    }
    return w;
}

__device__ __forceinline__ void unpack32(uint64_t w, uint8_t* codes) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        codes[i] = (uint8_t)((w >> (2 * i)) & 0x3u);
    }
}

__global__ void pack_kernel(const uint8_t* __restrict__ in,
                            uint64_t* __restrict__ out,
                            size_t n_groups) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    out[g] = pack32(in + g * 32);
}

__global__ void unpack_kernel(const uint64_t* __restrict__ in,
                              uint8_t* __restrict__ out,
                              size_t n_groups) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    unpack32(in[g], out + g * 32);
}

// Host smoke: compile with
//   nvcc -O3 -o trinary_pack_test trinary_pack.cu
// Full Python bridge lands in phase2_native_gpu/python/

#ifdef TRINARY_PACK_MAIN
int main() {
    const size_t groups = 1024;
    uint8_t* h_in = new uint8_t[groups * 32];
    for (size_t i = 0; i < groups * 32; ++i) h_in[i] = (uint8_t)(i % 3);

    uint8_t *d_in, *d_out;
    uint64_t *d_packed;
    cudaMalloc(&d_in, groups * 32);
    cudaMalloc(&d_packed, groups * sizeof(uint64_t));
    cudaMalloc(&d_out, groups * 32);
    cudaMemcpy(d_in, h_in, groups * 32, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((groups + threads - 1) / threads);
    pack_kernel<<<blocks, threads>>>(d_in, d_packed, groups);
    unpack_kernel<<<blocks, threads>>>(d_packed, d_out, groups);
    cudaDeviceSynchronize();

    uint8_t* h_out = new uint8_t[groups * 32];
    cudaMemcpy(h_out, d_out, groups * 32, cudaMemcpyDeviceToHost);

    size_t mismatches = 0;
    for (size_t i = 0; i < groups * 32; ++i) {
        if (h_in[i] != h_out[i]) ++mismatches;
    }
    printf("FSOT trinary pack roundtrip mismatches: %zu / %zu\n", mismatches, groups * 32);

    cudaFree(d_in); cudaFree(d_packed); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return mismatches == 0 ? 0 : 1;
}
#endif
