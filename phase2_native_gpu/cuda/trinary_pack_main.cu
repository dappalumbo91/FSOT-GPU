// FSOT Formal-GPU — trinary pack/unpack roundtrip (nvcc)
// Spec: phase1 Lean Trinary / F* FSOTGpuBoot / kernel trinary.rs
// Build (RTX 5070 / Blackwell):
//   nvcc -O3 -arch=sm_120 -o trinary_pack_test.exe trinary_pack_main.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__device__ __forceinline__ uint64_t pack32(const uint8_t *codes) {
  uint64_t w = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    w |= (uint64_t)(codes[i] & 0x3u) << (2 * i);
  return w;
}

__device__ __forceinline__ void unpack32(uint64_t w, uint8_t *codes) {
#pragma unroll
  for (int i = 0; i < 32; ++i)
    codes[i] = (uint8_t)((w >> (2 * i)) & 0x3u);
}

__global__ void pack_kernel(const uint8_t *__restrict__ in,
                            uint64_t *__restrict__ out, size_t n_groups) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (g >= n_groups)
    return;
  out[g] = pack32(in + g * 32);
}

__global__ void unpack_kernel(const uint64_t *__restrict__ in,
                              uint8_t *__restrict__ out, size_t n_groups) {
  size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (g >= n_groups)
    return;
  unpack32(in[g], out + g * 32);
}

int main() {
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {
    fprintf(stderr, "No CUDA devices\n");
    return 1;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("FSOT CUDA trinary pack\n");
  printf("  device: %s  CC %d.%d\n", prop.name, prop.major, prop.minor);

  const size_t groups = 65536; // 2M trits
  uint8_t *h_in = (uint8_t *)malloc(groups * 32);
  if (!h_in)
    return 1;
  for (size_t i = 0; i < groups * 32; ++i)
    h_in[i] = (uint8_t)(i % 3);

  uint8_t *d_in = nullptr, *d_out = nullptr;
  uint64_t *d_packed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, groups * 32));
  CUDA_CHECK(cudaMalloc(&d_packed, groups * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_out, groups * 32));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, groups * 32, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (int)((groups + threads - 1) / threads);

  pack_kernel<<<blocks, threads>>>(d_in, d_packed, groups);
  unpack_kernel<<<blocks, threads>>>(d_packed, d_out, groups);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < 50; ++i) {
    pack_kernel<<<blocks, threads>>>(d_in, d_packed, groups);
    unpack_kernel<<<blocks, threads>>>(d_packed, d_out, groups);
  }
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  uint8_t *h_out = (uint8_t *)malloc(groups * 32);
  if (!h_out)
    return 1;
  CUDA_CHECK(cudaMemcpy(h_out, d_out, groups * 32, cudaMemcpyDeviceToHost));

  size_t mismatches = 0;
  for (size_t i = 0; i < groups * 32; ++i)
    if (h_in[i] != h_out[i])
      ++mismatches;

  double trits = (double)groups * 32.0 * 50.0;
  double trit_per_s = (ms > 0.0f) ? (trits / (ms / 1000.0)) : 0.0;
  printf("  groups=%zu trits=%zu mismatches=%zu\n", groups, groups * 32,
         mismatches);
  printf("  50 iters: %.3f ms  => %.3f Mtrits/s pack+unpack\n", ms,
         trit_per_s / 1e6);
  printf("  ok=%s\n", mismatches == 0 ? "true" : "false");

  cudaFree(d_in);
  cudaFree(d_packed);
  cudaFree(d_out);
  free(h_in);
  free(h_out);
  return mismatches == 0 ? 0 : 1;
}
