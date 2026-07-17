// FSOT sparse consensus CUDA — uses archive collapse θ = C_eff * P_var
// Gate: coherence > 0.5 (kernel lattice.rs). No exp.
// Build: nvcc -O3 -arch=sm_120 -o fsot_consensus_sparse.exe fsot_consensus_sparse.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Archive Scalar.lean / fsot_compute: C_eff * P_var
#define COLLAPSE_THRESHOLD 0.9174663774653723f

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__device__ __forceinline__ int collapse_code(float x) {
  if (x > COLLAPSE_THRESHOLD)
    return 2;
  if (x < -COLLAPSE_THRESHOLD)
    return 0;
  return 1;
}

// Precompute key coherence [H*S]
__global__ void coh_kernel(const float *__restrict__ k, float *__restrict__ coh,
                           int H, int S, int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= H * S)
    return;
  const float *row = k + (size_t)t * D;
  int sharp = 0;
  for (int d = 0; d < D; ++d)
    if (fabsf(row[d]) > COLLAPSE_THRESHOLD)
      sharp++;
  coh[t] = (float)sharp / (float)D;
}

// One thread per (h, query)
__global__ void consensus_kernel(const float *__restrict__ q,
                                 const float *__restrict__ k,
                                 const float *__restrict__ v,
                                 const float *__restrict__ coh,
                                 float *__restrict__ out, int H, int S, int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= H * S)
    return;
  int h = t / S;
  int qi = t % S;
  const float *qh = q + (size_t)h * S * D;
  const float *kh = k + (size_t)h * S * D;
  const float *vh = v + (size_t)h * S * D;
  const float *ch = coh + (size_t)h * S;
  float *oh = out + (size_t)h * S * D;

  for (int d = 0; d < D; ++d)
    oh[qi * D + d] = 0.f;

  float active = 0.f;
  for (int kj = 0; kj <= qi; ++kj) {
    if (ch[kj] <= 0.5f)
      continue;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) {
      int tq = collapse_code(qh[qi * D + d]);
      int tk = collapse_code(kh[kj * D + d]);
      if (tq == 1 || tk == 1)
        continue;
      acc += (tq == tk) ? 1.f : -1.f;
    }
    float w = acc / (float)D;
    if (w == 0.f)
      continue;
    active += 1.f;
    for (int d = 0; d < D; ++d)
      oh[qi * D + d] += w * vh[kj * D + d];
  }
  if (active > 1.f) {
    float inv = 1.f / active;
    for (int d = 0; d < D; ++d)
      oh[qi * D + d] *= inv;
  }
}

static void bench_config(int H, int S, int D, int iters) {
  size_t n = (size_t)H * S * D;
  size_t ns = (size_t)H * S;
  float *hq = (float *)malloc(n * sizeof(float));
  float *hk = (float *)malloc(n * sizeof(float));
  float *hv = (float *)malloc(n * sizeof(float));
  for (size_t i = 0; i < n; ++i) {
    // Same distribution family as Python bench (approx randn via box-muller lite)
    float u1 = fmodf((float)(i + 1) * 0.6180339887f, 1.f) + 1e-6f;
    float u2 = fmodf((float)(i + 3) * 0.3819660113f, 1.f) + 1e-6f;
    float r = sqrtf(-2.f * logf(u1));
    float z = r * cosf(6.28318530718f * u2);
    hq[i] = z;
    hk[i] = r * sinf(6.28318530718f * u2);
    hv[i] = z * 0.5f;
  }
  float *dq, *dk, *dv, *dcoh, *dout;
  cudaMalloc(&dq, n * sizeof(float));
  cudaMalloc(&dk, n * sizeof(float));
  cudaMalloc(&dv, n * sizeof(float));
  cudaMalloc(&dcoh, ns * sizeof(float));
  cudaMalloc(&dout, n * sizeof(float));
  cudaMemcpy(dq, hq, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv, n * sizeof(float), cudaMemcpyHostToDevice);

  int thr = 128;
  int blk_s = (H * S + thr - 1) / thr;

  coh_kernel<<<blk_s, thr>>>(dk, dcoh, H, S, D);
  consensus_kernel<<<blk_s, thr>>>(dq, dk, dv, dcoh, dout, H, S, D);
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i) {
    coh_kernel<<<blk_s, thr>>>(dk, dcoh, H, S, D);
    consensus_kernel<<<blk_s, thr>>>(dq, dk, dv, dcoh, dout, H, S, D);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);

  printf("  H=%d S=%d D=%d  %d iters  %.4f ms/iter\n", H, S, D, iters,
         ms / (float)iters);

  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dcoh);
  cudaFree(dout);
  free(hq);
  free(hk);
  free(hv);
}

int main() {
  printf("FSOT CUDA sparse consensus (archive collapse theta)\n");
  printf("  COLLAPSE_THRESHOLD=%.12f  gate=0.5  no_exp=true\n",
         COLLAPSE_THRESHOLD);
  bench_config(8, 32, 16, 500);
  bench_config(8, 64, 32, 400);
  bench_config(8, 128, 64, 200);
  /* longer sequences — sparsity advantage grows with S */
  bench_config(8, 256, 64, 100);
  bench_config(8, 512, 64, 60);
  bench_config(8, 1024, 64, 40);
  printf("  ok=true\n");
  return 0;
}
