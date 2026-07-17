// FSOT consensus CUDA library — persistent workspace (no malloc per call)
// Layout: q,k,v,out [B,H,S,D] float32 device pointers
// Build: nvcc -O3 -arch=sm_120 -shared -o fsot_attn_lib.dll fsot_attn_lib.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define COLLAPSE_THRESHOLD 0.9174663774653723f
#define COH_GATE 0.5f

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

__device__ __forceinline__ int collapse_code(float x) {
  if (x > COLLAPSE_THRESHOLD)
    return 2;
  if (x < -COLLAPSE_THRESHOLD)
    return 0;
  return 1;
}

__global__ void k_coh_kernel(const float *k, float *coh, int B, int H, int S,
                             int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int n = B * H * S;
  if (t >= n)
    return;
  const float *row = k + (size_t)t * D;
  int sharp = 0;
#pragma unroll 4
  for (int d = 0; d < D; ++d)
    if (fabsf(row[d]) > COLLAPSE_THRESHOLD)
      sharp++;
  coh[t] = (float)sharp / (float)D;
}

__global__ void compact_kernel(const float *coh, int *act_idx, int *act_n,
                               int B, int H, int S) {
  int bh = blockIdx.x;
  if (bh >= B * H)
    return;
  if (threadIdx.x != 0)
    return;
  const float *ch = coh + (size_t)bh * S;
  int *out = act_idx + (size_t)bh * S;
  int n = 0;
  for (int j = 0; j < S; ++j)
    if (ch[j] > COH_GATE)
      out[n++] = j;
  act_n[bh] = n;
}

__global__ void consensus_kernel(const float *q, const float *k, const float *v,
                                 const int *act_idx, const int *act_n,
                                 float *out, int B, int H, int S, int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int n = B * H * S;
  if (t >= n)
    return;
  int bh = t / S;
  int qi = t % S;
  int A = act_n[bh];
  const float *qh = q + ((size_t)bh * S + qi) * D;
  const float *kbase = k + (size_t)bh * S * D;
  const float *vbase = v + (size_t)bh * S * D;
  const int *act = act_idx + (size_t)bh * S;
  float *oh = out + ((size_t)bh * S + qi) * D;

  for (int d = 0; d < D; ++d)
    oh[d] = 0.f;
  float active = 0.f;
  for (int a = 0; a < A; ++a) {
    int kj = act[a];
    if (kj > qi)
      continue;
    const float *kh = kbase + (size_t)kj * D;
    const float *vh = vbase + (size_t)kj * D;
    float acc = 0.f;
#pragma unroll 4
    for (int d = 0; d < D; ++d) {
      int tq = collapse_code(qh[d]);
      int tk = collapse_code(kh[d]);
      if (tq == 1 || tk == 1)
        continue;
      acc += (tq == tk) ? 1.f : -1.f;
    }
    float w = acc / (float)D;
    if (w == 0.f)
      continue;
    active += 1.f;
#pragma unroll 4
    for (int d = 0; d < D; ++d)
      oh[d] += w * vh[d];
  }
  if (active > 1.f) {
    float inv = 1.f / active;
    for (int d = 0; d < D; ++d)
      oh[d] *= inv;
  }
}

// Persistent workspace
static float *g_coh = nullptr;
static int *g_act = nullptr;
static int *g_an = nullptr;
static size_t g_ns_cap = 0;
static size_t g_bh_cap = 0;

static int ensure_workspace(int B, int H, int S) {
  size_t ns = (size_t)B * H * S;
  size_t bh = (size_t)B * H;
  if (ns <= g_ns_cap && bh <= g_bh_cap)
    return 0;
  if (g_coh)
    cudaFree(g_coh);
  if (g_act)
    cudaFree(g_act);
  if (g_an)
    cudaFree(g_an);
  g_coh = nullptr;
  g_act = nullptr;
  g_an = nullptr;
  // grow with headroom
  size_t ns2 = ns * 2 + 4096;
  size_t bh2 = bh * 2 + 64;
  if (cudaMalloc(&g_coh, ns2 * sizeof(float)) != cudaSuccess)
    return -1;
  if (cudaMalloc(&g_act, ns2 * sizeof(int)) != cudaSuccess)
    return -2;
  if (cudaMalloc(&g_an, bh2 * sizeof(int)) != cudaSuccess)
    return -3;
  g_ns_cap = ns2;
  g_bh_cap = bh2;
  return 0;
}

EXPORT int fsot_consensus_cuda_device(float *q, float *k, float *v, float *out,
                                      int B, int H, int S, int D) {
  if (ensure_workspace(B, H, S) != 0)
    return -1;
  size_t ns = (size_t)B * H * S;
  int thr = 256;
  int blk = (int)((ns + thr - 1) / thr);
  k_coh_kernel<<<blk, thr>>>(k, g_coh, B, H, S, D);
  compact_kernel<<<B * H, 32>>>(g_coh, g_act, g_an, B, H, S);
  consensus_kernel<<<blk, thr>>>(q, k, v, g_act, g_an, out, B, H, S, D);
  // No device-wide sync here — host syncs once per forward/bench for throughput.
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : -10;
}

// Host API (alloc once per call — slower; for tests)
EXPORT int fsot_consensus_cuda(const float *q_h, const float *k_h,
                               const float *v_h, float *out_h, int B, int H,
                               int S, int D) {
  size_t n = (size_t)B * H * S * D;
  float *q, *k, *v, *out;
  if (cudaMalloc(&q, n * sizeof(float)) != cudaSuccess)
    return -1;
  if (cudaMalloc(&k, n * sizeof(float)) != cudaSuccess)
    return -2;
  if (cudaMalloc(&v, n * sizeof(float)) != cudaSuccess)
    return -3;
  if (cudaMalloc(&out, n * sizeof(float)) != cudaSuccess)
    return -4;
  cudaMemcpy(q, q_h, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(k, k_h, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(v, v_h, n * sizeof(float), cudaMemcpyHostToDevice);
  int rc = fsot_consensus_cuda_device(q, k, v, out, B, H, S, D);
  cudaMemcpy(out_h, out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(out);
  return rc;
}

EXPORT void fsot_workspace_reset(void) {
  if (g_coh)
    cudaFree(g_coh);
  if (g_act)
    cudaFree(g_act);
  if (g_an)
    cudaFree(g_an);
  g_coh = nullptr;
  g_act = nullptr;
  g_an = nullptr;
  g_ns_cap = g_bh_cap = 0;
}
