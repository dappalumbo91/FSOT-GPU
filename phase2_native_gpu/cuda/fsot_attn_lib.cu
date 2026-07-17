// FSOT consensus CUDA library — adaptive fast path (sm_120)
// Layout: q,k,v,out [B,H,S,D] float32 device pointers
//
// Authority: collapse θ = C_eff·P_var, coh gate > 0.5, NO exp — consensus not softmax.
// Work O(B·H·S·A·D) with A ≪ S after coherence gate.
//
// Build: nvcc -O3 -arch=sm_120 -shared -o fsot_attn_lib.dll fsot_attn_lib.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define COLLAPSE_THRESHOLD 0.9174663774653723f
#define COH_GATE 0.5f
#define FSOT_MAX_A_SHARED 128

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

// ---------------------------------------------------------------------------
// Light fused: 1 block/(B,H), actives in shared — best when launch overhead
// dominates (short S).
// ---------------------------------------------------------------------------
__global__ void consensus_light_fused(const float *__restrict__ q,
                                      const float *__restrict__ k,
                                      const float *__restrict__ v,
                                      float *__restrict__ out, int B, int H,
                                      int S, int D) {
  const int bh = blockIdx.x;
  if (bh >= B * H)
    return;
  const int base = bh * S;

  extern __shared__ int s_act[];
  __shared__ int sA;
  __shared__ int sOverflow;

  if (threadIdx.x == 0) {
    sA = 0;
    sOverflow = 0;
  }
  __syncthreads();

  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float *row = k + ((size_t)base + j) * (size_t)D;
    int sharp = 0;
#pragma unroll 4
    for (int d = 0; d < D; ++d)
      if (fabsf(__ldg(row + d)) > COLLAPSE_THRESHOLD)
        sharp++;
    if ((float)sharp / (float)D > COH_GATE) {
      int slot = atomicAdd(&sA, 1);
      if (slot < FSOT_MAX_A_SHARED)
        s_act[slot] = j;
      else
        atomicExch(&sOverflow, 1);
    }
  }
  __syncthreads();

  const int A = sOverflow ? 0 : sA;
  for (int qi = threadIdx.x; qi < S; qi += blockDim.x) {
    const float *qh = q + ((size_t)base + qi) * (size_t)D;
    float *oh = out + ((size_t)base + qi) * (size_t)D;
    for (int d = 0; d < D; ++d)
      oh[d] = 0.f;
    float active = 0.f;

    if (!sOverflow) {
      for (int a = 0; a < A; ++a) {
        int kj = s_act[a];
        if (kj > qi)
          continue;
        const float *kh = k + ((size_t)base + kj) * (size_t)D;
        const float *vh = v + ((size_t)base + kj) * (size_t)D;
        float acc = 0.f;
#pragma unroll 4
        for (int d = 0; d < D; ++d) {
          int tq = collapse_code(__ldg(qh + d));
          int tk = collapse_code(__ldg(kh + d));
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
          oh[d] += w * __ldg(vh + d);
      }
    } else {
      for (int kj = 0; kj <= qi; ++kj) {
        const float *kh = k + ((size_t)base + kj) * (size_t)D;
        int sharp = 0;
        for (int d = 0; d < D; ++d)
          if (fabsf(__ldg(kh + d)) > COLLAPSE_THRESHOLD)
            sharp++;
        if ((float)sharp / (float)D <= COH_GATE)
          continue;
        float acc = 0.f;
        for (int d = 0; d < D; ++d) {
          int tq = collapse_code(__ldg(qh + d));
          int tk = collapse_code(__ldg(kh + d));
          if (tq == 1 || tk == 1)
            continue;
          acc += (tq == tk) ? 1.f : -1.f;
        }
        float w = acc / (float)D;
        if (w == 0.f)
          continue;
        active += 1.f;
        const float *vh = v + ((size_t)base + kj) * (size_t)D;
        for (int d = 0; d < D; ++d)
          oh[d] += w * __ldg(vh + d);
      }
    }
    if (active > 1.f) {
      float inv = 1.f / active;
      for (int d = 0; d < D; ++d)
        oh[d] *= inv;
    }
  }
}

// ---------------------------------------------------------------------------
// Multi-pass: coh+compact fused into one kernel, then consensus.
// Best for long S (beats fused SDPA at S>=2048 on RTX 5070).
// ---------------------------------------------------------------------------
__global__ void coh_compact_kernel(const float *__restrict__ k,
                                   int *__restrict__ act_idx,
                                   int *__restrict__ act_n, int B, int H, int S,
                                   int D) {
  // one block per (B,H)
  const int bh = blockIdx.x;
  if (bh >= B * H)
    return;
  const int base = bh * S;
  extern __shared__ int s_act[];
  __shared__ int sA;

  if (threadIdx.x == 0)
    sA = 0;
  __syncthreads();

  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float *row = k + ((size_t)base + j) * (size_t)D;
    int sharp = 0;
#pragma unroll 4
    for (int d = 0; d < D; ++d)
      if (fabsf(__ldg(row + d)) > COLLAPSE_THRESHOLD)
        sharp++;
    if ((float)sharp / (float)D > COH_GATE) {
      int slot = atomicAdd(&sA, 1);
      // write to global act_idx directly if large; use shared staging when small
      if (slot < S)
        act_idx[(size_t)bh * S + slot] = j;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    act_n[bh] = sA;
}

__global__ void consensus_kernel(const float *__restrict__ q,
                                 const float *__restrict__ k,
                                 const float *__restrict__ v,
                                 const int *__restrict__ act_idx,
                                 const int *__restrict__ act_n,
                                 float *__restrict__ out, int B, int H, int S,
                                 int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int n = B * H * S;
  if (t >= n)
    return;
  int bh = t / S;
  int qi = t % S;
  int A = act_n[bh];
  const float *qh = q + ((size_t)bh * S + qi) * (size_t)D;
  const float *kbase = k + (size_t)bh * S * (size_t)D;
  const float *vbase = v + (size_t)bh * S * (size_t)D;
  const int *act = act_idx + (size_t)bh * (size_t)S;
  float *oh = out + ((size_t)bh * S + qi) * (size_t)D;

  for (int d = 0; d < D; ++d)
    oh[d] = 0.f;
  float active = 0.f;
  for (int a = 0; a < A; ++a) {
    int kj = act[a];
    if (kj > qi)
      continue;
    const float *kh = kbase + (size_t)kj * (size_t)D;
    const float *vh = vbase + (size_t)kj * (size_t)D;
    float acc = 0.f;
#pragma unroll 4
    for (int d = 0; d < D; ++d) {
      int tq = collapse_code(__ldg(qh + d));
      int tk = collapse_code(__ldg(kh + d));
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
      oh[d] += w * __ldg(vh + d);
  }
  if (active > 1.f) {
    float inv = 1.f / active;
    for (int d = 0; d < D; ++d)
      oh[d] *= inv;
  }
}

static int *g_act = nullptr;
static int *g_an = nullptr;
static size_t g_ns_cap = 0;
static size_t g_bh_cap = 0;
// -1 adaptive, 0 multipass, 1 light fused
static int g_mode = -1;

static int ensure_workspace(int B, int H, int S) {
  size_t ns = (size_t)B * H * S;
  size_t bh = (size_t)B * H;
  if (ns <= g_ns_cap && bh <= g_bh_cap)
    return 0;
  if (g_act)
    cudaFree(g_act);
  if (g_an)
    cudaFree(g_an);
  g_act = nullptr;
  g_an = nullptr;
  size_t ns2 = ns * 2 + 4096;
  size_t bh2 = bh * 2 + 64;
  if (cudaMalloc(&g_act, ns2 * sizeof(int)) != cudaSuccess)
    return -2;
  if (cudaMalloc(&g_an, bh2 * sizeof(int)) != cudaSuccess)
    return -3;
  g_ns_cap = ns2;
  g_bh_cap = bh2;
  return 0;
}

static int launch_multipass(float *q, float *k, float *v, float *out, int B,
                            int H, int S, int D) {
  if (ensure_workspace(B, H, S) != 0)
    return -1;
  size_t ns = (size_t)B * H * S;
  int thr = 256;
  int blk = (int)((ns + thr - 1) / thr);
  int cthr = (S >= 256) ? 256 : 128;
  coh_compact_kernel<<<B * H, cthr>>>(k, g_act, g_an, B, H, S, D);
  consensus_kernel<<<blk, thr>>>(q, k, v, g_act, g_an, out, B, H, S, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -10;
}

static int launch_light_fused(float *q, float *k, float *v, float *out, int B,
                              int H, int S, int D) {
  int thr = (S >= 256) ? 256 : 128;
  size_t shmem = (size_t)FSOT_MAX_A_SHARED * sizeof(int);
  consensus_light_fused<<<B * H, thr, shmem>>>(q, k, v, out, B, H, S, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -11;
}

EXPORT int fsot_consensus_cuda_device(float *q, float *k, float *v, float *out,
                                      int B, int H, int S, int D) {
  if (D <= 0 || S <= 0 || H <= 0 || B <= 0)
    return -2;

  int use_fused = g_mode;
  if (use_fused < 0) {
    // Short: light fused (1 launch). Long: 2-pass multipass (scales past SDPA).
    use_fused = (S <= 96) ? 1 : 0;
  }

  if (use_fused)
    return launch_light_fused(q, k, v, out, B, H, S, D);
  return launch_multipass(q, k, v, out, B, H, S, D);
}

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
  cudaDeviceSynchronize();
  cudaMemcpy(out_h, out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(out);
  return rc;
}

EXPORT void fsot_workspace_reset(void) {
  if (g_act)
    cudaFree(g_act);
  if (g_an)
    cudaFree(g_an);
  g_act = nullptr;
  g_an = nullptr;
  g_ns_cap = g_bh_cap = 0;
}

// mode: -1 adaptive, 0 multipass, 1 light fused
EXPORT void fsot_set_fused(int mode) { g_mode = mode; }

EXPORT float fsot_consensus_cuda_device_bench(float *q, float *k, float *v,
                                              float *out, int B, int H, int S,
                                              int D, int iters) {
  for (int i = 0; i < 8; ++i) {
    if (fsot_consensus_cuda_device(q, k, v, out, B, H, S, D) != 0)
      return -1.f;
  }
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i)
    fsot_consensus_cuda_device(q, k, v, out, B, H, S, D);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / (float)iters;
}
