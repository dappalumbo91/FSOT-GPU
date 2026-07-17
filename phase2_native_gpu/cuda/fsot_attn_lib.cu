// FSOT consensus CUDA library — mid-S + long-S (sm_120)
// Layout: q,k,v,out [B,H,S,D] float32 device pointers
//
// Authority ONLY: collapse θ = C_eff·P_var, coh gate > 0.5, NO exp — consensus.
// Work O(B·H·S·A·D) with A ≪ S after coherence gate.
//
// Mid-S path: pack active key trits once (uint8), consensus reads packs + V.
// Long-S path: multipass coh+compact then consensus.
//
// Build: nvcc -O3 -arch=sm_120 -shared -o fsot_attn_lib.dll fsot_attn_lib.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define COLLAPSE_THRESHOLD 0.9174663774653723f
#define COH_GATE 0.5f
#define FSOT_MAX_A_SHARED 160

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
// Mid-S optimized: one block per (B,H)
// Phase 1: compact actives into shared indices
// Phase 2: pack trits of active K into shared (uint8) + cache V rows
// Phase 3: each query uses packed trit sim (no re-collapse of K)
// ---------------------------------------------------------------------------
__global__ void consensus_mids_kernel(const float *__restrict__ q,
                                      const float *__restrict__ k,
                                      const float *__restrict__ v,
                                      float *__restrict__ out, int B, int H,
                                      int S, int D) {
  const int bh = blockIdx.x;
  if (bh >= B * H)
    return;
  const int base = bh * S;

  // Dynamic shared layout after fixed sA:
  // [int act[MAX_A]] [uchar trit[MAX_A*D] 16-aligned] [float vpack[MAX_A*D]]
  extern __shared__ char smem[];
  int *act = (int *)smem;
  size_t off = (size_t)FSOT_MAX_A_SHARED * sizeof(int);
  unsigned char *trit = (unsigned char *)(smem + off);
  off += (size_t)FSOT_MAX_A_SHARED * (size_t)D;
  off = (off + 15u) & ~((size_t)15u);
  float *vpack = (float *)(smem + off);

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
#pragma unroll 8
    for (int d = 0; d < D; ++d)
      if (fabsf(__ldg(row + d)) > COLLAPSE_THRESHOLD)
        sharp++;
    if ((float)sharp / (float)D > COH_GATE) {
      int slot = atomicAdd(&sA, 1);
      if (slot < FSOT_MAX_A_SHARED)
        act[slot] = j;
      else
        atomicExch(&sOverflow, 1);
    }
  }
  __syncthreads();

  if (sOverflow) {
    // Fall back: serial-gated scan per query (rare under collapse law)
    for (int qi = threadIdx.x; qi < S; qi += blockDim.x) {
      const float *qh = q + ((size_t)base + qi) * (size_t)D;
      float *oh = out + ((size_t)base + qi) * (size_t)D;
      for (int d = 0; d < D; ++d)
        oh[d] = 0.f;
      float active = 0.f;
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
      if (active > 1.f) {
        float inv = 1.f / active;
        for (int d = 0; d < D; ++d)
          oh[d] *= inv;
      }
    }
    return;
  }

  const int A = sA;
  // pack active trits + V
  for (int a = threadIdx.x; a < A; a += blockDim.x) {
    int kj = act[a];
    const float *kh = k + ((size_t)base + kj) * (size_t)D;
    const float *vh = v + ((size_t)base + kj) * (size_t)D;
    unsigned char *tp = trit + (size_t)a * (size_t)D;
#pragma unroll 8
    for (int d = 0; d < D; ++d)
      tp[d] = (unsigned char)collapse_code(__ldg(kh + d));
#pragma unroll 8
    for (int d = 0; d < D; ++d)
      vpack[(size_t)a * D + d] = __ldg(vh + d);
  }
  __syncthreads();

  for (int qi = threadIdx.x; qi < S; qi += blockDim.x) {
    const float *qh = q + ((size_t)base + qi) * (size_t)D;
    float *oh = out + ((size_t)base + qi) * (size_t)D;
    // pack query trits once
    // D typically 64; stack pack
    unsigned char tq[128];
    int Dd = D < 128 ? D : 128;
#pragma unroll 8
    for (int d = 0; d < Dd; ++d)
      tq[d] = (unsigned char)collapse_code(__ldg(qh + d));

    for (int d = 0; d < D; ++d)
      oh[d] = 0.f;
    float active = 0.f;
    for (int a = 0; a < A; ++a) {
      int kj = act[a];
      if (kj > qi)
        continue;
      const unsigned char *tk = trit + (size_t)a * (size_t)D;
      float acc = 0.f;
#pragma unroll 8
      for (int d = 0; d < Dd; ++d) {
        unsigned char a0 = tq[d];
        unsigned char b0 = tk[d];
        if (a0 == 1 || b0 == 1)
          continue;
        acc += (a0 == b0) ? 1.f : -1.f;
      }
      float w = acc / (float)Dd;
      if (w == 0.f)
        continue;
      active += 1.f;
      const float *vh = vpack + (size_t)a * (size_t)D;
#pragma unroll 8
      for (int d = 0; d < D; ++d)
        oh[d] += w * vh[d];
    }
    if (active > 1.f) {
      float inv = 1.f / active;
      for (int d = 0; d < D; ++d)
        oh[d] *= inv;
    }
  }
}

// Short S: light fused (indices only, no trit pack overhead)
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
#pragma unroll 8
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
#pragma unroll 8
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
#pragma unroll 8
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

// Long-S multipass
__global__ void coh_compact_kernel(const float *__restrict__ k,
                                   int *__restrict__ act_idx,
                                   int *__restrict__ act_n, int B, int H, int S,
                                   int D) {
  const int bh = blockIdx.x;
  if (bh >= B * H)
    return;
  const int base = bh * S;
  __shared__ int sA;
  if (threadIdx.x == 0)
    sA = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float *row = k + ((size_t)base + j) * (size_t)D;
    int sharp = 0;
#pragma unroll 8
    for (int d = 0; d < D; ++d)
      if (fabsf(__ldg(row + d)) > COLLAPSE_THRESHOLD)
        sharp++;
    if ((float)sharp / (float)D > COH_GATE) {
      int slot = atomicAdd(&sA, 1);
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
#pragma unroll 8
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
#pragma unroll 8
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
// -1 adaptive, 0 multipass, 1 light, 2 mid pack
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

static size_t mids_shmem(int D) {
  size_t bytes = (size_t)FSOT_MAX_A_SHARED * sizeof(int);
  bytes += (size_t)FSOT_MAX_A_SHARED * (size_t)D;
  bytes = (bytes + 15) & ~((size_t)15);
  bytes += (size_t)FSOT_MAX_A_SHARED * (size_t)D * sizeof(float);
  bytes += 64;
  return bytes;
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

static int launch_light(float *q, float *k, float *v, float *out, int B, int H,
                        int S, int D) {
  int thr = 128;
  size_t shmem = (size_t)FSOT_MAX_A_SHARED * sizeof(int);
  consensus_light_fused<<<B * H, thr, shmem>>>(q, k, v, out, B, H, S, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -11;
}

static int launch_mids(float *q, float *k, float *v, float *out, int B, int H,
                       int S, int D) {
  if (D > 128)
    return launch_multipass(q, k, v, out, B, H, S, D);
  int thr = 256;
  size_t shmem = mids_shmem(D);
  // if shmem too large, multipass
  if (shmem > 48 * 1024)
    return launch_multipass(q, k, v, out, B, H, S, D);
  consensus_mids_kernel<<<B * H, thr, shmem>>>(q, k, v, out, B, H, S, D);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
    return launch_multipass(q, k, v, out, B, H, S, D);
  return 0;
}

EXPORT int fsot_consensus_cuda_device(float *q, float *k, float *v, float *out,
                                      int B, int H, int S, int D) {
  if (D <= 0 || S <= 0 || H <= 0 || B <= 0)
    return -2;

  int mode = g_mode;
  if (mode < 0) {
    // Adaptive under collapse sparsity:
    // short: light fused | mid: trit-pack fused | long: multipass scales past SDPA
    if (S <= 96)
      mode = 1;
    else if (S <= 2048)
      mode = 2;
    else
      mode = 0;
  }
  if (mode == 1)
    return launch_light(q, k, v, out, B, H, S, D);
  if (mode == 2)
    return launch_mids(q, k, v, out, B, H, S, D);
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
