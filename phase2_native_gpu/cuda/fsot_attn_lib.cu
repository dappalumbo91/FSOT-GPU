// FSOT consensus CUDA library — callable from Python (ctypes)
// Layout: q,k,v,out continguous [B,H,S,D] float32 row-major
// Math: collapse θ = C_eff*P_var, coh>0.5, no exp
//
// Build DLL:
//   nvcc -O3 -arch=sm_120 -shared -o fsot_attn_lib.dll fsot_attn_lib.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

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
  for (int d = 0; d < D; ++d)
    if (fabsf(row[d]) > COLLAPSE_THRESHOLD)
      sharp++;
  coh[t] = (float)sharp / (float)D;
}

// Compact active per (b,h) — one block per (b,h)
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
    for (int d = 0; d < D; ++d)
      oh[d] += w * vh[d];
  }
  if (active > 1.f) {
    float inv = 1.f / active;
    for (int d = 0; d < D; ++d)
      oh[d] *= inv;
  }
}

static int last_err = 0;

EXPORT int fsot_consensus_cuda(const float *q_h, const float *k_h,
                               const float *v_h, float *out_h, int B, int H,
                               int S, int D) {
  size_t n = (size_t)B * H * S * D;
  size_t ns = (size_t)B * H * S;
  float *q, *k, *v, *out, *coh;
  int *act, *an;
  if (cudaMalloc(&q, n * sizeof(float)) != cudaSuccess)
    return -1;
  if (cudaMalloc(&k, n * sizeof(float)) != cudaSuccess)
    return -2;
  if (cudaMalloc(&v, n * sizeof(float)) != cudaSuccess)
    return -3;
  if (cudaMalloc(&out, n * sizeof(float)) != cudaSuccess)
    return -4;
  if (cudaMalloc(&coh, ns * sizeof(float)) != cudaSuccess)
    return -5;
  if (cudaMalloc(&act, ns * sizeof(int)) != cudaSuccess)
    return -6;
  if (cudaMalloc(&an, (size_t)B * H * sizeof(int)) != cudaSuccess)
    return -7;

  cudaMemcpy(q, q_h, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(k, k_h, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(v, v_h, n * sizeof(float), cudaMemcpyHostToDevice);

  int thr = 256;
  int blk = (int)((ns + thr - 1) / thr);
  k_coh_kernel<<<blk, thr>>>(k, coh, B, H, S, D);
  compact_kernel<<<B * H, 32>>>(coh, act, an, B, H, S);
  consensus_kernel<<<blk, thr>>>(q, k, v, act, an, out, B, H, S, D);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    last_err = (int)e;
    return -10;
  }
  cudaMemcpy(out_h, out, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(out);
  cudaFree(coh);
  cudaFree(act);
  cudaFree(an);
  return 0;
}

// Device-pointer API (zero-copy when tensors already on GPU) — float* device
EXPORT int fsot_consensus_cuda_device(float *q, float *k, float *v, float *out,
                                      int B, int H, int S, int D) {
  size_t ns = (size_t)B * H * S;
  float *coh;
  int *act, *an;
  if (cudaMalloc(&coh, ns * sizeof(float)) != cudaSuccess)
    return -1;
  if (cudaMalloc(&act, ns * sizeof(int)) != cudaSuccess)
    return -2;
  if (cudaMalloc(&an, (size_t)B * H * sizeof(int)) != cudaSuccess)
    return -3;
  int thr = 256;
  int blk = (int)((ns + thr - 1) / thr);
  k_coh_kernel<<<blk, thr>>>(k, coh, B, H, S, D);
  compact_kernel<<<B * H, 32>>>(coh, act, an, B, H, S);
  consensus_kernel<<<blk, thr>>>(q, k, v, act, an, out, B, H, S, D);
  cudaError_t e = cudaDeviceSynchronize();
  cudaFree(coh);
  cudaFree(act);
  cudaFree(an);
  return e == cudaSuccess ? 0 : -10;
}
