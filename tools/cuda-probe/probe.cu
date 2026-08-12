// tf-cuda-probe
//
// Answers three questions before a single inference kernel is written:
//
//   1. Does this GPU support the primitives every ported Metal kernel needs?
//      Metal's simd_sum becomes a warp shuffle reduction, and Metal 4's
//      matmul2d (the MPP prefill path) becomes a tensor-core MMA. If either is
//      broken here, M5 and M8 are built on sand.
//
//   2. What is the pinned host-to-device bandwidth? The whole streaming design
//      turns on this. Worst case a token needs 30 layers x 8 experts x 3.36 MB
//      = ~806 MB crossing PCIe, so this number sets the ceiling on tok/s once
//      the OS page cache is warm.
//
//   3. How much VRAM is genuinely free? The M6 residency planner tiers layers
//      into resident-vs-streamed against this, and the desktop compositor
//      already owns a slice of it.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        const cudaError_t status_ = (expr);                                   \
        if (status_ != cudaSuccess) {                                         \
            std::printf("  FAIL  %s -> %s\n", #expr, cudaGetErrorString(status_)); \
            ++g_failures;                                                     \
            return;                                                           \
        }                                                                     \
    } while (false)

void report(const char* label, bool ok, const char* detail = nullptr) {
    std::printf("  %-6s %s", ok ? "ok" : "FAIL", label);
    if (detail != nullptr) {
        std::printf(" %s", detail);
    }
    std::printf("\n");
    if (!ok) {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// 1a. Warp shuffle reduction - the CUDA form of Metal's simd_sum(), used by
//     every GEMV, RMSNorm and router kernel in the port.
// ---------------------------------------------------------------------------
__global__ void warpReduceKernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n) {
    const int lane = threadIdx.x & 31;
    const int warp = static_cast<int>(threadIdx.x >> 5);

    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        sum += input[i];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    __shared__ float partial[32];
    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        const int warps = static_cast<int>(blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) {
            total += partial[w];
        }
        *output = total;
    }
}

void probeWarpReduction() {
    constexpr int kN = 4096;
    std::vector<float> host(kN, 0.25f);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dInput, host.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

    warpReduceKernel<<<1, 256>>>(dInput, dOutput, kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, dOutput, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dInput);
    cudaFree(dOutput);

    const float expected = kN * 0.25f;
    char detail[128];
    std::snprintf(detail, sizeof(detail), "(sum=%.1f expected=%.1f)", result, expected);
    report("warp shuffle reduction (simd_sum equivalent)", result == expected, detail);
}

// ---------------------------------------------------------------------------
// 1b. Tensor-core MMA - stands in for Metal Performance Primitives matmul2d on
//     the M8 prefill path. Uses the wmma API rather than raw mma.sync PTX so a
//     fragment-layout mistake here cannot masquerade as missing hardware.
// ---------------------------------------------------------------------------
__global__ void wmmaKernel(const __half* __restrict__ a, const __half* __restrict__ b,
                           float* __restrict__ c) {
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __half, row_major> fragA;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fragB;
    fragment<accumulator, 16, 16, 16, float> fragAcc;

    fill_fragment(fragAcc, 0.0f);
    load_matrix_sync(fragA, a, 16);
    load_matrix_sync(fragB, b, 16);
    mma_sync(fragAcc, fragA, fragB, fragAcc);
    store_matrix_sync(c, fragAcc, 16, mem_row_major);
}

void probeTensorCores() {
    constexpr int kDim = 16;
    constexpr int kElems = kDim * kDim;

    // A is all ones, B is the identity, so C must come back as A.
    std::vector<__half> hostA(kElems, __float2half(1.0f));
    std::vector<__half> hostB(kElems, __float2half(0.0f));
    for (int i = 0; i < kDim; ++i) {
        hostB[static_cast<size_t>(i) * kDim + i] = __float2half(1.0f);
    }

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, kElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, kElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, kElems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hostA.data(), kElems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hostB.data(), kElems * sizeof(__half), cudaMemcpyHostToDevice));

    wmmaKernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hostC(kElems, 0.0f);
    CUDA_CHECK(cudaMemcpy(hostC.data(), dC, kElems * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    bool ok = true;
    for (float value : hostC) {
        if (value != 1.0f) {
            ok = false;
            break;
        }
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "(wmma 16x16x16 f16->f32, c[0]=%.1f)", hostC[0]);
    report("tensor-core MMA (matmul2d equivalent)", ok, detail);
}

// ---------------------------------------------------------------------------
// 2. Pinned host-to-device bandwidth over a 64 MiB buffer, which is the shape
//    the expert staging ring will use.
// ---------------------------------------------------------------------------
void probeTransferBandwidth() {
    constexpr size_t kBytes = 64ull * 1024 * 1024;
    constexpr int kIterations = 20;

    void* pinned = nullptr;
    void* pageable = std::malloc(kBytes);
    void* device = nullptr;

    CUDA_CHECK(cudaHostAlloc(&pinned, kBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&device, kBytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto timeCopies = [&](const void* src) -> float {
        cudaMemcpy(device, src, kBytes, cudaMemcpyHostToDevice);  // warm up
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < kIterations; ++i) {
            cudaMemcpy(device, src, kBytes, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        const double seconds = static_cast<double>(ms) / 1000.0;
        return static_cast<float>((static_cast<double>(kBytes) * kIterations) /
                                  seconds / (1024.0 * 1024.0 * 1024.0));
    };

    const float pinnedGiBs = timeCopies(pinned);
    const float pageableGiBs = (pageable != nullptr) ? timeCopies(pageable) : 0.0f;

    std::printf("  info   pinned   H2D : %6.2f GiB/s\n", pinnedGiBs);
    std::printf("  info   pageable H2D : %6.2f GiB/s  (%.1fx slower)\n", pageableGiBs,
                pageableGiBs > 0.0f ? pinnedGiBs / pageableGiBs : 0.0);

    // Worst-case fully-streamed token: 30 layers x 8 experts x 3.36 MB.
    constexpr double kWorstCaseBytes = 30.0 * 8.0 * 3.36 * 1024.0 * 1024.0;
    const double secondsPerToken = kWorstCaseBytes / (pinnedGiBs * 1024.0 * 1024.0 * 1024.0);
    std::printf("  info   PCIe ceiling : %6.2f tok/s if every expert missed cache\n"
                "                        (%.0f MB/token over pinned H2D)\n",
                1.0 / secondsPerToken, kWorstCaseBytes / (1024.0 * 1024.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(pinned);
    cudaFree(device);
    std::free(pageable);
}

void printDeviceInfo(const cudaDeviceProp& prop, int device) {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);

    // CUDA 13 dropped memoryClockRate and memoryBusWidth from cudaDeviceProp;
    // the attribute API is the supported way to read them now.
    int memoryClockKHz = 0;
    int memoryBusWidth = 0;
    cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, device);

    std::printf("Device %d: %s\n", device, prop.name);
    std::printf("  compute capability : %d.%d (sm_%d%d)\n", prop.major, prop.minor,
                prop.major, prop.minor);
    std::printf("  SMs                : %d\n", prop.multiProcessorCount);
    std::printf("  warp size          : %d\n", prop.warpSize);
    std::printf("  VRAM total         : %.2f GiB\n",
                static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  VRAM free          : %.2f GiB  <- residency planner budget\n",
                static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  memory bus         : %d-bit @ %.2f GHz (%.0f GB/s peak)\n",
                memoryBusWidth, memoryClockKHz / 1.0e6,
                (2.0 * memoryClockKHz * 1000.0 * memoryBusWidth / 8.0) / 1.0e9);
    std::printf("  shared mem / block : %zu KiB\n", prop.sharedMemPerBlock / 1024);
    std::printf("  async engines      : %d  (overlapping copy with compute)\n",
                prop.asyncEngineCount);
    std::printf("  unified addressing : %s\n", prop.unifiedAddressing ? "yes" : "no");
    std::printf("  integrated (UMA)   : %s\n", prop.integrated ? "yes" : "no");
    std::printf("\n");
}

}  // namespace

int main() {
    std::printf("tf-cuda-probe\n\n");

    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    std::printf("CUDA runtime %d.%d, driver supports %d.%d\n\n", runtimeVersion / 1000,
                (runtimeVersion % 1000) / 10, driverVersion / 1000,
                (driverVersion % 1000) / 10);

    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess || deviceCount == 0) {
        std::printf("No CUDA device available: %s\n", cudaGetErrorString(countStatus));
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printDeviceInfo(prop, 0);

    std::printf("Kernel primitives\n");
    probeWarpReduction();
    probeTensorCores();

    std::printf("\nTransfer bandwidth\n");
    probeTransferBandwidth();

    std::printf("\n%s\n", g_failures == 0 ? "All probes passed."
                                          : "SOME PROBES FAILED - see above.");
    return g_failures == 0 ? 0 : 1;
}
