// The C++23 half of the CUDA kernel implementation.
//
// Argument validation, error messages and Result live here; CudaKernels.cu
// holds the kernels themselves. The split exists because nvcc compiles at
// C++20 and cannot see std::expected, but it also keeps every diagnostic in
// one place and out of device code.

#include <cmath>
#include <memory>

#include "CudaLaunch.h"
#include "tf/gpu/Kernels.h"

namespace tf::gpu {

/// Defined in CudaBackend.cpp.
void* cudaDevicePointer(const Buffer& buffer);
cudaStream_t cudaStreamOf(Stream& stream);

namespace {

[[nodiscard]] Error launchError(cudaError_t status, std::string_view what) {
    return Error{ErrorCode::GpuFailure,
                 std::format("{}: {} ({})", what, cudaGetErrorString(status),
                             cudaGetErrorName(status))};
}

[[nodiscard]] void* rawPointer(const DeviceView& view) {
    if (!view.valid()) {
        return nullptr;
    }
    auto* base = static_cast<u8*>(cudaDevicePointer(*view.buffer));
    return base == nullptr ? nullptr : base + view.offset;
}

template <class T>
[[nodiscard]] T* pointerTo(const DeviceView& view) {
    return static_cast<T*>(rawPointer(view));
}

/// Rejects an unset view before it becomes a null dereference on the device,
/// and rejects one whose extent runs past its buffer.
[[nodiscard]] Status requireView(const DeviceView& view, u64 bytes, std::string_view name) {
    if (!view.valid()) {
        return makeError(ErrorCode::InvalidArgument, "{} is not set", name);
    }
    if (view.offset > view.buffer->size() || bytes > view.buffer->size() - view.offset) {
        return makeError(ErrorCode::InvalidArgument,
                         "{}: [{}, {}) runs past buffer '{}' of {} bytes", name, view.offset,
                         view.offset + bytes, view.buffer->debugName(), view.buffer->size());
    }
    if (rawPointer(view) == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "{} is not device-addressable", name);
    }
    return {};
}

// Activations are fp32: Gemma 4's do not fit in fp16. See CudaLaunch.h.
constexpr u64 kActivationBytes = 4;
// The KV cache stays fp16, which is safe because keys and values are
// post-normalization and bounded near unit RMS.
constexpr u64 kKvBytes = 2;
constexpr u64 kBf16Bytes = 2;

/// Checks the three tensors of a quantized linear against the layout's own
/// idea of their sizes, so a mis-sliced expert is caught at the call rather
/// than as silently wrong arithmetic.
[[nodiscard]] Status requireWeights(const QuantizedWeights& weights, std::string_view name) {
    TF_CHECK(weights.layout.validate());
    TF_CHECK(requireView(weights.packed, weights.layout.weightBytes(),
                         std::format("{} packed", name)));
    TF_CHECK(requireView(weights.scales, weights.layout.scaleBytes(),
                         std::format("{} scales", name)));
    TF_CHECK(requireView(weights.biases, weights.layout.biasBytes(),
                         std::format("{} biases", name)));
    return {};
}

class CudaKernels final : public IKernels {
public:
    [[nodiscard]] Status rmsNorm(Stream& stream, const RmsNormArgs& args) override {
        if (args.count == 0 || args.rows == 0) {
            return makeError(ErrorCode::InvalidArgument, "rmsNorm has a zero extent");
        }
        const u64 bytes = u64{args.rows} * args.count * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "rmsNorm input"));
        TF_CHECK(requireView(args.output, bytes, "rmsNorm output"));
        if (args.weight.valid()) {
            TF_CHECK(requireView(args.weight, u64{args.count} * kBf16Bytes,
                                 "rmsNorm weight"));
        }

        const cudaError_t status = launch::rmsNorm(
                cudaStreamOf(stream), pointerTo<const float>(args.input),
                pointerTo<const __nv_bfloat16>(args.weight), pointerTo<float>(args.output),
                args.rows, args.count, args.eps);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching rmsNorm"));
        }
        return {};
    }

    [[nodiscard]] Status dequantGemv(Stream& stream, const DequantGemvArgs& args) override {
        const auto& layout = args.weights.layout;
        TF_CHECK(requireWeights(args.weights, "gemv"));
        TF_CHECK(requireView(args.input, layout.inFeatures * kActivationBytes, "gemv input"));
        TF_CHECK(requireView(args.output, layout.outFeatures * kActivationBytes, "gemv output"));

        const cudaError_t status = launch::dequantGemv(
                cudaStreamOf(stream), layout.spec.bits,
                pointerTo<const u32>(args.weights.packed),
                pointerTo<const __nv_bfloat16>(args.weights.scales),
                pointerTo<const __nv_bfloat16>(args.weights.biases),
                pointerTo<const float>(args.input), pointerTo<float>(args.output),
                static_cast<u32>(layout.outFeatures), static_cast<u32>(layout.inFeatures),
                layout.spec.groupSize);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching dequantGemv"));
        }
        return {};
    }

    [[nodiscard]] Status dequantGemm(Stream& stream, const DequantGemmArgs& args) override {
        const auto& layout = args.weights.layout;
        if (args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "gemm has no tokens");
        }
        TF_CHECK(requireWeights(args.weights, "gemm"));
        TF_CHECK(requireView(args.input, u64{args.tokens} * layout.inFeatures * kActivationBytes,
                             "gemm input"));
        TF_CHECK(requireView(args.output,
                             u64{args.tokens} * layout.outFeatures * kActivationBytes,
                             "gemm output"));

        // One token is the decode path, and the batched kernel is the wrong
        // shape for it: it blocks over 16 tokens, so a single one leaves 15/16
        // of every shared tile empty and does sixteen multiply-adds per weight
        // where one is wanted. Measured at 21.6 tok/s against the GEMV's 38.7.
        // Same arithmetic either way, so this is purely kernel selection and
        // belongs here rather than in the runtime.
        const cudaError_t status =
                args.tokens == 1
                        ? launch::dequantGemv(
                                  cudaStreamOf(stream), layout.spec.bits,
                                  pointerTo<const u32>(args.weights.packed),
                                  pointerTo<const __nv_bfloat16>(args.weights.scales),
                                  pointerTo<const __nv_bfloat16>(args.weights.biases),
                                  pointerTo<const float>(args.input),
                                  pointerTo<float>(args.output),
                                  static_cast<u32>(layout.outFeatures),
                                  static_cast<u32>(layout.inFeatures), layout.spec.groupSize)
                        : launch::dequantGemm(
                                  cudaStreamOf(stream), layout.spec.bits,
                                  pointerTo<const u32>(args.weights.packed),
                                  pointerTo<const __nv_bfloat16>(args.weights.scales),
                                  pointerTo<const __nv_bfloat16>(args.weights.biases),
                                  pointerTo<const float>(args.input),
                                  pointerTo<float>(args.output), args.tokens,
                                  static_cast<u32>(layout.outFeatures),
                                  static_cast<u32>(layout.inFeatures), layout.spec.groupSize);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching dequantGemm"));
        }
        return {};
    }

    [[nodiscard]] u32 gemmTokenTile() const override { return launch::dequantGemmTokenTile(); }

    [[nodiscard]] Status embedLookup(Stream& stream, const EmbedLookupArgs& args) override {
        const auto& layout = args.table.layout;
        TF_CHECK(requireWeights(args.table, "embedding table"));
        TF_CHECK(requireView(args.output, layout.inFeatures * kActivationBytes,
                             "embedding output"));

        if (args.tokenId >= layout.outFeatures) {
            return makeError(ErrorCode::InvalidArgument,
                             "token id {} is outside a vocabulary of {}", args.tokenId,
                             layout.outFeatures);
        }

        const auto hiddenSize = static_cast<u32>(layout.inFeatures);
        const auto embedScale = static_cast<float>(std::sqrt(static_cast<double>(hiddenSize)));

        const cudaError_t status = launch::embedLookup(
                cudaStreamOf(stream), layout.spec.bits,
                pointerTo<const u32>(args.table.packed),
                pointerTo<const __nv_bfloat16>(args.table.scales),
                pointerTo<const __nv_bfloat16>(args.table.biases),
                pointerTo<float>(args.output), args.tokenId, hiddenSize,
                layout.spec.groupSize, embedScale);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching embedLookup"));
        }
        return {};
    }

    [[nodiscard]] Status rope(Stream& stream, const RopeArgs& args) override {
        if (args.heads == 0 || args.headDim == 0 || args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "rope has a zero extent");
        }
        TF_CHECK(requireView(args.data,
                             u64{args.tokens} * args.heads * args.headDim * kActivationBytes,
                             "rope data"));

        // Must stay even: the rotation pairs dimensions.
        auto rotated = static_cast<u32>(static_cast<float>(args.headDim) *
                                        args.partialRotaryFactor);
        rotated -= rotated % 2;
        if (rotated == 0) {
            return {};  // nothing to rotate
        }
        if (rotated > args.headDim) {
            return makeError(ErrorCode::InvalidArgument,
                             "partial rotary factor {} exceeds the head dimension",
                             args.partialRotaryFactor);
        }

        const cudaError_t status =
                launch::rope(cudaStreamOf(stream), pointerTo<float>(args.data), args.tokens,
                             args.heads, args.headDim, rotated, args.position, args.theta);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching rope"));
        }
        return {};
    }

    [[nodiscard]] Status geglu(Stream& stream, const GegluArgs& args) override {
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.gate, bytes, "geglu gate"));
        TF_CHECK(requireView(args.up, bytes, "geglu up"));
        TF_CHECK(requireView(args.output, bytes, "geglu output"));

        const cudaError_t status = launch::geglu(
                cudaStreamOf(stream), pointerTo<const float>(args.gate),
                pointerTo<const float>(args.up), pointerTo<float>(args.output), args.count);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching geglu"));
        }
        return {};
    }

    [[nodiscard]] Status add(Stream& stream, const AddArgs& args) override {
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.a, bytes, "add a"));
        TF_CHECK(requireView(args.b, bytes, "add b"));
        TF_CHECK(requireView(args.output, bytes, "add output"));

        const cudaError_t status =
                launch::add(cudaStreamOf(stream), pointerTo<const float>(args.a),
                            pointerTo<const float>(args.b), pointerTo<float>(args.output),
                            args.count);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching add"));
        }
        return {};
    }

    [[nodiscard]] Status scale(Stream& stream, const ScaleArgs& args) override {
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "scale input"));
        TF_CHECK(requireView(args.output, bytes, "scale output"));

        const cudaError_t status =
                launch::scale(cudaStreamOf(stream), pointerTo<const float>(args.input),
                              pointerTo<float>(args.output), args.count, args.scalar);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching scale"));
        }
        return {};
    }

    [[nodiscard]] Status logitSoftcap(Stream& stream,
                                      const LogitSoftcapArgs& args) override {
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "softcap input"));
        TF_CHECK(requireView(args.output, bytes, "softcap output"));
        if (args.cap == 0.0f) {
            return makeError(ErrorCode::InvalidArgument, "softcap of zero would divide by 0");
        }

        const cudaError_t status = launch::logitSoftcap(
                cudaStreamOf(stream), pointerTo<const float>(args.input),
                pointerTo<float>(args.output), args.count, args.cap);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching logitSoftcap"));
        }
        return {};
    }

    [[nodiscard]] Status argmax(Stream& stream, const ArgmaxArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "argmax over an empty vector");
        }
        TF_CHECK(requireView(args.input, u64{args.count} * kActivationBytes, "argmax input"));
        TF_CHECK(requireView(args.output, sizeof(u32), "argmax output"));

        const cudaError_t status =
                launch::argmax(cudaStreamOf(stream), pointerTo<const float>(args.input),
                               pointerTo<u32>(args.output), args.count);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching argmax"));
        }
        return {};
    }

    [[nodiscard]] Status kvWrite(Stream& stream, const KvWriteArgs& args) override {
        if (args.kvHeads == 0 || args.headDim == 0 || args.capacity == 0 || args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "kvWrite has a zero extent");
        }
        // A linear cache must not be written past its end; a ring wraps by
        // design, so only the linear case is bounded by position. The last
        // token of the batch is the one that can run off the end.
        if (!args.circular && args.position + args.tokens > args.capacity) {
            return makeError(ErrorCode::InvalidArgument,
                             "positions {}..{} exceed the linear cache capacity of {}",
                             args.position, args.position + args.tokens - 1, args.capacity);
        }

        const u64 tokenBytes =
                u64{args.tokens} * args.kvHeads * args.headDim * kActivationBytes;
        const u64 cacheBytes = u64{args.kvHeads} * args.capacity * args.headDim * kKvBytes;
        TF_CHECK(requireView(args.key, tokenBytes, "kvWrite key"));
        TF_CHECK(requireView(args.value, tokenBytes, "kvWrite value"));
        TF_CHECK(requireView(args.keyCache, cacheBytes, "kvWrite key cache"));
        TF_CHECK(requireView(args.valueCache, cacheBytes, "kvWrite value cache"));

        const cudaError_t status = launch::kvWrite(
                cudaStreamOf(stream), pointerTo<const float>(args.key),
                pointerTo<const float>(args.value), pointerTo<__half>(args.keyCache),
                pointerTo<__half>(args.valueCache), args.tokens, args.kvHeads, args.headDim,
                args.capacity, args.position, args.circular);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching kvWrite"));
        }
        return {};
    }

    [[nodiscard]] Status attention(Stream& stream, const AttentionArgs& args) override {
        if (args.numHeads == 0 || args.kvHeads == 0 || args.headDim == 0 ||
            args.capacity == 0 || args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "attention has a zero extent");
        }
        if (args.numHeads % args.kvHeads != 0) {
            return makeError(ErrorCode::InvalidArgument,
                             "{} query heads do not divide evenly over {} KV heads",
                             args.numHeads, args.kvHeads);
        }
        // The last query of the chunk determines how much history is read.
        const u64 cachedLength = args.basePosition + args.tokens;
        if (!args.circular && cachedLength > args.capacity) {
            return makeError(ErrorCode::InvalidArgument,
                             "cached length {} exceeds the linear cache capacity of {}",
                             cachedLength, args.capacity);
        }

        const u64 queryBytes =
                u64{args.tokens} * args.numHeads * args.headDim * kActivationBytes;
        const u64 cacheBytes = u64{args.kvHeads} * args.capacity * args.headDim * kKvBytes;
        TF_CHECK(requireView(args.queries, queryBytes, "attention queries"));
        TF_CHECK(requireView(args.output, queryBytes, "attention output"));
        TF_CHECK(requireView(args.keyCache, cacheBytes, "attention key cache"));
        TF_CHECK(requireView(args.valueCache, cacheBytes, "attention value cache"));

        const cudaError_t status = launch::attention(
                cudaStreamOf(stream), pointerTo<const float>(args.queries),
                pointerTo<const __half>(args.keyCache),
                pointerTo<const __half>(args.valueCache), pointerTo<float>(args.output),
                args.tokens, args.numHeads, args.kvHeads, args.headDim, args.capacity,
                args.basePosition, args.slidingWindow, args.circular, args.scale);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching attention"));
        }
        return {};
    }

    [[nodiscard]] Status gatherRows(Stream& stream, const GatherRowsArgs& args) override {
        if (args.width == 0) {
            return makeError(ErrorCode::InvalidArgument, "gatherRows has a zero width");
        }
        if (args.count == 0) {
            return {};
        }
        TF_CHECK(requireView(args.rows, u64{args.count} * sizeof(u32), "gatherRows rows"));
        TF_CHECK(requireView(args.output, u64{args.count} * args.width * kActivationBytes,
                             "gatherRows output"));
        // The source is indexed indirectly, so its extent is not known here.
        if (!args.input.valid()) {
            return makeError(ErrorCode::InvalidArgument, "gatherRows input is unset");
        }

        const cudaError_t status = launch::gatherRows(
                cudaStreamOf(stream), pointerTo<const float>(args.input),
                pointerTo<const u32>(args.rows), pointerTo<float>(args.output), args.count,
                args.width);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching gatherRows"));
        }
        return {};
    }

    [[nodiscard]] Status scatterAddRows(Stream& stream,
                                        const ScatterAddRowsArgs& args) override {
        if (args.width == 0) {
            return makeError(ErrorCode::InvalidArgument, "scatterAddRows has a zero width");
        }
        if (args.count == 0) {
            return {};
        }
        TF_CHECK(requireView(args.input, u64{args.count} * args.width * kActivationBytes,
                             "scatterAddRows input"));
        TF_CHECK(requireView(args.rows, u64{args.count} * sizeof(u32),
                             "scatterAddRows rows"));
        TF_CHECK(requireView(args.scales, u64{args.count} * kActivationBytes,
                             "scatterAddRows scales"));
        if (!args.output.valid()) {
            return makeError(ErrorCode::InvalidArgument, "scatterAddRows output is unset");
        }

        const cudaError_t status = launch::scatterAddRows(
                cudaStreamOf(stream), pointerTo<const float>(args.input),
                pointerTo<const u32>(args.rows), pointerTo<const float>(args.scales),
                pointerTo<float>(args.output), args.count, args.width);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching scatterAddRows"));
        }
        return {};
    }

    [[nodiscard]] Status fillZero(Stream& stream, const FillZeroArgs& args) override {
        if (args.count == 0) {
            return {};
        }
        TF_CHECK(requireView(args.output, args.count * kActivationBytes, "fillZero output"));

        const cudaError_t status = launch::fillZero(
                cudaStreamOf(stream), pointerTo<float>(args.output), args.count);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching fillZero"));
        }
        return {};
    }

    [[nodiscard]] Status routerTopK(Stream& stream, const RouterTopKArgs& args) override {
        if (args.numExperts == 0 || args.topK == 0) {
            return makeError(ErrorCode::InvalidArgument, "router has a zero extent");
        }
        if (args.topK > args.numExperts) {
            return makeError(ErrorCode::InvalidArgument,
                             "top-{} requested from only {} experts", args.topK,
                             args.numExperts);
        }

        if (args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "router has no tokens");
        }

        TF_CHECK(requireView(args.scores,
                             u64{args.tokens} * args.numExperts * kActivationBytes,
                             "router scores"));
        TF_CHECK(requireView(args.perExpertScale, u64{args.numExperts} * kBf16Bytes,
                             "router per-expert scale"));
        TF_CHECK(requireView(args.outIndices, u64{args.tokens} * args.topK * sizeof(u32),
                             "router indices"));
        TF_CHECK(requireView(args.outWeights, u64{args.tokens} * args.topK * kActivationBytes,
                             "router weights"));

        const cudaError_t status = launch::routerTopK(
                cudaStreamOf(stream), pointerTo<const float>(args.scores),
                pointerTo<const __nv_bfloat16>(args.perExpertScale),
                pointerTo<u32>(args.outIndices), pointerTo<float>(args.outWeights), args.tokens,
                args.numExperts, args.topK);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching routerTopK"));
        }
        return {};
    }

    [[nodiscard]] Status moeCombine(Stream& stream, const MoeCombineArgs& args) override {
        if (args.topK == 0 || args.hidden == 0) {
            return makeError(ErrorCode::InvalidArgument, "moeCombine has a zero extent");
        }
        TF_CHECK(requireView(args.expertOutputs, u64{args.topK} * args.hidden * kActivationBytes,
                             "expert outputs"));
        TF_CHECK(requireView(args.weights, u64{args.topK} * kActivationBytes, "expert weights"));
        TF_CHECK(requireView(args.output, u64{args.hidden} * kActivationBytes, "combine output"));

        const cudaError_t status = launch::moeCombine(
                cudaStreamOf(stream), pointerTo<const float>(args.expertOutputs),
                pointerTo<const float>(args.weights), pointerTo<float>(args.output),
                args.topK, args.hidden);
        if (status != cudaSuccess) {
            return std::unexpected(launchError(status, "launching moeCombine"));
        }
        return {};
    }
};

}  // namespace

std::unique_ptr<IKernels> createCudaKernels() { return std::make_unique<CudaKernels>(); }

}  // namespace tf::gpu
