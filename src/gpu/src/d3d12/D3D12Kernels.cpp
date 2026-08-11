// Binding and dispatch for the D3D12 shaders.
//
// The counterpart of CudaKernelDispatch.cpp: every argument check and error
// message lives here, and the shaders themselves assume their inputs are sane.
// Unlike the CUDA side there is no language barrier forcing the split - it is
// here because the same validation should not be written twice, and because a
// bad extent must fail the same way on both backends.

#include <algorithm>
#include <array>
#include <cmath>

#include "D3D12Common.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"

namespace tf::gpu::d3d12 {

// Defined in D3D12Backend.cpp; declared here to keep the two files from needing
// a shared private header for three symbols.
class D3D12Backend;
class D3D12Buffer;
class D3D12Stream;

[[nodiscard]] ID3D12Device* backendDevice(D3D12Backend& backend);
[[nodiscard]] ID3D12RootSignature* backendRootSignature(D3D12Backend& backend);
[[nodiscard]] ID3D12PipelineState* backendPipeline(D3D12Backend& backend, ShaderId id);
[[nodiscard]] Status streamBegin(Stream& stream);
[[nodiscard]] ID3D12GraphicsCommandList* streamList(Stream& stream);
[[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS viewAddress(const DeviceView& view);
[[nodiscard]] bool viewIsDevice(const DeviceView& view);
void streamBarrierAll(Stream& stream);

namespace {

constexpr u64 kActivationBytes = 4;
constexpr u64 kKvBytes = 2;
constexpr u64 kBf16Bytes = 2;
constexpr u32 kBlock = 256;

[[nodiscard]] u32 groupsFor(u64 count, u32 size) {
    return static_cast<u32>((count + size - 1) / size);
}

[[nodiscard]] u32 asBits(float value) {
    u32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

[[nodiscard]] Status requireView(const DeviceView& view, u64 bytes, std::string_view name) {
    if (!view.valid()) {
        return makeError(ErrorCode::InvalidArgument, "{} is not set", name);
    }
    if (view.offset > view.buffer->size() || bytes > view.buffer->size() - view.offset) {
        return makeError(ErrorCode::InvalidArgument,
                         "{}: [{}, {}) runs past buffer '{}' of {} bytes", name, view.offset,
                         view.offset + bytes, view.buffer->debugName(), view.buffer->size());
    }
    if (!viewIsDevice(view)) {
        return makeError(ErrorCode::InvalidArgument, "{} is not device memory", name);
    }
    return {};
}

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

/// One dispatch: bind the pipeline, up to six raw buffers and sixteen root
/// constants, then Dispatch.
///
/// Root descriptors carry a byte address, so a DeviceView's offset folds into
/// the address and the shader indexes from zero. That is why no shader takes an
/// offset parameter.
class Dispatch {
public:
    Dispatch(D3D12Backend& backend, Stream& stream, ShaderId shader)
        : backend_(backend), stream_(stream), shader_(shader) {}

    Dispatch& bind(u32 slot, const DeviceView& view) {
        addresses_[slot] = viewAddress(view);
        return *this;
    }

    Dispatch& constants(std::initializer_list<u32> values) {
        usize index = 0;
        for (const u32 value : values) {
            if (index < constants_.size()) {
                constants_[index++] = value;
            }
        }
        return *this;
    }

    [[nodiscard]] Status go(u32 x, u32 y = 1, u32 z = 1) {
        TF_CHECK(streamBegin(stream_));
        ID3D12GraphicsCommandList* list = streamList(stream_);

        list->SetComputeRootSignature(backendRootSignature(backend_));
        list->SetPipelineState(backendPipeline(backend_, shader_));
        for (u32 slot = 0; slot < addresses_.size(); ++slot) {
            if (addresses_[slot] != 0) {
                list->SetComputeRootUnorderedAccessView(slot, addresses_[slot]);
            }
        }
        list->SetComputeRoot32BitConstants(8, static_cast<UINT>(constants_.size()),
                                           constants_.data(), 0);
        list->Dispatch(x, y, z);

        // Every kernel here reads what the previous one wrote, and D3D12 will
        // otherwise overlap them. CUDA gets this ordering from the stream; here
        // it has to be said.
        streamBarrierAll(stream_);
        return {};
    }

private:
    D3D12Backend& backend_;
    Stream& stream_;
    ShaderId shader_;
    std::array<D3D12_GPU_VIRTUAL_ADDRESS, 6> addresses_{};
    std::array<u32, 16> constants_{};
};

class D3D12Kernels final : public IKernels {
public:
    explicit D3D12Kernels(D3D12Backend& backend) : backend_(backend) {}

    [[nodiscard]] Status rmsNorm(Stream& stream, const RmsNormArgs& args) override {
        if (args.count == 0 || args.rows == 0) {
            return makeError(ErrorCode::InvalidArgument, "rmsNorm has a zero extent");
        }
        const u64 bytes = u64{args.rows} * args.count * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "rmsNorm input"));
        TF_CHECK(requireView(args.output, bytes, "rmsNorm output"));

        const bool weighted = args.weight.valid();
        if (weighted) {
            TF_CHECK(requireView(args.weight, u64{args.count} * kBf16Bytes, "rmsNorm weight"));
        }

        Dispatch dispatch{backend_, stream,
                          weighted ? ShaderId::RmsNorm : ShaderId::RmsNormNoWeight};
        dispatch.bind(0, args.input).bind(2, args.output);
        if (weighted) {
            dispatch.bind(1, args.weight);
        }
        return dispatch.constants({args.count, 0, 0, 0, asBits(args.eps)}).go(args.rows);
    }

    [[nodiscard]] Status dequantGemv(Stream& stream, const DequantGemvArgs& args) override {
        const auto& layout = args.weights.layout;
        TF_CHECK(requireWeights(args.weights, "gemv"));
        TF_CHECK(requireView(args.input, layout.inFeatures * kActivationBytes, "gemv input"));
        TF_CHECK(requireView(args.output, layout.outFeatures * kActivationBytes,
                             "gemv output"));

        const ShaderId shader =
                layout.spec.bits == 4 ? ShaderId::DequantGemv4 : ShaderId::DequantGemv8;
        if (layout.spec.bits != 4 && layout.spec.bits != 8) {
            return makeError(ErrorCode::Unsupported, "{}-bit weights are not supported",
                             layout.spec.bits);
        }

        // Waves per group is not known until the device reports its lane count,
        // and the shader computes its own row from it, so the group count is
        // the conservative one that covers every lane width.
        const u32 groups = groupsFor(layout.outFeatures, kBlock / 32);
        return Dispatch{backend_, stream, shader}
                .bind(0, args.weights.packed)
                .bind(1, args.weights.scales)
                .bind(2, args.weights.biases)
                .bind(3, args.input)
                .bind(4, args.output)
                .constants({static_cast<u32>(layout.outFeatures),
                            static_cast<u32>(layout.inFeatures), layout.spec.groupSize,
                            layout.spec.bits})
                .go(groups);
    }

    [[nodiscard]] Status dequantGemm(Stream& stream, const DequantGemmArgs& args) override {
        const auto& layout = args.weights.layout;
        if (args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "gemm has no tokens");
        }
        TF_CHECK(requireWeights(args.weights, "gemm"));
        TF_CHECK(requireView(args.input,
                             u64{args.tokens} * layout.inFeatures * kActivationBytes,
                             "gemm input"));
        TF_CHECK(requireView(args.output,
                             u64{args.tokens} * layout.outFeatures * kActivationBytes,
                             "gemm output"));
        if (layout.spec.bits != 4 && layout.spec.bits != 8) {
            return makeError(ErrorCode::Unsupported, "{}-bit weights are not supported",
                             layout.spec.bits);
        }

        // One token is the decode path, and the batched shader is the wrong
        // shape for it: fifteen sixteenths of every tile would be empty. Same
        // reasoning as the CUDA dispatch.
        if (args.tokens == 1) {
            return dequantGemv(stream, DequantGemvArgs{.weights = args.weights,
                                                       .input = args.input,
                                                       .output = args.output});
        }

        constexpr u32 kTokenTile = 16;
        constexpr u32 kRows = 4;
        const ShaderId shader =
                layout.spec.bits == 4 ? ShaderId::DequantGemm4 : ShaderId::DequantGemm8;

        return Dispatch{backend_, stream, shader}
                .bind(0, args.weights.packed)
                .bind(1, args.weights.scales)
                .bind(2, args.weights.biases)
                .bind(3, args.input)
                .bind(4, args.output)
                .constants({static_cast<u32>(layout.outFeatures),
                            static_cast<u32>(layout.inFeatures), layout.spec.groupSize,
                            layout.spec.bits, args.tokens})
                .go(groupsFor(layout.outFeatures, (kBlock / 32) * kRows),
                    groupsFor(args.tokens, kTokenTile));
    }

    [[nodiscard]] u32 gemmTokenTile() const override { return 16; }

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
        if (layout.spec.bits != 4 && layout.spec.bits != 8) {
            return makeError(ErrorCode::Unsupported, "{}-bit weights are not supported",
                             layout.spec.bits);
        }

        const auto hiddenSize = static_cast<u32>(layout.inFeatures);
        const auto embedScale = static_cast<float>(std::sqrt(static_cast<double>(hiddenSize)));
        const ShaderId shader =
                layout.spec.bits == 4 ? ShaderId::EmbedLookup4 : ShaderId::EmbedLookup8;

        return Dispatch{backend_, stream, shader}
                .bind(0, args.table.packed)
                .bind(1, args.table.scales)
                .bind(2, args.table.biases)
                .bind(4, args.output)
                .constants({hiddenSize, layout.spec.groupSize, args.tokenId, layout.spec.bits,
                            asBits(embedScale)})
                .go(groupsFor(hiddenSize, kBlock));
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
            return {};
        }
        if (rotated > args.headDim) {
            return makeError(ErrorCode::InvalidArgument,
                             "partial rotary factor {} exceeds the head dimension",
                             args.partialRotaryFactor);
        }

        return Dispatch{backend_, stream, ShaderId::Rope}
                .bind(0, args.data)
                .constants({args.heads, args.headDim, rotated, 0, asBits(args.theta), 0, 0, 0,
                            static_cast<u32>(args.position)})
                .go(args.heads, args.tokens);
    }

    [[nodiscard]] Status geglu(Stream& stream, const GegluArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "geglu has a zero extent");
        }
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.gate, bytes, "geglu gate"));
        TF_CHECK(requireView(args.up, bytes, "geglu up"));
        TF_CHECK(requireView(args.output, bytes, "geglu output"));

        return Dispatch{backend_, stream, ShaderId::Geglu}
                .bind(0, args.gate)
                .bind(1, args.up)
                .bind(2, args.output)
                .constants({args.count})
                .go(groupsFor(args.count, kBlock));
    }

    [[nodiscard]] Status add(Stream& stream, const AddArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "add has a zero extent");
        }
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.a, bytes, "add a"));
        TF_CHECK(requireView(args.b, bytes, "add b"));
        TF_CHECK(requireView(args.output, bytes, "add output"));

        return Dispatch{backend_, stream, ShaderId::Add}
                .bind(0, args.a)
                .bind(1, args.b)
                .bind(2, args.output)
                .constants({args.count})
                .go(groupsFor(args.count, kBlock));
    }

    [[nodiscard]] Status scale(Stream& stream, const ScaleArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "scale has a zero extent");
        }
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "scale input"));
        TF_CHECK(requireView(args.output, bytes, "scale output"));

        return Dispatch{backend_, stream, ShaderId::Scale}
                .bind(0, args.input)
                .bind(2, args.output)
                .constants({args.count, 0, 0, 0, asBits(args.scalar)})
                .go(groupsFor(args.count, kBlock));
    }

    [[nodiscard]] Status logitSoftcap(Stream& stream, const LogitSoftcapArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "logit softcap has a zero extent");
        }
        const u64 bytes = u64{args.count} * kActivationBytes;
        TF_CHECK(requireView(args.input, bytes, "softcap input"));
        TF_CHECK(requireView(args.output, bytes, "softcap output"));

        return Dispatch{backend_, stream, ShaderId::LogitSoftcap}
                .bind(0, args.input)
                .bind(2, args.output)
                .constants({args.count, 0, 0, 0, asBits(args.cap)})
                .go(groupsFor(args.count, kBlock));
    }

    [[nodiscard]] Status argmax(Stream& stream, const ArgmaxArgs& args) override {
        if (args.count == 0) {
            return makeError(ErrorCode::InvalidArgument, "argmax has a zero extent");
        }
        TF_CHECK(requireView(args.input, u64{args.count} * kActivationBytes, "argmax input"));
        TF_CHECK(requireView(args.output, sizeof(u32), "argmax output"));

        return Dispatch{backend_, stream, ShaderId::Argmax}
                .bind(0, args.input)
                .bind(2, args.output)
                .constants({args.count})
                .go(1);
    }

    [[nodiscard]] Status kvWrite(Stream& stream, const KvWriteArgs& args) override {
        if (args.kvHeads == 0 || args.headDim == 0 || args.capacity == 0 || args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "kvWrite has a zero extent");
        }
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

        return Dispatch{backend_, stream, ShaderId::KvWrite}
                .bind(0, args.key)
                .bind(1, args.value)
                .bind(2, args.keyCache)
                .bind(3, args.valueCache)
                .constants({args.kvHeads, args.headDim, args.capacity,
                            args.circular ? 1u : 0u, static_cast<u32>(args.position)})
                .go(args.kvHeads, args.tokens);
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
        if (args.headDim > 512) {
            // The shader's groupshared accumulator is sized for the largest
            // head this architecture uses.
            return makeError(ErrorCode::Unsupported,
                             "a head dimension of {} exceeds the 512 these shaders allocate",
                             args.headDim);
        }
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

        return Dispatch{backend_, stream, ShaderId::Attention}
                .bind(0, args.queries)
                .bind(1, args.keyCache)
                .bind(2, args.valueCache)
                .bind(4, args.output)
                .constants({args.numHeads, args.kvHeads, args.headDim, args.capacity,
                            static_cast<u32>(args.basePosition), args.slidingWindow,
                            args.circular ? 1u : 0u, 0, asBits(args.scale)})
                .go(args.numHeads, args.tokens);
    }

    [[nodiscard]] Status routerTopK(Stream& stream, const RouterTopKArgs& args) override {
        if (args.numExperts == 0 || args.topK == 0 || args.tokens == 0) {
            return makeError(ErrorCode::InvalidArgument, "router has a zero extent");
        }
        if (args.topK > args.numExperts) {
            return makeError(ErrorCode::InvalidArgument,
                             "top-{} requested from only {} experts", args.topK,
                             args.numExperts);
        }
        if (args.numExperts > 256 || args.topK > 32) {
            return makeError(ErrorCode::Unsupported,
                             "the router shader holds at most 256 experts and a top-32");
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

        return Dispatch{backend_, stream, ShaderId::RouterTopK}
                .bind(0, args.scores)
                .bind(1, args.perExpertScale)
                .bind(2, args.outIndices)
                .bind(3, args.outWeights)
                .constants({args.numExperts, args.topK})
                .go(args.tokens);
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
        if (!args.input.valid()) {
            return makeError(ErrorCode::InvalidArgument, "gatherRows input is unset");
        }

        return Dispatch{backend_, stream, ShaderId::GatherRows}
                .bind(0, args.input)
                .bind(1, args.rows)
                .bind(2, args.output)
                .constants({args.count, args.width})
                .go(groupsFor(args.width, kBlock), args.count);
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
        TF_CHECK(requireView(args.rows, u64{args.count} * sizeof(u32), "scatterAddRows rows"));
        TF_CHECK(requireView(args.scales, u64{args.count} * kActivationBytes,
                             "scatterAddRows scales"));
        if (!args.output.valid()) {
            return makeError(ErrorCode::InvalidArgument, "scatterAddRows output is unset");
        }

        return Dispatch{backend_, stream, ShaderId::ScatterAddRows}
                .bind(0, args.input)
                .bind(1, args.rows)
                .bind(2, args.output)
                .bind(3, args.scales)
                .constants({args.count, args.width})
                .go(groupsFor(args.width, kBlock), args.count);
    }

    [[nodiscard]] Status fillZero(Stream& stream, const FillZeroArgs& args) override {
        if (args.count == 0) {
            return {};
        }
        TF_CHECK(requireView(args.output, args.count * kActivationBytes, "fillZero output"));

        return Dispatch{backend_, stream, ShaderId::FillZero}
                .bind(2, args.output)
                .constants({static_cast<u32>(args.count)})
                .go(groupsFor(args.count, kBlock));
    }

    [[nodiscard]] Status moeCombine(Stream& stream, const MoeCombineArgs& args) override {
        if (args.topK == 0 || args.hidden == 0) {
            return makeError(ErrorCode::InvalidArgument, "moeCombine has a zero extent");
        }
        TF_CHECK(requireView(args.expertOutputs,
                             u64{args.topK} * args.hidden * kActivationBytes,
                             "moeCombine expert outputs"));
        TF_CHECK(requireView(args.weights, u64{args.topK} * kActivationBytes,
                             "moeCombine weights"));
        TF_CHECK(requireView(args.output, u64{args.hidden} * kActivationBytes,
                             "moeCombine output"));

        return Dispatch{backend_, stream, ShaderId::MoeCombine}
                .bind(0, args.expertOutputs)
                .bind(1, args.weights)
                .bind(2, args.output)
                .constants({args.topK, args.hidden})
                .go(groupsFor(args.hidden, kBlock));
    }

private:
    D3D12Backend& backend_;
};

}  // namespace

Result<std::unique_ptr<IKernels>> createD3D12Kernels(D3D12Backend& backend) {
    return std::make_unique<D3D12Kernels>(backend);
}

}  // namespace tf::gpu::d3d12
