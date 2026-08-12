#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/ArchInfo.h"
#include "tf/core/format/DType.h"

namespace tf::gturbo {

/// Bumped whenever the on-disk layout changes incompatibly. The loader refuses
/// anything it does not recognise rather than misreading it.
///
/// v2 aligns resident tensor offsets. v1 packed them back to back, so a 2-byte
/// tensor such as layer_scalar left the cursor at 2 mod 4 and every packed u32
/// weight after it was misaligned - undefined behaviour on the host and a hard
/// fault on the GPU.
inline constexpr u32 kFormatVersion = 2;

/// Alignment applied to every resident tensor's offset.
///
/// 16 rather than the 4 a u32 load strictly needs: it costs at most 15 bytes
/// per tensor, about 16 KB across the whole index, and leaves room for
/// vectorized loads later without another format change.
inline constexpr u64 kResidentTensorAlignment = 16;

// Canonical names inside a .gturbo directory.
inline constexpr std::string_view kManifestFile = "manifest.json";
inline constexpr std::string_view kReceiptFile = "verified-install.json";
inline constexpr std::string_view kResidentFile = "model_weights.bin";
inline constexpr std::string_view kExpertsDir = "packed_experts";
inline constexpr std::string_view kExpertLayoutFile = "layout.json";
inline constexpr std::string_view kTokenizerDir = "tokenizer";

/// One tensor inside model_weights.bin.
struct ResidentTensor {
    /// Canonical name with the checkpoint's `language_model.model.` prefix
    /// stripped, e.g. "layers.0.self_attn.q_proj.weight".
    std::string name;
    DType dtype = DType::BF16;
    std::vector<u64> shape;
    ByteRange range;

    /// Bytes implied by shape and dtype, which must equal range.length.
    [[nodiscard]] u64 expectedBytesForShape() const noexcept {
        u64 count = 1;
        for (const u64 dim : shape) {
            count *= dim;
        }
        return count * byteWidth(dtype);
    }

    [[nodiscard]] std::string shapeString() const;

    friend bool operator==(const ResidentTensor&, const ResidentTensor&) = default;
};

/// Index of the resident weight file: everything that stays in memory for the
/// whole session, as opposed to the routed experts that stream.
class ResidentIndex {
public:
    void add(ResidentTensor tensor);

    [[nodiscard]] const std::vector<ResidentTensor>& tensors() const noexcept {
        return tensors_;
    }

    [[nodiscard]] const ResidentTensor* find(std::string_view name) const;
    [[nodiscard]] Result<const ResidentTensor*> require(std::string_view name) const;

    /// End of the highest range, i.e. the size the resident file must have.
    [[nodiscard]] u64 totalBytes() const noexcept { return totalBytes_; }

    /// Verifies no two tensors overlap and that the ranges tile the file without
    /// leaving unreachable gaps.
    [[nodiscard]] Status validate() const;

private:
    std::vector<ResidentTensor> tensors_;
    u64 totalBytes_ = 0;
};

/// One component of a routed expert, at a fixed offset within every expert blob.
struct ExpertComponent {
    /// "gate.weight", "gate.scales", "gate.biases", "up.*", "down.*".
    std::string role;
    DType dtype = DType::U32;
    std::vector<u64> shape;
    /// Offset relative to the start of the expert blob.
    u64 offset = 0;
    u64 size = 0;

    friend bool operator==(const ExpertComponent&, const ExpertComponent&) = default;
};

/// Layout of the packed expert files.
///
/// Diverges deliberately from upstream, which writes an explicit offset/size
/// entry per expert. Every expert here is byte-identical in shape, so the
/// layout is fully described by a stride plus one shared component table:
/// 30 x 128 = 3840 entries collapse from megabytes of JSON to a few hundred
/// lines, and the loader becomes arithmetic instead of a lookup.
struct ExpertLayout {
    u64 numLayers = 0;
    u64 expertsPerLayer = 0;
    /// Unpadded bytes of one expert blob.
    u64 blobBytes = 0;
    /// Distance between consecutive expert blobs: blobBytes rounded up to
    /// `alignment`.
    u64 stride = 0;
    /// Stride alignment. 4 KiB is the minimum FILE_FLAG_NO_BUFFERING accepts;
    /// larger values waste a quarter of the file at this blob size.
    u64 alignment = align::kSector;
    /// Per-layer file names, indexed by layer.
    std::vector<std::string> layerFiles;
    /// Shared component table, identical for every expert.
    std::vector<ExpertComponent> components;

    [[nodiscard]] static std::string layerFileName(u64 layer);

    /// Byte range of an expert blob within its layer file.
    [[nodiscard]] Result<ByteRange> expertRange(u64 layer, u64 expert) const;

    /// Byte range of one component of one expert, within its layer file.
    [[nodiscard]] Result<ByteRange> componentRange(u64 layer, u64 expert,
                                                   std::string_view role) const;

    /// Size each layer file must have.
    [[nodiscard]] u64 layerFileBytes() const { return stride * expertsPerLayer; }

    [[nodiscard]] u64 totalBytes() const { return layerFileBytes() * numLayers; }

    /// True when stride and alignment permit unbuffered reads on a volume with
    /// the given sector size.
    [[nodiscard]] bool supportsUnbufferedReads(u64 sectorSize) const;

    [[nodiscard]] Status validate() const;

    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] static Result<ExpertLayout> fromJson(std::string_view json);
};

/// SHA-256 digest plus size for one file in the install.
struct FileDigest {
    std::string path;  ///< relative to the .gturbo root, forward slashes
    u64 size = 0;
    std::string sha256;  ///< lowercase hex

    friend bool operator==(const FileDigest&, const FileDigest&) = default;
};

/// Provenance of the install: which upstream checkpoint it was built from.
struct SourceInfo {
    std::string repoId;
    std::string revision;
    std::string toolVersion;
};

/// The manifest. Its presence is what distinguishes a complete install from a
/// partial one; the loader treats a .gturbo directory without it as unusable.
struct Manifest {
    u32 formatVersion = kFormatVersion;
    ArchInfo arch;
    SourceInfo source;
    ResidentIndex resident;
    ExpertLayout experts;
    std::vector<FileDigest> files;

    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] static Result<Manifest> fromJson(std::string_view json);

    [[nodiscard]] Status validate() const;
};

/// Reads and validates the manifest of an installed .gturbo directory.
[[nodiscard]] Result<Manifest> readManifest(const std::filesystem::path& installDir);

/// Confirms every file named in the manifest exists with the recorded size and
/// SHA-256. This is the `--verify-install` path.
[[nodiscard]] Status verifyInstall(const std::filesystem::path& installDir,
                                   const Manifest& manifest);

}  // namespace tf::gturbo
