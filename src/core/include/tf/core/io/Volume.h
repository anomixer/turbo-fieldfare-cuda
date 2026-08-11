#pragma once

#include <filesystem>
#include <string>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::io {

enum class StorageKind {
    /// No seek penalty reported: NVMe or SATA SSD.
    SolidState,
    /// Seek penalty reported: a spinning disk.
    Rotational,
    /// Network share, virtual disk, or a driver that declines to answer.
    Unknown,
};

[[nodiscard]] std::string_view toString(StorageKind kind) noexcept;

struct VolumeInfo {
    std::string rootPath;
    StorageKind storage = StorageKind::Unknown;
    u64 freeBytes = 0;
    u64 totalBytes = 0;
    /// Physical sector size, which sets the alignment unbuffered reads need.
    u64 sectorSize = 0;
};

/// Describes the volume holding `path`, walking up to the nearest existing
/// ancestor so it works for a directory that has not been created yet.
///
/// The storage kind matters: a .gturbo install on a rotational disk defeats the
/// entire streaming design, since expert fetches become seek-bound. Worth
/// warning about rather than silently running at a tenth of the speed.
[[nodiscard]] Result<VolumeInfo> queryVolume(const std::filesystem::path& path);

/// Installed physical memory. Governs whether the operating system can hold the
/// whole expert set in its page cache, which is the difference between expert
/// reads running at RAM speed and at SSD speed.
[[nodiscard]] Result<u64> totalPhysicalMemory();

}  // namespace tf::io
