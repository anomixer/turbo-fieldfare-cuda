#include "tf/core/io/Volume.h"

#include <windows.h>
// winioctl.h must follow windows.h
#include <winioctl.h>

#include <system_error>

#include "tf/core/io/File.h"

namespace tf::io {
namespace {

/// Asks the storage driver whether the device has a seek penalty. That is the
/// documented way to distinguish an SSD from a spinning disk on Windows;
/// there is no direct "is this an SSD" query.
///
/// Requires opening the volume itself, which needs no special privilege as long
/// as no access rights are requested.
[[nodiscard]] StorageKind querySeekPenalty(wchar_t driveLetter) {
    wchar_t volumePath[] = L"\\\\.\\X:";
    volumePath[4] = driveLetter;

    const HANDLE volume = ::CreateFileW(volumePath, 0,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                        OPEN_EXISTING, 0, nullptr);
    if (volume == INVALID_HANDLE_VALUE) {
        return StorageKind::Unknown;
    }

    STORAGE_PROPERTY_QUERY query{};
    query.PropertyId = StorageDeviceSeekPenaltyProperty;
    query.QueryType = PropertyStandardQuery;

    DEVICE_SEEK_PENALTY_DESCRIPTOR descriptor{};
    DWORD returned = 0;
    const BOOL ok = ::DeviceIoControl(volume, IOCTL_STORAGE_QUERY_PROPERTY, &query,
                                      sizeof(query), &descriptor, sizeof(descriptor),
                                      &returned, nullptr);
    ::CloseHandle(volume);

    if (ok == 0 || returned < sizeof(descriptor)) {
        // Network shares and some virtual disks decline the query. Reporting
        // Unknown lets callers warn without asserting something false.
        return StorageKind::Unknown;
    }
    return descriptor.IncursSeekPenalty ? StorageKind::Rotational : StorageKind::SolidState;
}

}  // namespace

std::string_view toString(StorageKind kind) noexcept {
    switch (kind) {
        case StorageKind::SolidState:  return "SSD";
        case StorageKind::Rotational:  return "HDD";
        case StorageKind::Unknown:     return "unknown";
    }
    return "unknown";
}

Result<u64> totalPhysicalMemory() {
    ULONGLONG kilobytes = 0;
    if (::GetPhysicallyInstalledSystemMemory(&kilobytes) == 0) {
        // Falls back to what the OS reports as usable, which is slightly below
        // the installed total but close enough for a caching decision.
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (::GlobalMemoryStatusEx(&status) == 0) {
            return std::unexpected(lastWin32Error("querying physical memory"));
        }
        return static_cast<u64>(status.ullTotalPhys);
    }
    return static_cast<u64>(kilobytes) * 1024;
}

Result<VolumeInfo> queryVolume(const std::filesystem::path& path) {
    // Walk up to something that exists: the install directory is routinely
    // absent when this runs.
    std::filesystem::path probe = std::filesystem::absolute(path);
    std::error_code ec;
    while (!probe.empty() && !std::filesystem::exists(probe, ec)) {
        const std::filesystem::path parent = probe.parent_path();
        if (parent == probe) {
            break;
        }
        probe = parent;
    }

    const std::filesystem::path root = probe.root_path();

    VolumeInfo info;
    info.rootPath = root.string();

    ULARGE_INTEGER available{};
    ULARGE_INTEGER total{};
    ULARGE_INTEGER free{};
    if (::GetDiskFreeSpaceExW(probe.c_str(), &available, &total, &free) == 0) {
        return std::unexpected(
                lastWin32Error(std::format("querying free space on {}", root.string())));
    }
    info.freeBytes = available.QuadPart;
    info.totalBytes = total.QuadPart;

    if (const auto sector = File::sectorSize(probe); sector.has_value()) {
        info.sectorSize = *sector;
    }

    const std::wstring rootWide = root.wstring();
    if (rootWide.size() >= 2 && rootWide[1] == L':') {
        info.storage = querySeekPenalty(rootWide[0]);
    }

    return info;
}

}  // namespace tf::io
