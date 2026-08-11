# tfruntime - residency planning, the KV cache and expert streaming.
#
# Free of CUDA headers: everything GPU-facing goes through tf::gpu, so the
# D3D12 backend planned for M12 requires no change here.

add_library(tfruntime STATIC
        src/ExpertStreamer.cpp
        src/ForwardRunner.cpp
        src/Generator.cpp
        src/KVCache.cpp
        src/Model.cpp
        src/ModelCatalog.cpp
        src/Residency.cpp
        src/Sampler.cpp
        src/StopMatcher.cpp)

add_library(tf::runtime ALIAS tfruntime)

target_include_directories(tfruntime PUBLIC include)

target_link_libraries(tfruntime
        PUBLIC tf::core tf::gpu
        PRIVATE tf::options)

set_target_properties(tfruntime PROPERTIES FOLDER "libs")
