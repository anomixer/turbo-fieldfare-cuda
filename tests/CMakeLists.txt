include(FetchContent)

FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(Catch2)

add_executable(tf_tests
        cli/ArgsTests.cpp
        core/ArchInfoTests.cpp
        core/AssistantDecoderTests.cpp
        core/BaseTests.cpp
        core/CheckpointTests.cpp
        core/FileTests.cpp
        core/JsonTests.cpp
        core/SafetensorsTests.cpp
        core/TokenizerTests.cpp
        gpu/AttentionKernelTests.cpp
        gpu/BackendTests.cpp
        gpu/KernelTests.cpp
        reference/FloatTests.cpp
        reference/OpsTests.cpp
        reference/RealWeightsTests.cpp
        repack/InstallerTests.cpp
        repack/RemoteTests.cpp
        repack/RepackPlanTests.cpp
        repack/RepackerTests.cpp
        runtime/ExpertStreamerTests.cpp
        runtime/LayerReferenceTests.cpp
        runtime/ModelCatalogTests.cpp
        runtime/PrefillTests.cpp
        runtime/PromptCacheTests.cpp
        runtime/ResidencyTests.cpp
        runtime/SamplerTests.cpp
        server/ApiTests.cpp
        server/HttpTests.cpp
        svc/ProtocolTests.cpp
        ../src/cli/Args.cpp)

target_link_libraries(tf_tests
        PRIVATE tf::core
                tf::gpu
                tf::reference
                tf::repack
                tf::runtime
                tf::server
                tf::svc
                tf::options
                Catch2::Catch2WithMain)

target_include_directories(tf_tests PRIVATE ../src/cli gpu)

set_target_properties(tf_tests PROPERTIES FOLDER "tests")

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(tf_tests)
