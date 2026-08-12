# tf-cuda-probe - validates that this machine can actually run the kernels the
# port is about to be built on, and measures the bandwidth figures that drive the
# M6 streaming design and M13 tuning.
#
# Built only when a CUDA toolkit was found. Seeds the preflight checks the CLI,
# server and GUI will share.

add_executable(tf-cuda-probe probe.cu)

target_link_libraries(tf-cuda-probe PRIVATE tf::cuda_options tf::options)

set_target_properties(tf-cuda-probe PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        FOLDER "tools")
