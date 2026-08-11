# tfreference - scalar CPU implementations of every model operation.
#
# The backbone of the kernel test strategy: each CUDA kernel in M5 and M8 is
# checked against the corresponding function here before anything downstream
# depends on it. Without this, a wrong kernel surfaces as garbage tokens 40
# layers later instead of as a failing assertion.
#
# Not linked into the shipping runtime. Correctness and legibility over speed:
# these accumulate in double and do no blocking or vectorization, so that a
# disagreement points at the kernel rather than at the reference.

add_library(tfreference STATIC
        src/Ops.cpp)

add_library(tf::reference ALIAS tfreference)

target_include_directories(tfreference PUBLIC include)

target_link_libraries(tfreference
        PUBLIC tf::core
        PRIVATE tf::options)

set_target_properties(tfreference PROPERTIES FOLDER "libs")
