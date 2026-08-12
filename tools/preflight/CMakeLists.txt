# tf-preflight reports what the engine would find on this machine, and which
# model variant suits it. The first thing to run when a user says it does not
# work.
#
# Lives here rather than under src/gpu because it needs both the GPU checks and
# the runtime's model catalog, and tfruntime depends on tfgpu.

add_executable(tf-preflight main.cpp)

target_link_libraries(tf-preflight PRIVATE tf::gpu tf::runtime tf::options)

set_target_properties(tf-preflight PROPERTIES FOLDER "tools")
