# tfrepack - builds a .gturbo install from a source checkpoint.
#
# M1a covers repacking from a locally downloaded checkpoint. M1b adds the
# WinHTTP range-request front end so the same planner drives a streaming
# install that never materializes the full 15 GB source.

add_library(tfrepack STATIC
        src/RepackPlan.cpp
        src/Installer.cpp
        src/Remote.cpp
        src/Repacker.cpp)

add_library(tf::repack ALIAS tfrepack)

target_include_directories(tfrepack PUBLIC include)

target_link_libraries(tfrepack
        PUBLIC tf::core
        PRIVATE tf::options)

set_target_properties(tfrepack PROPERTIES FOLDER "libs")

add_executable(tf-repack src/main.cpp)

target_link_libraries(tf-repack PRIVATE tf::repack tf::options)

set_target_properties(tf-repack PROPERTIES FOLDER "tools")
