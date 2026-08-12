# tf-server - the OpenAI-compatible endpoint.
#
# Http and Api are separate from main so both can be linked into the test
# binary: the wire format is the compatibility surface, and it deserves tests
# that need neither a socket nor a GPU.

add_library(tfserver STATIC
        Api.cpp
        Engine.cpp
        Http.cpp)

add_library(tf::server ALIAS tfserver)

target_include_directories(tfserver PUBLIC .)

# ws2_32 is in-box; no redistributable.
target_link_libraries(tfserver
        PUBLIC tf::core tf::gpu tf::runtime
        PRIVATE tf::options ws2_32)

set_target_properties(tfserver PROPERTIES FOLDER "libs")

add_executable(tf-server main.cpp)

target_link_libraries(tf-server PRIVATE tf::server tf::options)

set_target_properties(tf-server PROPERTIES FOLDER "tools")