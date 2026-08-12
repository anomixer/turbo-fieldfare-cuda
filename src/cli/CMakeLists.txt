# tf-cli - the command-line front end.
#
# Args is a separate translation unit from main so the parser can be tested
# directly; argument handling is where a CLI accumulates quiet bugs.

add_executable(tf-cli
        Args.cpp
        main.cpp)

target_include_directories(tf-cli PRIVATE .)

target_link_libraries(tf-cli
        PRIVATE tf::core tf::gpu tf::runtime tf::options)

set_target_properties(tf-cli PROPERTIES FOLDER "tools")