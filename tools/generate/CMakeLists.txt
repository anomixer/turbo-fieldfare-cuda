# tf-generate drives the decode path against a real install.
#
# The M7 milestone tool: proves the forward pass runs end to end and reports
# throughput and expert cache behaviour. Superseded by the real CLI in M9, once
# the tokenizer from M2 makes text input possible.

add_executable(tf-generate main.cpp)

target_link_libraries(tf-generate PRIVATE tf::runtime tf::gpu tf::core tf::options)

set_target_properties(tf-generate PROPERTIES FOLDER "tools")
