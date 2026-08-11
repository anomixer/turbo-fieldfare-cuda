# tf-decode - the process that owns the model.
#
# Protocol and Pipe are a library so the GUI and the tests link them without
# the service's main. The GUI in particular must not pull in tf::runtime: the
# whole point of the split is that it never loads a model.

add_library(tfsvc STATIC
        Pipe.cpp
        Protocol.cpp)

add_library(tf::svc ALIAS tfsvc)

target_include_directories(tfsvc PUBLIC .)

# tf::runtime is here only for SamplingParams::validate(). A static link pulls
# just that object, so a front end gets no GPU dependency - which is what makes
# the GUI able to link this without a CUDA toolkit.
target_link_libraries(tfsvc
        PUBLIC tf::core tf::runtime
        PRIVATE tf::options)

set_target_properties(tfsvc PROPERTIES FOLDER "libs")

add_executable(tf-decode main.cpp)

target_link_libraries(tf-decode PRIVATE tf::svc tf::gpu tf::runtime tf::options)

set_target_properties(tf-decode PROPERTIES FOLDER "tools")