# tf-decode-client - a terminal front end for the decode service.
#
# Deliberately links only tf::svc, not tf::runtime or tf::gpu: it proves the
# service boundary really is one, since a front end that needed the runtime
# would not be a front end.

add_executable(tf-decode-client main.cpp)

target_link_libraries(tf-decode-client PRIVATE tf::svc tf::options)

set_target_properties(tf-decode-client PROPERTIES FOLDER "tools")