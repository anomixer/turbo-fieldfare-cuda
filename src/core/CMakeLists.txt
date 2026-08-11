# tfcore - format, I/O, JSON, tokenizer and chat templating.
#
# Deliberately free of CUDA and UI dependencies: the test suite links this
# headless, and the D3D12 backend added in M12 must not require changes here.

add_library(tfcore STATIC
        src/base/Error.cpp
        src/format/ArchInfo.cpp
        src/format/GTurbo.cpp
        src/format/Safetensors.cpp
        src/io/File.cpp
        src/io/Sha256.cpp
        src/io/Volume.cpp
        src/json/Json.cpp
        src/net/Http.cpp
        src/tokenizer/AssistantDecoder.cpp
        src/tokenizer/ChatTemplate.cpp
        src/tokenizer/Tokenizer.cpp)

add_library(tf::core ALIAS tfcore)

target_include_directories(tfcore PUBLIC include)

# winhttp is in-box on every supported Windows, so this adds no redistributable.
target_link_libraries(tfcore
        PUBLIC tf::options
        PRIVATE winhttp)

set_target_properties(tfcore PROPERTIES FOLDER "libs")
