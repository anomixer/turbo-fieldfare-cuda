# Compiles the D3D12 compute shaders to DXIL at build time.
#
# Build time rather than runtime: it removes the dxcompiler dependency from the
# shipped binary, there is no shader cache to go stale, and a shader that does
# not compile fails the build rather than the first inference.
#
# Each entry point becomes a header holding a byte array, which
# D3D12Shaders.cpp includes and maps to a ShaderId.

find_program(TF_DXC dxc
        HINTS
            "$ENV{WindowsSdkVerBinPath}x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
        DOC "The DirectX shader compiler, from the Windows SDK")

# Adds a command compiling one entry point of `source` into `outputDir`.
#
# The variable name matches the entry point, so the mapping in
# D3D12Shaders.cpp needs no generated glue.
function(tf_compile_hlsl source entry outputDir outputsVariable)
    set(header "${outputDir}/${entry}.h")
    add_custom_command(
            OUTPUT "${header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${outputDir}"
            COMMAND "${TF_DXC}"
                    -T cs_6_6
                    -E ${entry}
                    -Fh "${header}"
                    -Vn g_${entry}
                    # Warnings as errors: a shader that compiles with a warning
                    # about an uninitialized value produces silently wrong
                    # numbers, which is the failure mode this project has spent
                    # the most time on.
                    -WX
                    -O3
                    "${source}"
            DEPENDS "${source}"
            COMMENT "dxc ${entry}"
            VERBATIM)
    set(${outputsVariable} ${${outputsVariable}} "${header}" PARENT_SCOPE)
endfunction()
