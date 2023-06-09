
# Get LCC Compiler version and target

if(NOT CMAKE_C_COMPILER_VERSION_INTERNAL)

    set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
    set(CMAKE_C_COMPILER_TARGET "")

    set(LCC_EXP "^([^:]+):([^:]+):([^:]+):([^:]+).*")
    set(LCC_FULL "")

    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -v ERROR_VARIABLE LCC_FULL
    )
    string(REGEX REPLACE ${LCC_EXP} "\\1 \\2 \\3" CMAKE_C_COMPILER_VERSION_INTERNAL ${LCC_FULL})
    string(REGEX REPLACE ${LCC_EXP} "\\4"         CMAKE_C_COMPILER_TARGET           ${LCC_FULL})

    unset(FULL_VER)
    unset(LCC_REG)
endif()

# LCC Compiler Options

if (CMAKE_C_COMPILER_VERSION_INTERNAL MATCHES "lcc")
    message(STATUS "MCST Elbrus Compiler detected")

    option(LCC_FLAGS "C flags for e2k"   OFF)
    option(LXX_FLAGS "C++ flags for e2k" OFF)

    set(LCC_DEFAULT "-O4 -mcpu=native -ffast")
    set(LCC_NOINTEL "-mno-mmx -mno-sse -mno-avx -mno-f16c -mno-fma")
    set(LXX_DEFAULT "-O3 -mcpu=native")

    if (NOT LCC_FLAGS)
        message(NOTICE "\n"
        "  VLIW may need finer compiler tuning (-fprofile-use for example)\n"
        "\n"
        "  use -DLCC_FLAGS for change c flags    [ default: ${LCC_DEFAULT} ]\n"
        "  use -DLXX_FLAGS for change c++ flags  [ default: ${LXX_DEFAULT} ]\n"
        )
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LCC_DEFAULT} ${LCC_NOINTEL}")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LCC_FLAGS} ${LCC_NOINTEL}")
    endif()

    if (NOT LXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LXX_DEFAULT}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LXX_FLAGS}")
    endif()
    unset(LCC_DEFAULT)
    unset(LXX_DEFAULT)
endif()
