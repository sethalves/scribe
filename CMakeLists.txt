cmake_minimum_required(VERSION 3.7)

# squelch the Policy CMP0074 warning without requiring an update to cmake 3.12.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    cmake_policy(SET CMP0074 NEW)
endif()

project(scribe)
include("cmake/init.cmake")
include("cmake/compiler.cmake")


file(GLOB TARGET_SRCS src/*)
add_executable(scribe ${TARGET_SRCS})
if (WIN32)
    set_property(TARGET scribe APPEND_STRING PROPERTY LINK_FLAGS_DEBUG "/OPT:NOREF /OPT:NOICF")
endif()
install(TARGETS scribe RUNTIME DESTINATION tools)
