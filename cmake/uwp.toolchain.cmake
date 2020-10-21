# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME WindowsStore)

if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0)
endif()

if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp"
    "#ifdef WINAPI_FAMILY\n"
    "#undef WINAPI_FAMILY\n"
    "#define WINAPI_FAMILY WINAPI_FAMILY_DESKTOP_APP\n"
    "#endif\n")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FI\"${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp\"")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /FI\"${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp\"")

# UWP setting for package isolation
# set(CMAKE_VS_GLOBALS "AppContainerApplication=true")
set(CMAKE_VS_GLOBALS "WindowsTargetPlatformMinVersion=${CMAKE_SYSTEM_VERSION}")
