# 最低cmake版本
cmake_minimum_required(VERSION 3.14)
# C++ 标准：17+
set(CMAKE_CXX_STANDARD 17)

project(MiraiAIBot LANGUAGES CXX)
set(MIRAICP_TARGET_NAME MiraiCPPlugin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/json/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/)

# 添加动态库编译目标
add_library(
        MiraiCPPlugin
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp # MiraiCP 源文件
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
        ${PLUGIN_SOURCE_PATH}
)

target_compile_definitions(${MIRAICP_TARGET_NAME} PUBLIC JSON_MultipleHeaders=ON MIRAICP_LIB_SDK)