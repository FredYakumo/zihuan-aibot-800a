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
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)

set(ALL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp # MiraiCP 源文件
    ${SRC_FILES}
    ${PLUGIN_SOURCE_PATH}
)

foreach(file IN LISTS ALL_SOURCES)
    message(STATUS "Source file: ${file}")
endforeach()

# 创建共享库
add_library(MiraiCPPlugin SHARED ${ALL_SOURCES})

target_compile_definitions(${MIRAICP_TARGET_NAME} PUBLIC JSON_MultipleHeaders=ON MIRAICP_LIB_SDK)

# 安装指令
install(TARGETS MiraiCPPlugin
    DESTINATION ${CMAKE_INSTALL_PREFIX}/mcl_bot/aibot_800a
    RENAME libAIBot800a.so
)

if(APPLE)
    # 对于 macOS，这里可能需要额外的配置，比如设置安装名称工具（install_name_tool）
    set_target_properties(MiraiCPPlugin PROPERTIES INSTALL_NAME_DIR "@rpath")
endif()