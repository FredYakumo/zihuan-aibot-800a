cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(MiraiAIBot LANGUAGES CXX)
set(MIRAICP_TARGET_NAME AIBot800a)
set(MIRAICP_MSG_DB_TARGET_NAME AIBot800a_msg_db)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(cpr REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/json/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/)

enable_testing()

find_package(Catch2 REQUIRED)

# 添加动态库编译目标
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)

set(ALL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp # MiraiCP 源文件
    ${SRC_FILES}
    ${PLUGIN_SOURCE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)

foreach(file IN LISTS ALL_SOURCES)
    message(STATUS "Source file: ${file}")
endforeach()


# 创建共享库
add_library(${MIRAICP_TARGET_NAME} SHARED ${ALL_SOURCES})

target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE cpr::cpr)
target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE fmt::fmt)
target_link_libraries(utils_test Catch2::Catch2WithMain)
include(CTest)
include(Catch)


target_compile_definitions(${MIRAICP_TARGET_NAME} PUBLIC JSON_MultipleHeaders=ON MIRAICP_LIB_SDK)

add_test(NAME unit_test COMMAND unit_test)



# 安装指令
install(TARGETS ${MIRAICP_TARGET_NAME}
    # DESTINATION ${CMAKE_INSTALL_PREFIX}/aibot_800a
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    RENAME libAIBot800a.so
)

if(APPLE)
    # 对于 macOS，这里可能需要额外的配置，比如设置安装名称工具（install_name_tool）
    set_target_properties(${MIRAICP_TARGET_NAME} PROPERTIES INSTALL_NAME_DIR "@rpath")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()