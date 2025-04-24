cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0167 NEW)
set(CMAKE_CXX_STANDARD 23)
project(ZiHuanAIBot LANGUAGES CXX)

# 设置默认构建类型为 Debug（如果没有通过命令行指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
Message(Build_type: ${CMAKE_BUILD_TYPE})

# 定义目标名称
set(MIRAICP_TARGET_NAME AIBot800a)
set(AIBOT_TARGET_NAME AIBot800b)
set(MIRAICP_MSG_DB_TARGET_NAME AIBot800a_msg_db)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找所需的第三方包
find_package(cpr REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(spdlog REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)

# 设置头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/json/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/)

enable_testing()

# 如果插件源路径未定义，则设为空
if(NOT DEFINED PLUGIN_SOURCE_PATH)
    set(PLUGIN_SOURCE_PATH "")
endif()

# 收集源码文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)

# set(ALL_SOURCES
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp  # MiraiCP 源文件
#     ${SRC_FILES}
#     ${PLUGIN_SOURCE_PATH}
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
# )

set(BIN_SOURCES
    ${SRC_FILES}
    ${PLUGIN_SOURCE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)

# 打印所有源文件（方便调试）
foreach(file IN LISTS ALL_SOURCES)
    message(STATUS "Source file: ${file}")
endforeach()

# 创建共享库
# add_library(${MIRAICP_TARGET_NAME} SHARED ${ALL_SOURCES})
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE cpr::cpr)
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE fmt::fmt)

# 创建可执行文件
add_executable(${AIBOT_TARGET_NAME} ${BIN_SOURCES})
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE cpr::cpr)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE fmt::fmt)

# 单元测试
include(CTest)
add_executable(unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bot_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)
target_link_libraries(unit_test GTest::GTest GTest::Main fmt::fmt)
add_test(NAME unit_test COMMAND unit_test)

# 添加预编译宏定义
# target_compile_definitions(${MIRAICP_TARGET_NAME} PUBLIC JSON_MultipleHeaders=ON MIRAICP_LIB_SDK)

# 安装指令
# install(TARGETS ${MIRAICP_TARGET_NAME}
#     DESTINATION ${CMAKE_INSTALL_PREFIX}
#     RENAME libAIBot800a.so
# )

# if(APPLE)
#     # macOS 可能需要额外的配置，比如设置 rpath
#     set_target_properties(${MIRAICP_TARGET_NAME} PROPERTIES INSTALL_NAME_DIR "@rpath")
# endif()

# 根据构建类型设置 spdlog 的日志级别
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()
