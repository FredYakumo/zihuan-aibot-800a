cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0167 NEW)
set(CMAKE_CXX_STANDARD 23)
project(ZiHuanAIBot LANGUAGES CXX)

# 设置默认构建类型为 Debug（如果没有通过命令行指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
Message(Build_type: ${CMAKE_BUILD_TYPE})


set(MIRAICP_TARGET_NAME AIBot800a)
set(AIBOT_TARGET_NAME AIBot800b)
set(MIRAICP_MSG_DB_TARGET_NAME AIBot800a_msg_db)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(cpr REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(spdlog REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(general-wheel-cpp REQUIRED)



set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/json/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/)
include_directories(${AIBOT_TARGET_NAME} PRIVATE ${TOKENZIER_CPP_PATH}/include)
include_directories(/opt/homebrew/include/ncnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/base64/include)
if(APPLE)
    include_directories(/usr/local/include)
endif()



enable_testing()

# 如果插件源路径未定义，则设为空
if(NOT DEFINED PLUGIN_SOURCE_PATH)
    set(PLUGIN_SOURCE_PATH "")
endif()

# 收集源码文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network NN_SRC_FILES)


# set(ALL_SOURCES
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp  # MiraiCP 源文件
#     ${SRC_FILES}
#     ${PLUGIN_SOURCE_PATH}
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
# )

set(BIN_SOURCES
    ${SRC_FILES}
    ${NN_SRC_FILES}
    ${PLUGIN_SOURCE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)

# 打印所有源文件（方便调试）
foreach(file IN LISTS SRC_FILES)
    message(STATUS "Source file: ${file}")
endforeach()
foreach(file IN LISTS NN_SRC_FILES)
    message(STATUS "Source file: ${file}")
endforeach()

# 创建共享库
# add_library(${MIRAICP_TARGET_NAME} SHARED ${ALL_SOURCES})
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE cpr::cpr)
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE fmt::fmt)


# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary


# 创建可执行文件
add_executable(${AIBOT_TARGET_NAME} ${BIN_SOURCES})
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE cpr::cpr)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE fmt::fmt)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE Boost::filesystem)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE ncnn)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE tokenizers_cpp)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE onnxruntime::onnxruntime)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE general-wheel-cpp::general-wheel-cpp)


if(APPLE)
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE /opt/homebrew/lib/libmysqlcppconnx.dylib)
endif()

# 单元测试
include(CTest)
add_executable(unit_test
    ${MARKDOWN_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bot_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/nn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/text_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/global_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)
target_link_libraries(unit_test cpr::cpr GTest::GTest GTest::Main fmt::fmt yaml-cpp::yaml-cpp Boost::filesystem tokenizers_cpp ncnn onnxruntime::onnxruntime 
general-wheel-cpp::general-wheel-cpp)

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