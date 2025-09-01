cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0167 NEW)
set(CMAKE_CXX_STANDARD 23)
project(ZiHuanAIBot LANGUAGES CXX)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.5")
endif()

# Set default build type to Debug (if not specified via command line)
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
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)
# find_package(onnxruntime REQUIRED)  # Moved to conditional branch
find_package(general-wheel-cpp REQUIRED)


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  FetchContent_MakeAvailable(googletest)
else()
  find_package(GTest REQUIRED)
endif()


set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/json/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/3rd_include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/)
include_directories(${AIBOT_TARGET_NAME} PRIVATE ${TOKENZIER_CPP_PATH}/include)
# include_directories(/opt/homebrew/include/ncnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/base64/include)
if(APPLE)
    include_directories(/usr/local/include)
endif()



enable_testing()

# Set plugin source path to empty if not defined
if(NOT DEFINED PLUGIN_SOURCE_PATH)
    set(PLUGIN_SOURCE_PATH "")
endif()

# Collect source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/agent AGENT_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network NN_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/text_model NN_TEXT_MODEL_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/text_model/lac NN_TEXT_MODEL_LAC_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/nlp NN_NLP_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/llm NN_LLM_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/vec_db VEC_DB_SRC_FILES)


# set(ALL_SOURCES
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/MiraiCP-template/single_include/MiraiCP/MiraiCP.cpp  # MiraiCP source file
#     ${SRC_FILES}
#     ${PLUGIN_SOURCE_PATH}
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
# )

set(BIN_SOURCES
    ${SRC_FILES}
    ${AGENT_SRC_FILES}
    ${NN_SRC_FILES}
    ${NN_TEXT_MODEL_SRC_FILES}
    ${NN_TEXT_MODEL_LAC_SRC_FILES}
    ${NN_NLP_SRC_FILES}
    ${NN_LLM_SRC_FILES}
    ${VEC_DB_SRC_FILES}
    ${PLUGIN_SOURCE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/database.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/constants.cpp
)

# print all source files (for debugging)
foreach(file IN LISTS SRC_FILES)
    message(STATUS "Source file: ${file}")
endforeach()
foreach(file IN LISTS AGENT_SRC_FILES)
    message(STATUS "Agent Source file: ${file}")
endforeach()
foreach(file IN LISTS NN_SRC_FILES)
    message(STATUS "NN Source file: ${file}")
endforeach()
foreach(file IN LISTS NN_TEXT_MODEL_SRC_FILES)
    message(STATUS "NN Text Model Source file: ${file}")
endforeach()
foreach(file IN LISTS NN_TEXT_MODEL_LAC_SRC_FILES)
    message(STATUS "NN Text Model LAC Source file: ${file}")
endforeach()
foreach(file IN LISTS VEC_DB_SRC_FILES)
    message(STATUS "Vec DB Source file: ${file}")
endforeach()
foreach(file IN LISTS NN_LLM_SRC_FILES)
    message(STATUS "NN LLM Source file: ${file}")
endforeach()

# Create shared library
# add_library(${MIRAICP_TARGET_NAME} SHARED ${ALL_SOURCES})
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE cpr::cpr)
# target_link_libraries(${MIRAICP_TARGET_NAME} PRIVATE fmt::fmt)


# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary


# Create executable
add_executable(${AIBOT_TARGET_NAME} ${BIN_SOURCES})
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE cpr::cpr)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE fmt::fmt)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE Boost::filesystem)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE ncnn)
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE tokenizers_cpp)
if(NOT use_libtorch)
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE onnxruntime::onnxruntime)
endif()
if(use_libtorch)
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE torch)
endif()
target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE general-wheel-cpp::general-wheel-cpp)


if(APPLE)
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE /opt/homebrew/lib/libmysqlcppconnx.dylib)
elseif(LINUX)
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE /usr/local/lib64/libmysqlcppconnx.so)
endif()


# Unit tests
include(CTest)
add_executable(unit_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/nn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/model_set.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/text_model/text_embedding_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_network/text_model/tokenizer_wrapper.cpp
    # Include LAC sources to satisfy tests referencing lac:: symbols
    ${NN_TEXT_MODEL_LAC_SRC_FILES}
    # Include NLP sources (Dependency Parser, etc.) required by action_agent and model_set
    ${NN_NLP_SRC_FILES}
    # Include agent sources
    ${AGENT_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vec_db/weaviate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/global_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/database.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/msg_prop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bot_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bot_cmd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/easywsclient/easywsclient.cpp
)

# Allow users to specify libtorch path directly via use_libtorch (string), empty means no libtorch
set(use_libtorch "" CACHE PATH "Path to libtorch directory; leave empty to disable libtorch")
# Allow users to specify onnxruntime path via use_ort (string), empty means use default search
set(use_ort "" CACHE PATH "Path to onnxruntime directory; leave empty to use system default")
# Allow users to specify paddle inference path for LAC
set(paddle_inference_path "" CACHE PATH "Path to paddle inference directory for LAC")

# If use_libtorch is specified (i.e., path is not empty), include libtorch, otherwise include onnxruntime
if(use_libtorch)
    set(Torch_DIR "${use_libtorch}/share/cmake/Torch")
    find_package(Torch REQUIRED PATHS ${use_libtorch})
    message(STATUS "Using libtorch from: ${use_libtorch}")
else()
    if(use_ort)
        set(onnxruntime_DIR "${use_ort}")
        find_package(onnxruntime REQUIRED PATHS ${use_ort})
        message(STATUS "Using onnxruntime from: ${use_ort}")
    else()
        find_package(onnxruntime REQUIRED)
    endif()
endif()

# Configure Paddle Inference Library for LAC
if(paddle_inference_path)

    # Check if the path exists
    if(NOT EXISTS ${paddle_inference_path})
        message(FATAL_ERROR "Paddle Inference path does not exist: ${paddle_inference_path}")
    endif()

    message(STATUS "Using Paddle Inference from: ${paddle_inference_path}")

    # Include Paddle headers
    include_directories(${paddle_inference_path}/paddle/include)

    # Include third-party libraries
    set(PADDLE_LIB_THIRD_PARTY_PATH "${paddle_inference_path}/third_party/install/")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
    
    # Link directories
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
    link_directories("${paddle_inference_path}/paddle/lib")

    # Check for MKL
    set(mklml_inc_path ${paddle_inference_path}/third_party/install/mklml/include)
    set(mklml_lib_path ${paddle_inference_path}/third_party/install/mklml/lib)
    if(EXISTS ${mklml_inc_path} AND EXISTS ${mklml_lib_path})
        message(STATUS "MKLML lib found in Paddle.")
        include_directories(${paddle_inference_path}/third_party/install/mklml/include)
        if(APPLE)
            set(MATH_LIB ${mklml_lib_path}/libmklml_intel.dylib
                         ${mklml_lib_path}/libiomp5.dylib)
        elseif(UNIX)
            set(MATH_LIB ${mklml_lib_path}/libmklml_intel.so
                         ${mklml_lib_path}/libiomp5.so)
        endif()
    endif()
    
    # Define the Paddle library to link against
    if(APPLE)
        set(PADDLE_LIB ${paddle_inference_path}/paddle/lib/libpaddle_inference.dylib)
    else()
        set(PADDLE_LIB ${paddle_inference_path}/paddle/lib/libpaddle_inference.so)
    endif()
    
    # Add definition to indicate Paddle is available
    add_definitions(-D__USE_PADDLE_INFERENCE__)
    message(STATUS "Defined macro: __USE_PADDLE_INFERENCE__")

    # Link the Paddle Inference library to main target now that PADDLE_LIB is defined
    target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE ${PADDLE_LIB})
    if(DEFINED MATH_LIB)
        target_link_libraries(${AIBOT_TARGET_NAME} PRIVATE ${MATH_LIB})
    endif()
endif()

if(LINUX)
    target_link_libraries(unit_test PRIVATE cpr::cpr gtest gtest_main fmt::fmt yaml-cpp::yaml-cpp Boost::filesystem tokenizers_cpp ncnn general-wheel-cpp::general-wheel-cpp)
    if(NOT use_libtorch)
        target_link_libraries(unit_test PRIVATE onnxruntime::onnxruntime)
    endif()
    if(use_libtorch)
        target_link_libraries(unit_test PRIVATE torch)
    endif()
    # Link Paddle Inference library for unit tests if defined
    if(paddle_inference_path)
        target_link_libraries(unit_test PRIVATE ${PADDLE_LIB})
        if(DEFINED MATH_LIB)
            target_link_libraries(unit_test PRIVATE ${MATH_LIB})
        endif()
    endif()
    target_link_libraries(unit_test PRIVATE /usr/local/lib64/libmysqlcppconnx.so)
else()
    target_link_libraries(unit_test PRIVATE cpr::cpr GTest::GTest GTest::Main fmt::fmt yaml-cpp::yaml-cpp Boost::filesystem tokenizers_cpp ncnn general-wheel-cpp::general-wheel-cpp)
    if(NOT use_libtorch)
        target_link_libraries(unit_test PRIVATE onnxruntime::onnxruntime)
    endif()
    if(use_libtorch)
        target_link_libraries(unit_test PRIVATE torch)
    endif()
    # Link Paddle Inference library for unit tests if defined
    if(paddle_inference_path)
        target_link_libraries(unit_test PRIVATE ${PADDLE_LIB})
        if(DEFINED MATH_LIB)
            target_link_libraries(unit_test PRIVATE ${MATH_LIB})
        endif()
    endif()
    target_link_libraries(unit_test PRIVATE /opt/homebrew/lib/libmysqlcppconnx.dylib)
endif()

add_test(NAME unit_test COMMAND unit_test)

# Add precompiled macro definitions
# target_compile_definitions(${MIRAICP_TARGET_NAME} PUBLIC JSON_MultipleHeaders=ON MIRAICP_LIB_SDK)

# Install instructions
# install(TARGETS ${MIRAICP_TARGET_NAME}
#     DESTINATION ${CMAKE_INSTALL_PREFIX}
#     RENAME libAIBot800a.so
# )

# if(APPLE)
#     # macOS may need additional configuration, such as setting rpath
#     set_target_properties(${MIRAICP_TARGET_NAME} PROPERTIES INSTALL_NAME_DIR "@rpath")
# endif()

# set spdlog log level based on build type
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()

# get git commit hash for BUILD_VERSION
set(BUILD_VERSION "unknown" CACHE STRING "Version string for the build")

# try to get git commit hash if not specified manually
if(BUILD_VERSION STREQUAL "unknown")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT_HASH
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(GIT_COMMIT_HASH)
            set(BUILD_VERSION "${GIT_COMMIT_HASH}")
            message(STATUS "Using git commit hash for BUILD_VERSION: ${BUILD_VERSION}")
            
            # Get the commit message for the current hash
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --pretty=%B ${GIT_COMMIT_HASH}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_COMMIT_MESSAGE
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(GIT_COMMIT_MESSAGE)
                message(STATUS "Got commit message: ${GIT_COMMIT_MESSAGE}")
                # Escape quotes to avoid issues in C-string literals
                string(REPLACE "\"" "\\\"" GIT_COMMIT_MESSAGE "${GIT_COMMIT_MESSAGE}")
            else()
                set(GIT_COMMIT_MESSAGE "No commit message available")
            endif()
            # Define the commit message
            add_definitions(-DCOMMIT_MESSAGE="${GIT_COMMIT_MESSAGE}")
        endif()
    endif()
endif()

# add definition for BUILD_VERSION
add_definitions(-DBUILD_VERSION="${BUILD_VERSION}")

add_definitions(-DREPOS_ADDR="https://github.com/FredYakumo/zihuan-aibot-800a")

# add custom macro definitions
add_definitions(-DAIBOT_VERSION_800A)

# platform-based macro definitions
if(APPLE)
    add_definitions(-DPLATFORM_MACOS)
elseif(LINUX)
    add_definitions(-DPLATFORM_LINUX)
elseif(WIN32)
    add_definitions(-DPLATFORM_WINDOWS)
endif()

# build type-based macro definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_BUILD)
else()
    add_definitions(-DRELEASE_BUILD)
endif()

# AI inference backend-based macro definitions
if(use_libtorch)
    add_definitions(-D__USE_LIBTORCH__)
    message(STATUS "Defined macro: __USE_LIBTORCH__")
else()
    add_definitions(-D__USE_ONNX_RUNTIME__)
    message(STATUS "Defined macro: __USE_ONNX_RUNTIME__")
endif()

# Create symlink for compile_commands.json to main build directory for clangd
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    # Ensure the main build directory exists
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
    
    # Create symlink immediately during configure phase
    # First, remove existing symlink if it exists to avoid conflicts
    if(EXISTS ${CMAKE_SOURCE_DIR}/build/compile_commands.json)
        file(REMOVE ${CMAKE_SOURCE_DIR}/build/compile_commands.json)
    endif()
    
    # Create the symlink to the binary directory's compile_commands.json
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink 
            ${CMAKE_BINARY_DIR}/compile_commands.json 
            ${CMAKE_SOURCE_DIR}/build/compile_commands.json
        RESULT_VARIABLE symlink_result
    )
    
    if(symlink_result EQUAL 0)
        message(STATUS "Created compile_commands.json symlink: ${CMAKE_SOURCE_DIR}/build/compile_commands.json -> ${CMAKE_BINARY_DIR}/compile_commands.json")
    else()
        message(WARNING "Failed to create compile_commands.json symlink")
    endif()
endif()