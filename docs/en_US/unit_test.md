# Unit Tests Documentation

This document provides a comprehensive overview of the unit tests in the zihuan-aibot-800a project. The tests are organized by functionality and cover different inference engines (ONNX Runtime and LibTorch), device types (CPU/MPS), and performance scenarios.

## Test Summary

| Test Category | Test Name | Status | Purpose |
|---------------|-----------|--------|---------|
| **Basic Setup** | `BasicSetup.FrameworkInitialization` | ✅ Active | Test framework validation |
| **ONNX Runtime** | `ONNXRuntime.ProviderAvailabilityCheck` | ✅ Active | Provider enumeration |
| **ONNX Runtime** | `ONNXRuntime.CosineSimilarityInference` | ✅ Active | Similarity computation |
| **ONNX Runtime** | `ONNXRuntime.BatchTextEmbeddingInference` | ✅ Active | Batch text processing |
| **Bot Adapter** | `BotAdapter.WebSocketConnectionSetup` | 🚫 Commented | WebSocket connectivity |
| **Bot Adapter** | `BotAdapter.MessageIdRetrieval` | 🚫 Commented | Message ID retrieval |
| **Bot Adapter** | `BotAdapter.LongTextMessageSending` | 🚫 Commented | Long message handling |
| **String Utils** | `StringUtils.KeywordReplacementOperations` | ✅ Active | Text manipulation |
| **LibTorch MPS** | `LibTorchMPS.TextEmbeddingPerformanceComparison` | ✅ Active | MPS performance testing |
| **LibTorch MPS** | `LibTorchMPS.LargeBatchTextEmbedding` | ✅ Active | Large batch processing |
| **LibTorch CPU** | `LibTorchCPU.TextEmbeddingPerformanceComparison` | ✅ Active | CPU performance testing |
| **LibTorch CPU** | `LibTorchCPU.CosineSimilarityInference` | ✅ Active | CPU similarity computation |
| **LibTorch Hybrid** | `LibTorchHybrid.MPSEmbeddingWithCPUSimilarity` | ✅ Active | Cross-device processing |
| **LibTorch Hybrid** | `LibTorchHybrid.MPSIndividualEmbeddingWithCPUSimilarity` | ✅ Active | Individual vs batch validation |
| **LibTorch Accuracy** | `LibTorchAccuracy.TextEmbeddingBatchVsIndividualConsistency` | ✅ Active | Numerical consistency |
| **LibTorch Accuracy** | `LibTorchAccuracy.TokenEmbeddingBatchVsIndividualConsistency` | ✅ Active | Token-level consistency |

**Total Tests**: 16 (13 active, 3 commented out)

## Test Categories

### 1. Basic Setup Tests

#### `BasicSetup.FrameworkInitialization`
- **Purpose**: Validates that the Google Test framework initializes correctly
- **Scope**: Basic test infrastructure verification
- **Dependencies**: Google Test framework

### 2. ONNX Runtime Tests

These tests are conditionally compiled when `__USE_ONNX_RUNTIME__` is defined.

#### `ONNXRuntime.ProviderAvailabilityCheck`
- **Purpose**: Verifies ONNX Runtime providers and version information
- **Functionality**: 
  - Logs ONNX Runtime version
  - Lists all available execution providers (CPU, CUDA, etc.)
- **Expected Output**: Provider list and version information in logs

#### `ONNXRuntime.CosineSimilarityInference`
- **Purpose**: Tests cosine similarity computation using ONNX models
- **Test Data**: Chinese text samples including "杀猪盘" related phrases
- **Functionality**:
  - Computes text embeddings using ONNX text embedding model
  - Calculates cosine similarities using ONNX cosine similarity model
  - Validates similarity scores are within [-1, 1] range
  - Verifies highest similarity for identical text ("杀猪" with itself)
- **Performance Metrics**: Embedding and similarity computation timing

#### `ONNXRuntime.BatchTextEmbeddingInference`
- **Purpose**: Tests batch text embedding processing with ONNX models
- **Functionality**:
  - Batch processing of multiple text inputs
  - Integration with cosine similarity computation
  - Validation of batch processing efficiency
- **Expected Results**: Consistent results with individual processing

### 3. Bot Adapter Tests

#### `BotAdapter.WebSocketConnectionSetup`
- **Purpose**: Tests WebSocket connection setup for QQ bot integration
- **Functionality**: 
  - WebSocket connection to localhost:13378
  - Event handling registration
  - Message sending capabilities
- **Status**: Currently commented out (requires running MiraiCP server)

#### `BotAdapter.MessageIdRetrieval`
- **Purpose**: Tests message ID retrieval functionality
- **Functionality**: Retrieves message IDs from specific groups/users
- **Status**: Currently commented out (requires live connection)

#### `BotAdapter.LongTextMessageSending` (Commented Out)
- **Purpose**: Tests long text message sending functionality with automatic chunking
- **Functionality**: 
  - Sends very long plain text messages
  - Tests message chunking and delivery mechanisms
  - Validates large message handling in QQ groups
- **Status**: Currently commented out (requires running MiraiCP server)
- **Test Data**: ~1.5KB of repetitive text to test chunking behavior

### 4. String Utilities Tests

#### `StringUtils.KeywordReplacementOperations`
- **Purpose**: Tests string manipulation utilities
- **Test Cases**:
  - Basic keyword replacement: "#联网 abc" → " abc"
  - Keyword with parentheses removal: "abc #联网(123)" → "abc "
- **Functionality**: Validates `replace_str` and `replace_keyword_and_parentheses_content` functions

### 5. LibTorch Tests

These tests are conditionally compiled when `__USE_LIBTORCH__` is defined.

#### Performance Tests

##### `LibTorchMPS.TextEmbeddingPerformanceComparison`
- **Purpose**: Compares individual vs batch embedding performance on MPS (Metal Performance Shaders)
- **Test Scale**: 250 text samples (50 iterations × 5 base texts)
- **Metrics**: Processing time comparison between individual and batch inference
- **Device**: MPS (Apple Silicon GPU acceleration)

##### `LibTorchMPS.LargeBatchTextEmbedding`
- **Purpose**: Tests large-scale batch processing capabilities
- **Test Scale**: 10,000 text samples (2000 iterations × 5 base texts)
- **Focus**: Memory efficiency and processing speed for large batches
- **Device**: MPS

##### `LibTorchCPU.TextEmbeddingPerformanceComparison`
- **Purpose**: Same as MPS version but running on CPU
- **Test Scale**: 250 text samples
- **Device**: CPU
- **Comparison**: Performance baseline against MPS acceleration

#### Functional Tests

##### `LibTorchCPU.CosineSimilarityInference`
- **Purpose**: Tests complete similarity computation pipeline with LibTorch
- **Functionality**:
  - Text embedding generation using LibTorch models
  - Cosine similarity computation
  - Result validation and accuracy testing
- **Device**: CPU

##### `LibTorchHybrid.MPSEmbeddingWithCPUSimilarity`
- **Purpose**: Tests hybrid processing: MPS for embeddings, CPU for similarity
- **Workflow**:
  - Generate embeddings using MPS-accelerated models
  - Compute similarities using CPU implementation
  - Cross-validate with pure LibTorch results
- **Validation**: Comparison between CPU and LibTorch similarity results (tolerance: 0.001)

##### `LibTorchHybrid.MPSIndividualEmbeddingWithCPUSimilarity`
- **Purpose**: Tests individual embedding processing with hybrid approach
- **Workflow**:
  - Individual text embedding using MPS
  - CPU-based cosine similarity computation
  - Comparison with batch embedding results
- **Validation**: Individual vs batch embedding consistency

#### Accuracy Tests

##### `LibTorchAccuracy.TextEmbeddingBatchVsIndividualConsistency`
- **Purpose**: Validates numerical consistency between batch and individual inference
- **Analysis**:
  - Vector-level difference analysis
  - Statistical metrics: mean absolute difference, max difference, RMSE
  - Element-by-element comparison for first/last 20 elements
- **Tolerance**: Mean absolute difference < 1e-6, max difference < 1e-5
- **Performance**: Reports speed improvement of batch vs individual processing

##### `LibTorchAccuracy.TokenEmbeddingBatchVsIndividualConsistency`
- **Purpose**: Tests token-level embedding consistency (without mean pooling)
- **Analysis**:
  - Token-by-token embedding comparison
  - Per-token statistical analysis
  - Overall aggregated statistics
- **Model**: Uses `TextEmbeddingModel` (no mean pooling) vs `TextEmbeddingWithMeanPoolingModel`
- **Validation**: Token-level numerical consistency verification

## Test Data

### Sample Texts
The tests use Chinese text samples related to fraud detection scenarios:
- "如何进行杀猪盘" (How to conduct pig butchering scam)
- "怎么快速杀猪" (How to quickly slaughter pigs)
- "怎么学习Rust" (How to learn Rust)
- "杀猪的经验" (Experience in pig slaughtering)
- "杀猪" (Pig slaughtering)

### Expected Behavior
- Highest similarity should be between identical texts
- Similarity scores should be in range [-1, 1]
- "杀猪" should have highest similarity with itself
- Related terms should show higher similarity than unrelated ones

## Build and Execution

### Prerequisites
- CMake with preset configuration
- Google Test framework
- Conditional dependencies:
  - ONNX Runtime (when `__USE_ONNX_RUNTIME__` is defined)
  - LibTorch (when `__USE_LIBTORCH__` is defined)

### Build Commands
```bash
# For ONNX Runtime tests
cmake --preset ninja-onnxruntime-debug
cmake --build --preset ninja-onnxruntime-debug-build

# For LibTorch tests  
cmake --preset ninja-libtorch-debug
cmake --build --preset ninja-libtorch-debug-build
```

### Execution
```bash
# Run all tests
./build/{preset}/unit_test

# Run specific test categories
./build/{preset}/unit_test --gtest_filter="ONNXRuntime.*"
./build/{preset}/unit_test --gtest_filter="LibTorchCPU.*"
./build/{preset}/unit_test --gtest_filter="LibTorchMPS.*"
```

## Performance Expectations

### Typical Results
- **Batch vs Individual**: Batch processing typically 2-5x faster
- **MPS vs CPU**: MPS acceleration shows significant speedup for large batches
- **Numerical Accuracy**: Differences should be < 1e-6 for mean, < 1e-5 for max

### Memory Usage
- Large batch tests (10,000 samples) validate memory efficiency
- Individual processing has lower peak memory usage
- Batch processing has better throughput

## Troubleshooting

### Common Issues
1. **Missing Models**: Ensure model files are in `exported_model/` directory
2. **Device Availability**: MPS tests require Apple Silicon with Metal support
3. **Memory Limits**: Large batch tests may fail on systems with limited memory
4. **Compilation Flags**: Ensure correct conditional compilation flags are set

### Debug Information
- Tests use `spdlog` for detailed logging
- Set log level to `debug` for verbose output
- Performance metrics are logged for all timing-sensitive tests

## Future Extensions

### Planned Test Additions
- Multi-threaded inference testing
- Model accuracy regression tests
- Memory leak detection
- Cross-platform compatibility validation
- Real-world data integration tests

### Test Coverage Goals
- Increase coverage of edge cases
- Add stress testing for production scenarios
- Implement automated performance regression detection
