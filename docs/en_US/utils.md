# Utils API Reference

This document provides a reference for the utility functions and classes used throughout the AIBot Zihuan project. These utilities include time formatting, string manipulation, thread management, and other helper functions.

## Time Utilities

### `get_current_time_formatted`

```cpp
std::string get_current_time_formatted();
```

Returns the current system time formatted as a string in the format: "YYYY年MM月DD日 HH:MM:SS".

### `time_point_to_db_str`

```cpp
std::string time_point_to_db_str(const std::chrono::system_clock::time_point &time_point);
```

Converts a `std::chrono::system_clock::time_point` to a string formatted for database storage in ISO 8601 format with timezone offset.

- **Parameters**: `time_point` - The time point to convert
- **Returns**: Formatted string in the format: "YYYY-MM-DDTHH:MM:SS±HH:MM"

### `get_current_time_db`

```cpp
std::string get_current_time_db();
```

Returns the current system time formatted as a string suitable for database storage using `time_point_to_db_str`.

### `get_today_date_str`

```cpp
std::string get_today_date_str();
```

Returns the current date formatted as a string in the format: "YYYY年MM月DD日".

## String Utilities

### `extract_parentheses_content_after_keyword`

```cpp
std::string_view extract_parentheses_content_after_keyword(const std::string_view s, const std::string_view keyword);
```

Extracts the content inside parentheses following the first occurrence of a specified keyword.

- **Parameters**:
  - `s` - The input string to search
  - `keyword` - The keyword to search for (expected format: "#keyword")
- **Returns**: String view containing the content inside the parentheses, or empty string view if not found

### `replace_keyword_and_parentheses_content`

```cpp
std::string replace_keyword_and_parentheses_content(const std::string_view original_str, const std::string_view keyword, const std::string_view replacement);
```

Replaces the first occurrence of a keyword and its following parentheses content with a replacement string.

- **Parameters**:
  - `original_str` - The original string to modify
  - `keyword` - The keyword to search for
  - `replacement` - The string to replace the matched content with
- **Returns**: Modified string with replacement applied, or original string if keyword/parentheses not found

### `is_strict_format`

```cpp
bool is_strict_format(const std::string_view s, const std::string_view keyword);
```

Checks if a string strictly follows the format: "#keyword(content)".

- **Parameters**:
  - `s` - The input string to validate
  - `keyword` - The keyword to check for
- **Returns**: `true` if string matches format, `false` otherwise

## Thread Utilities

### `set_thread_name`

```cpp
void set_thread_name(const std::string &name);
```

Sets the name of the current thread. Implementation is platform-specific.

- **Parameters**: `name` - The name to assign to the thread
- **Platform Notes**:
  - Linux: Limited to 15 characters
  - macOS: Limited to 63 characters
  - Windows: Converts name to wide string

## Collection Utilities

### `group_by`

```cpp
template <typename T, typename KEY_GENERATOR>
auto group_by(const std::vector<T> &collection, KEY_GENERATOR key_generator) -> std::unordered_map<decltype(key_generator(std::declval<T>())), T>;
```

Groups elements of a collection into an unordered map using a key generator function.

- **Parameters**:
  - `collection` - The vector of elements to group
  - `key_generator` - A function that generates a key from an element
- **Returns**: Unordered map with generated keys and corresponding elements

## Value Check Utilities

### `is_positive_value`

```cpp
bool is_positive_value(const std::string_view s);
```

Checks if a string represents a positive value.

- **Parameters**: `s` - The string to check
- **Returns**: `true` if string is one of: "true", "1", "yes", "y", "是"

### `is_negative_value`

```cpp
bool is_negative_value(const std::string_view s);
```

Checks if a string represents a negative value.

- **Parameters**: `s` - The string to check
- **Returns**: `true` if string is one of: "false", "0", "no", "n", "否"

## Constants

### `AVAILABLE_VALUE_STRINGS`

```cpp
constexpr std::string_view AVAILABLE_VALUE_STRINGS[] = {"true", "1", "yes", "y", "是", "否"};
```

Array containing all recognized string values for positive/negative checks.



# get_optional API Reference

This document provides a reference for the `get_optional` utility functions, which handle safe extraction and transformation of optional values from JSON objects and other contexts.

## Overview

The `get_optional` library provides template functions for safely extracting values from JSON objects into `std::optional<T>` and transforming optional values. These utilities help prevent runtime errors when dealing with missing or null values in JSON data.

## Functions

### `get_optional`

```cpp
template <typename T = nlohmann::json>
std::optional<T> get_optional(const nlohmann::json &j, const std::string_view key);
```

Converts a JSON field to an `std::optional<T>` by checking if the specified key exists and contains a non-null value.

#### Parameters
- `j`: The JSON object from which to extract the value.
- `key`: The key of the field to extract from the JSON object.

#### Returns
- `std::optional<T>` containing the converted value if the key exists, the value is non-null, and conversion succeeds.
- `std::nullopt` if the key is missing, the value is null, or conversion fails.

#### Example
```cpp
nlohmann::json j = {{"name", "Alice"}, {"age", nullptr}};
std::optional<std::string> name = get_optional<std::string>(j, "name"); // Contains "Alice"
std::optional<int> age = get_optional<int>(j, "age"); // nullopt (value is null)
std::optional<int> height = get_optional<int>(j, "height"); // nullopt (key missing)
```

### `map_optional`

```cpp
template <typename T, typename F>
auto map_optional(const std::optional<T> &opt, F transform) -> std::optional<decltype(transform(*opt))>;
```

Applies a transformation function to the value contained within an `std::optional<T>`, returning a new `std::optional` with the transformed value if the input optional has a value.

#### Template Parameters
- `T`: The type of the value stored in the input `std::optional`.
- `F`: The type of the callable transform function that accepts a `const T&` and returns a result.

#### Parameters
- `opt`: An `std::optional<T>` that may contain a value to transform.
- `transform`: A callable that accepts a `const T&` and returns a new value.

#### Returns
- `std::optional` containing the result of the transform if `opt` has a value.
- `std::nullopt` if `opt` is empty.

#### Example
```cpp
std::optional<int> number = 5;
auto squared = map_optional(number, [](int n) { return n * n; }); // Contains 25
