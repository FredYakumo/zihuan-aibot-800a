#ifndef GET_OPTIONAL_HPP
#define GET_OPTIONAL_HPP

#include "nlohmann/json.hpp"
#include <optional>
#include <function_ref.

/**
 * @brief Converts a JSON field to an `std::optional<T>`.
 *
 * This function checks if the specified key exists in the JSON object and if the value is not null.
 * If both conditions are met, it attempts to convert the value to the specified type `T`.
 * If the key does not exist, the value is null, or the conversion fails, it returns `std::nullopt`.
 *
 * @tparam T The type to which the JSON value should be converted.
 * @param j The JSON object from which to extract the value.
 * @param key The key of the field to extract from the JSON object.
 * @return An `std::optional<T>` containing the converted value if successful; otherwise, `std::nullopt`.
 */
template <typename T = nlohmann::json>
std::optional<T> get_optional(const nlohmann::json &j, const std::string_view key) {
    if (j.contains(key) && !j[key].is_null()) {
        try {
            return j[key].get<T>();
        } catch (const nlohmann::json::exception &) {
            // If the type conversion fails, return an empty optional.
            return std::nullopt;
        }
    }
    // Return an empty optional if the key does not exist or the value is null.
    return std::nullopt;
}

/**
 * @brief Applies a transformation to the value contained within an std::optional.
 *
 * This function takes an std::optional<T> and a callable transform function. If the optional
 * holds a value, the transform function is applied to that value, and the result is wrapped
 * in an std::optional. If the optional is empty, the function returns std::nullopt.
 *
 * Template Parameters:
 * @tparam T The type of the value stored in the input std::optional.
 * @tparam F The type of the callable that takes a const T& and returns a result.
 *
 * @param opt An std::optional<T> that potentially holds a value to be transformed.
 * @param transform A callable that accepts a const T& and returns a new value.
 *
 * @return An std::optional containing the result of applying transform to the contained value,
 *         or std::nullopt if the input optional is empty.
 *
 * @note The return type is automatically deduced from the result of the transform callable.
 */
 template <typename T, typename F>
 auto map_optional(const std::optional<T> &opt, F transform) -> std::optional<decltype(transform(*opt))> {
     if (opt.has_value()) {
         return transform(opt.value());
     }
     return std::nullopt;
 }

#endif