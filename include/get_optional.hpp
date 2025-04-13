#ifndef GET_OPTIONAL_HPP
#define GET_OPTIONAL_HPP

#include "nlohmann/json.hpp"

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

#endif