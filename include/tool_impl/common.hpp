// Common helpers for tool implementations
#pragma once

#include <string>
#include <string_view>

namespace tool_impl {

inline std::string get_permission_chs(const std::string_view perm) {
    if (perm == "OWNER") {
        return "群主";
    } else if (perm == "ADMINISTRATOR") {
        return "管理员";
    }
    return "普通群友";
}

} // namespace tool_impl
