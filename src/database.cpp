#include "database.h"
#include "config.h"
#include <memory>
#include <stdexcept>

using namespace database;

std::shared_ptr<DBConnection> g_db_connection;

void database::init_db_connection() {
    const auto &config = Config::instance();
    g_db_connection = std::make_shared<DBConnection>(std::string(config.database_host), config.database_port, std::string(config.database_user),
                                                     std::string(config.database_password));
    spdlog::info("Init db connection successed. host: {}:{}", config.database_host, config.database_port);
}

DBConnection &database::get_global_db_connection() {
    if (g_db_connection == nullptr) {
        throw std::runtime_error("get_global_db_connection() Error: Database connection is not initialized.");
    }
    return *g_db_connection;
}