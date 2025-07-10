#pragma once
#include <string>
#include <chrono>


struct UserProtait {
    std::string protait;
    double favorability;
    std::chrono::system_clock::time_point create_time;
};