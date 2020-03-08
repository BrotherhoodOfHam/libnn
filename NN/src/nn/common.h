/*
	Common headers and functions
*/

#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cassert>

namespace nn
{
	using byte = unsigned char;
	using uint = unsigned int;

	// print the time
	std::ostream& time_stamp(std::ostream& out);
}
