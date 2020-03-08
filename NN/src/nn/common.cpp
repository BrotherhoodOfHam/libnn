#include "common.h"

#include <chrono>
#include <iomanip>

/*************************************************************************************************************************************/

std::ostream& nn::time_stamp(std::ostream& out)
{
	using namespace std;
	using namespace std::chrono;

	const static auto start = system_clock::now();
	auto x = 1s;
	auto time = duration_cast<seconds>(system_clock::now() - start).count();
	auto h = time / 3600;
	auto m = (time / 60) - (h * 60);
	auto s = time % 60;

	out << "["
		<< setfill('0') << setw(2) << h << ":"
		<< setfill('0') << setw(2) << m << ":"
		<< setfill('0') << setw(2) << s << "]";
	return out;
}

/*************************************************************************************************************************************/
