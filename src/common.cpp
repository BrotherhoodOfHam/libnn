#include "nn/common.h"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "nn/tensors.h"

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

inline std::string make_message(nn::uint expected_dims, nn::uint dims)
{
	std::stringstream ss;
	ss << "Expected shape of size " << expected_dims << " was given size " << dims;
	return ss.str();
}

nn::tensor_layout_error::tensor_layout_error(uint expected_dims, uint dims) :
	runtime_error(make_message(expected_dims, dims))
{}

/*************************************************************************************************************************************/
