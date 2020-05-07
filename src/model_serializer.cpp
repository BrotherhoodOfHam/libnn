/*
	Model serialization
*/

#include <fstream>

#include "nn/model.h"

using namespace nn;

/*************************************************************************************************************************************/

static const char s_magic[4] = { 'N', 'N', 'M', 'D' };

template<typename type>
void write(std::ostream& out, const type& value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(type));
}

template<typename type>
type read(std::istream& in)
{
	type v;
	in.read(reinterpret_cast<char*>(&v), sizeof(type));
	return v;
}

/*************************************************************************************************************************************/


void model::serialize(const std::string& filename)
{
	std::fstream f(filename, std::ios::binary | std::ios::out);
	if (f.fail())
	{
		std::cerr << "could not write: " << filename << std::endl;
		throw std::exception();
	}

	std::vector<scalar> param_buffer;
	std::vector<node_parameter> parameter_set;
	for (auto node : _nodes)
		node->get_parameters(parameter_set);

	write(f, s_magic);
	write<uint>(f, (uint)parameter_set.size());

	auto& d = device::get();

	for (auto param : parameter_set)
	{
		const auto& p = param.p;

		// write shape
		write<uint>(f, (uint)p.shape().size());
		for (uint i : p.shape())
			write<uint>(f, i);

		// write size of tensor
		write<uint>(f, p.shape().datasize());

		// write data
		d.read(p, param_buffer);
		f.write((const char*)param_buffer.data(), (std::streamoff)sizeof(scalar) * param_buffer.size());
	}
}

void model::deserialize(const std::string& filename)
{
	std::fstream f(filename, std::ios::binary | std::ios::in);
	if (f.fail())
	{
		std::cout << "could not read file: " << filename << std::endl;
		throw std::exception();
	}

	uint magic = read<uint>(f);
	if (magic != *reinterpret_cast<const uint*>(s_magic))
	{
		std::cout << "incorrect file format: " << filename << std::endl;
		throw std::exception();
	}

	std::vector<node_parameter> parameter_set;
	for (auto node : _nodes)
		node->get_parameters(parameter_set);

	uint list_count = read<uint>(f);
	if (list_count != parameter_set.size())
	{
		std::cout << "incorrect parameter count";
		throw std::exception();
	}

	auto& d = device::get();
	std::vector<scalar> buf;

	for (auto param : parameter_set)
	{
		const auto& p = param.p;

		// read shape
		tensor_shape& shape = p.shape();
		uint shape_size = read<uint>(f);
		if (shape.size() != shape_size)
		{
			std::cerr << "parameter dimensionality does not match: " << shape_size << " != " << p.shape().size() << std::endl;
			throw std::exception();
		}

		for (uint shape_dim : shape)
		{
			uint dim = read<uint>(f);
			if (dim != shape_dim)
			{
				std::cerr << "parameter shape dimension does not match" << std::endl;
				throw std::exception();
			}
		} 

		// read size of tensor
		uint parameter_size = read<uint>(f);
		if (parameter_size != p.size())
		{
			std::cerr << "parameter count does not match: " << parameter_size << " != " << p.size() << std::endl;
			throw std::exception();
		}

		// read tensor data
		buf.resize(parameter_size);
		f.read((char*)buf.data(), (std::streamoff)sizeof(scalar) * parameter_size);
		d.update(param.p, buf);
	}
}

/*************************************************************************************************************************************/
