/*
	Model serialization
*/

#include <fstream>

#include "model.h"
#include "dense_layer.h"

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

bool model::serialize(const std::string& filename)
{
	std::fstream f(filename, std::ios::binary | std::ios::out);
	if (f.fail())
	{
		std::cout << "could not write: " << filename << std::endl;
		return false;
	}

	std::vector<nodes::dense_layer*> layers;
	for (auto& node : _nodes)
	{
		auto dense = dynamic_cast<nodes::dense_layer*>(node.get());
		if (dense != nullptr) layers.push_back(dense);
	}

	write(f, s_magic);
	write<uint>(f, layers.size());

	for (auto layer : layers)
	{
		const tensor& w = layer->weights();
		const tensor& b = layer->biases();

		write<uint>(f, w.memory_size());
		write<uint>(f, b.memory_size());
		write<uint>(f, layer->input_shape().length());
		for (uint i : layer->input_shape())
			write<uint>(f, i);

		write<uint>(f, layer->output_shape().length());
		for (uint i : layer->output_shape())
			write<uint>(f, i);

		f.write((const char*)w.memory(), sizeof(scalar) * w.memory_size());
		f.write((const char*)b.memory(), sizeof(scalar) * b.memory_size());
	}

	return true;
}

bool model::deserialize(const std::string& filename)
{
	std::fstream f(filename, std::ios::binary | std::ios::in);
	if (f.fail())
	{
		std::cout << "could not read file: " << filename << std::endl;
		return false;
	}

	std::vector<nodes::dense_layer*> layers;
	for (auto& node : _nodes)
	{
		auto dense = dynamic_cast<nodes::dense_layer*>(node.get());
		if (dense != nullptr) layers.push_back(dense);
	}

	uint magic = read<uint>(f);
	if (magic != *reinterpret_cast<const uint*>(s_magic))
	{
		std::cout << "incorrect file format: " << filename << std::endl;
		return false;
	}

	uint layer_count = read<uint>(f);
	if (layer_count != layers.size())
	{
		std::cout << "incorrect layer count";
		return false;
	}

	for (auto layer : layers)
	{
		const auto& in_shape = layer->input_shape();
		const auto& out_shape = layer->output_shape();

		uint w_size = read<uint>(f);
		uint b_size = read<uint>(f);

		const tensor& w = layer->weights();
		const tensor& b = layer->biases();

		if (w_size != w.memory_size())
		{
			std::cout << "weight count does not match: " << w_size << " != " << w.memory_size() << std::endl;
			return false;
		}
		if (b_size != b.memory_size())
		{
			std::cout << "bias count does not match: " << b_size << " != " << b.memory_size() << std::endl;
			return false;
		}

		uint in_shape_size = read<uint>(f);
		if (in_shape_size != in_shape.length())
		{
			std::cout << "shape does not match" << std::endl;
			return false;
		}
		for (uint i = 0; i < in_shape_size; i++)
		{
			if (read<uint>(f) != in_shape[i])
			{
				std::cout << "shape dimension does not match" << std::endl;
				return false;
			}
		}

		uint out_shape_size = read<uint>(f);
		if (out_shape_size != out_shape.length())
		{
			std::cout << "shape does not match" << std::endl;
			return false;
		}
		for (uint i = 0; i < out_shape_size; i++)
		{
			if (read<uint>(f) != out_shape[i])
			{
				std::cout << "shape dimension does not match" << std::endl;
				return false;
			}
		}

		f.read((char*)w.memory(), sizeof(scalar) * w.memory_size());
		f.read((char*)b.memory(), sizeof(scalar) * b.memory_size());
	}

	return true;
}

/*************************************************************************************************************************************/
