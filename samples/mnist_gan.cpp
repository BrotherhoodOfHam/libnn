/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"

#include "nn/ops/activations.h"
#include "nn/ops/dense.h"
#include "nn/ops/dropout.h"
#include "gan.h"

using namespace nn;

/*************************************************************************************************************************************/

void process_image(std::vector<trainer::data>& dataset, const std::vector<uint8_t>& image, bool normalize_negative = false)
{
	auto& data = dataset.emplace_back(image.size());
	for (size_t i = 0; i < data.size(); i++)
	{
		data[i] = (float)image[i] / 255.0f;
		if (normalize_negative)
			data[i] = (data[i] - 0.5f) * 2;
	}
}

int main()
{
	uint batch_size = 100;
	uint z_size = 10;
	uint img_size = 28 * 28;

	model g(z_size, batch_size);
	g.add<dense_layer>(256);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(512);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(1024);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(img_size);
	g.add<activation::tanh>();

	model d(img_size, batch_size);
	d.add<dense_layer>(1024);
	d.add<activation::leaky_relu>(0.2f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(512);
	d.add<activation::leaky_relu>(0.2f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(256);
	d.add<activation::leaky_relu>(0.2f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(1);
	d.add<activation::sigmoid>();

	gan gn(g, d);

	//prepare data
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	const size_t dataset_size = dataset.training_images.size();
	std::vector<trainer::data> real_data;
	real_data.reserve(dataset_size);
	
	for (const auto& image : dataset.training_images)
		process_image(real_data, image, true);

	gn.train(real_data, 300);

	return 0;
}

/*************************************************************************************************************************************/
