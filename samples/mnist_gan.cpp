/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"

#include "gan.h"
#include "nn/node/activations.h"
#include "nn/node/dense.h"
#include "nn/node/dropout.h"

using namespace nn;

/*************************************************************************************************************************************/

void process_image(std::vector<trainer::data>& dataset, const std::vector<uint8_t>& image, bool normalize_negative = false)
{
	dataset.emplace_back(image.size());
	auto& data = dataset.back();
	for (size_t i = 0; i < data.size(); i++)
	{
		data[i] = (float)image[i] / 255.0f;
		if (normalize_negative)
			data[i] = (data[i] - 0.5f) * 2;
	}
}

int main()
{
	uint z_size = 10;
	uint img_size = 28 * 28;
	uint batch_size = 25;

	model g(z_size);
	g.add<dense_layer>(256);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(512);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(1024);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(img_size);
	g.add<activation::tanh>();

	model d(img_size);
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

	GAN gan(g, d);

	//prepare data
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	const size_t dataset_size = dataset.training_images.size();
	std::vector<trainer::data> real_data;
	real_data.reserve(dataset_size);
	
	for (const auto& image : dataset.training_images)
		process_image(real_data, image, true);

	gan.train(real_data, 300, batch_size);

	return 0;
}

/*************************************************************************************************************************************/
