/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"
#include "nn/model.h"
#include "gan.h"

using namespace nn;

/*************************************************************************************************************************************/

int gan_main()
{
	const size_t z_size = 10;
	const size_t img_size = 28 * 28;

	model g(
		z_size,
		{
			layer(256, activation::leaky_relu),
			layer(512, activation::leaky_relu),
			layer(img_size, activation::tanh),
		}
	);
	model d(
		img_size,
		{
			layer(512, activation::leaky_relu),
			layer(256, activation::leaky_relu),
			layer(1, activation::sigmoid),
		}
	);

	gan gn(&g, &d);

	//prepare data
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	const size_t dataset_size = dataset.training_images.size();
	std::vector<vector> data;
	data.reserve(dataset.training_images.size());
	
	for (size_t d = 0; d < dataset_size; d++)
	{
		data.emplace_back(img_size);
		//(0,255) -> (-1,1)
		const auto& img = dataset.training_images[d];
		for (size_t i = 0; i < img.size(); i++)
		{
			data[d][i] = (((float)img[i] / 255.0f) - 0.5f) * 2;
		}
	}

	gn.train(data, 300);

	return 0;
}

/*************************************************************************************************************************************/

std::vector<std::pair<vector, vector>> preprocess(const std::vector<std::vector<uint8_t>>& data, const std::vector<uint8_t>& labels)
{
	std::cout << "preprocessing..." << std::endl;

	std::vector<std::pair<vector, vector>> t;
	t.reserve(data.size());

	for (size_t i = 0; i < data.size(); i++)
	{
		const auto& d = data[i];
		const uint8_t l = labels[i];

		vector dv(d.size());
		vector lv(10);

		for (size_t i = 0; i < d.size(); i++)
			dv[i] = (float)d[i] / 255.0f;

		lv[l] = 1.0f;

		t.push_back(std::make_pair(std::move(dv), std::move(lv)));
	}
	return std::move(t);
}

int main()
{
	//!!!!!!!!!!!!!!!!!
	return gan_main();

	std::cout << "Loading MNIST" << std::endl;
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	model classifier(
		28*28, {
			layer(100, activation::relu),
			layer(32, activation::relu),
			layer(10, activation::softmax)
		}
	);

	classifier.train(
		preprocess(dataset.training_images, dataset.training_labels),
		preprocess(dataset.test_images, dataset.test_labels),
		10,
		0.01f,
		5
	);

	return 0;
}

/*************************************************************************************************************************************/
