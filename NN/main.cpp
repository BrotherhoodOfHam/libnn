#include <vector>
#include <iostream>

#include "mnist/mnist_reader_less.hpp"
#include "nn.h"

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
	std::cout << "Loading MNIST" << std::endl;
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	nn n(784, {
		//layer(100, activation::relu),
		layer(32, activation::relu),
		layer(10, activation::softmax)
		});
	n.train(
		preprocess(dataset.training_images, dataset.training_labels),
		preprocess(dataset.test_images, dataset.test_labels),
		10,
		0.01f,
		5
	);

	return 0;
}

/*************************************************************************************************************************************/
