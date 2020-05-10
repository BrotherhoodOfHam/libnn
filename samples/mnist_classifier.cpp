/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"

#include "nn/node/activations.h"
#include "nn/node/dense.h"
#include "nn/node/dropout.h"
#include "nn/node/debug.h"
#include "nn/training.h"

using namespace nn;

/*************************************************************************************************************************************/

struct ds
{
	std::vector<trainer::data> x_train, x_test;
	std::vector<trainer::label> y_train, y_test;
};

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

ds load_mnist()
{
	using namespace std::chrono;
	using clock = std::chrono::high_resolution_clock;

	std::cout << "Loading MNIST" << std::endl;
	auto t = clock::now();

	ds d;

	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();
	d.x_train.reserve(dataset.training_images.size());
	d.y_train.reserve(dataset.training_labels.size());
	d.x_test.reserve(dataset.test_images.size());
	d.y_test.reserve(dataset.test_labels.size());

	for (auto& img : dataset.training_images)
		process_image(d.x_train, img);
	for (auto& label : dataset.training_labels)
		d.y_train.push_back(label);

	for (auto& img : dataset.test_images)
		process_image(d.x_test, img);
	for (auto& label : dataset.test_labels)
		d.y_test.push_back(label);

	std::cout << "Loaded: " << duration_cast<milliseconds>(clock::now() - t).count() << "ms" << std::endl;

	return std::move(d);
}

/*************************************************************************************************************************************/

int main()
{
	ds ds = load_mnist();

	model classifier(28*28);
	classifier.add<dense_layer>(100);
	classifier.add<activation::relu>();
	classifier.add<dropout>(0.2f);
	classifier.add<dense_layer>(32);
	classifier.add<activation::relu>();
	classifier.add<dense_layer>(10);
	classifier.add<activation::softmax>();

	trainer t(classifier, adam());
	t.train(
		ds.x_train, ds.y_train, ds.x_test, ds.y_test,
		15, 100
	);

	classifier.serialize("classifier.bin");

	return 0;
}

/*************************************************************************************************************************************/
