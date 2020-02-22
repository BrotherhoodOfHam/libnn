/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"
#include "nn/model.h"
#include "gan.h"

#include "nn/tensors.h"

using namespace nn;

/*************************************************************************************************************************************/

struct dataset
{
	std::vector<tensor> x_train, y_train, x_test, y_test;
};

void process_labels(std::vector<tensor>& labels, uint8_t label, uint8_t number_of_classes)
{
	labels.emplace_back(number_of_classes);
	tensor& _label = labels.back();
	for (size_t i = 0; i < _label.shape(0); i++)
		_label(i) = 0.0f;
	_label(label) = 1.0f;
}

void process_images(std::vector<tensor>& data, const std::vector<uint8_t>& image)
{
	data.emplace_back(image.size());
	tensor& _data = data.back();
	for (size_t i = 0; i < image.size(); i++)
		_data(i) = (float)image[i] / 255.0f;
}

dataset load_mnist()
{
	using namespace std::chrono;
	using clock = std::chrono::high_resolution_clock;

	std::cout << "Loading MNIST" << std::endl;
	auto t = clock::now();

	dataset d;

	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();
	d.x_train.reserve(dataset.training_images.size());
	d.y_train.reserve(dataset.training_labels.size());
	d.x_test.reserve(dataset.test_images.size());
	d.y_test.reserve(dataset.test_labels.size());

	for (auto& img : dataset.training_images)
		process_images(d.x_train, img);
	for (auto& label : dataset.training_labels)
		process_labels(d.y_train, label, 10);

	for (auto& img : dataset.test_images)
		process_images(d.x_test, img);
	for (auto& label : dataset.test_labels)
		process_labels(d.y_test, label, 10);

	std::cout << "Loaded: " << duration_cast<milliseconds>(clock::now() - t).count() << "ms" << std::endl;

	return d;
}

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
			layer(1024, activation::leaky_relu),
			layer(img_size, activation::tanh),
		},
		0.01f
	);
	model d(
		img_size,
		{
			layer(1024, activation::leaky_relu, 0.1f, 0.3f),
			layer(512, activation::leaky_relu, 0.1f, 0.3f),
			layer(256, activation::leaky_relu, 0.1f, 0.3f),
			layer(1, activation::sigmoid),
		},
		0.01f
	);

	gan gn(&g, &d);

	//prepare data
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	const size_t dataset_size = dataset.training_images.size();
	std::vector<tensor> data;
	data.reserve(dataset.training_images.size());
	
	for (size_t d = 0; d < dataset_size; d++)
	{
		data.emplace_back(img_size);
		//(0,255) -> (-1,1)
		const auto& img = dataset.training_images[d];
		for (size_t i = 0; i < img.size(); i++)
		{
			data[d](i) = (((float)img[i] / 255.0f) - 0.5f) * 2;
		}
	}

	gn.train(data, 300);

	return 0;
}

/*************************************************************************************************************************************/

int main()
{
	//!!!!!!!!!!!!!!!!!
	return gan_main();

	dataset ds = load_mnist();

	model classifier(
		28*28, {
			layer(100, activation::relu, 0.1f, 0.2f),
			layer(32, activation::relu),
			layer(10, activation::softmax)
		},
		0.01f
	);

	classifier.train(
		ds.x_train, ds.y_train, ds.x_test, ds.y_test,
		10
	);

	classifier.serialize("classifier.bin");

	return 0;
}

/*************************************************************************************************************************************/
