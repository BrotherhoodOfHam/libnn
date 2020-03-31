/*
	Program entrypoint
*/

#include "mnist/mnist_reader_less.hpp"
#include "nn/model.h"
#include "nn/activations.h"
#include "nn/dense_layer.h"
#include "nn/dropout.h"
#include "gan.h"

#include "nn/tensors.h"

using namespace nn;

/*************************************************************************************************************************************/

struct dataset
{
	std::vector<model::data> x_train, x_test;
	std::vector<model::label> y_train, y_test;
};

void process_image(std::vector<model::data>& dataset, const std::vector<uint8_t>& image)
{
	auto& data = dataset.emplace_back(image.size());
	for (size_t i = 0; i < data.size(); i++)
		data[i] = (float)image[i] / 255.0f;
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

int gan_main()
{
	size_t z_size = 10;
	size_t img_size = 28 * 28;

	model g(z_size, 1, 0.01f);
	g.add<dense_layer>(256);
	g.add<activation::leaky_relu>(0.1f);
	g.add<dense_layer>(512);
	g.add<activation::leaky_relu>(0.1f);
	g.add<dense_layer>(1024);
	g.add<activation::leaky_relu>(0.1f);
	g.add<dense_layer>(img_size);
	g.add<activation::tanh>();

	model d(img_size, 1, 0.01f);
	d.add<dense_layer>(1024);
	d.add<activation::leaky_relu>(0.1f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(512);
	d.add<activation::leaky_relu>(0.1f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(256);
	d.add<activation::leaky_relu>(0.1f);
	d.add<dropout>(0.3f);
	d.add<dense_layer>(1);
	d.add<activation::sigmoid>();

	gan gn(g, d);

	//prepare data
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	const size_t dataset_size = dataset.training_images.size();
	std::vector<buffer> data;
	data.reserve(dataset_size);
	
	for (size_t d = 0; d < dataset_size; d++)
	{
		buffer& dta = data.emplace_back(img_size);
		//(0,255) -> (-1,1)
		const auto& img = dataset.training_images[d];
		for (size_t i = 0; i < img.size(); i++)
		{
			dta.ptr()[i] = (((float)img[i] / 255.0f) - 0.5f) * 2;
		}
	}

	gn.train(data, 300);

	return 0;
}

/*************************************************************************************************************************************/

int main()
{
	//!!!!!!!!!!!!!!!!!
	//return gan_main();

	dataset ds = load_mnist();

	model classifier(28*28, 100, 0.01f);
	classifier.add<dense_layer>(100);
	classifier.add<activation::relu>();
	classifier.add<dropout>(0.1f);
	classifier.add<dense_layer>(32);
	classifier.add<activation::relu>();
	classifier.add<dense_layer>(10);
	classifier.add<activation::softmax>();

	classifier.train(
		ds.x_train, ds.y_train, ds.x_test, ds.y_test,
		10
	);

	classifier.serialize("classifier.bin");

	return 0;
}

/*************************************************************************************************************************************/
