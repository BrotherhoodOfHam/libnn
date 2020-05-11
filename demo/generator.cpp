/*
	Libnn based generator
*/

#include "generator.h"

#include <QFile>
#include <sstream>

#include "nn/model.h"
#include "nn/node/activations.h"
#include "nn/node/dense.h"
#include "nn/node/dropout.h"

/*************************************************************************************************************************************/

static void load_model_parameters(nn::model& m, const QString& file_path)
{
	QFile f(file_path);
	if (!f.open(QIODevice::ReadOnly))
		throw std::runtime_error((QStringLiteral("could not open ") + file_path).toStdString());

	auto buf = f.readAll();

	std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
	ss.write(buf.begin(), buf.size());
	assert(ss.good());
	m.deserialize(ss);

	std::cout << "Loaded model" << file_path.toStdString() << std::endl;
}

static nn::model load_discriminator()
{
	using namespace nn;

	uint z_size = 10;
	uint img_size = 28 * 28;

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

	load_model_parameters(d, ":/discriminator.bin");

	return d;
}

static nn::model load_generator()
{
	using namespace nn;

	uint z_size = 10;
	uint img_size = 28 * 28;

	model g(z_size);
	g.add<dense_layer>(256);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(512);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(1024);
	g.add<activation::leaky_relu>(0.2f);
	g.add<dense_layer>(img_size);
	g.add<activation::tanh>();

	load_model_parameters(g, ":/generator.bin");

	return g;
}

static nn::model load_classifier()
{
	using namespace nn;

	model classifier(28 * 28);
	classifier.add<dense_layer>(100);
	classifier.add<activation::relu>();
	classifier.add<dropout>(0.2f);
	classifier.add<dense_layer>(32);
	classifier.add<activation::relu>();
	classifier.add<dense_layer>(10);
	classifier.add<activation::softmax>();

	load_model_parameters(classifier, ":/classifier.bin");

	return classifier;
}

/*************************************************************************************************************************************/

Generator::Generator() :
	_G(load_generator())//, _D(load_discriminator())
{}

void Generator::generate(const LatentVector& z, ImageOutput& img)
{
	using namespace nn;

	std::vector<float> _ybuf;
	std::vector<float> _isrealbuf;

	{
		// prepare input tensor
		auto dc = device::begin();
		auto dev_z = dc.alloc(1, 10);
		dc.update(dev_z, z);

		// generate image
		auto dev_y = _G.forward(dc, dev_z);
		//auto dev_isreal = _D.forward(dc, dev_y);

		// copy output back to host
		dc.read(dev_y, _ybuf);
		//dc.read(dev_isreal, _isrealbuf);
	}

	//[-1..1] -> [0..1]
	std::transform(_ybuf.begin(), _ybuf.end(), img.begin(), [](float i) { return (i + 1) / 2; });
}

/*************************************************************************************************************************************/
