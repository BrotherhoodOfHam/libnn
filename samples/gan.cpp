/*
	GAN
*/

#include <string>

#include "CImg.h"
#include "gan.h"
#include "nn/training.h"

using namespace nn;
using namespace cimg_library;

/*************************************************************************************************************************************/

GAN::GAN(model& g, model& d) :
	_g(g), _d(d)
{
	assert(tensor_shape::equals(_g.output_shape(), _d.input_shape()));
	assert(_d.output_shape()[0] == 1);
}

void GAN::train(const std::vector<trainer::data>& data, uint epochs, uint batch_size)
{
	auto& dev = device::get();

	// Construct a composite model
	auto gan = _g.compose(_d.immutable());

	adam opt(0.0002f, 0.5f);
	binary_cross_entropy loss;

	trainer gopt(gan, opt, loss);
	trainer dopt(_d,  opt, loss);

	tensor_layout<2> input_layout(batch_size,  _g.input_shape().datasize());
	tensor_layout<2> image_layout(batch_size,  _g.output_shape().datasize());
	tensor_layout<2> output_layout(batch_size, _d.output_shape().datasize());

	auto z_input    = _constants.alloc(input_layout);
	auto real_input = _constants.alloc(image_layout);
	auto y_d_1      = _constants.alloc(output_layout); dev.fill(y_d_1, 0.9f);
	auto y_d_0      = _constants.alloc(output_layout); dev.fill(y_d_0, 0.0f);
	auto y_g        = _constants.alloc(output_layout); dev.fill(y_g,   1.0f);
	// testing batch
	auto z_test = _constants.alloc(input_layout);
	dev.random_uniform(z_test);

	assert(data.size() % batch_size == 0);

	auto rng = new_random_engine();
	std::vector<size_t> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);

	for (uint e = 0; e < epochs; e++)
	{
		std::cout << time_stamp << " epoch: " << e << std::endl;

		std::shuffle(indices.begin(), indices.end(), rng);

		progress_printer progress(data.size() / batch_size);

		for (uint i = 0; i < indices.size(); i++)
		{
			// update batch
			uint batch_index = i % batch_size;
			dev.update(real_input[batch_index], data[indices[i]]);
			
			if (batch_index == 0 && i > 0)
			{
				progress.next();
				auto dc = dev.scope(execution_mode::training, batch_size);

				//randomize z
				dc.random_uniform(z_input);

				//train discriminator on real batch
				dopt.train_batch(dc, real_input, y_d_1);

				//train discriminator on generated batch
				auto g_output = _g.forward(dc, z_input).reshape(image_layout);
				dopt.train_batch(dc, g_output, y_d_0);

				//randomize z
				dc.random_uniform(z_input);

				//train the generator
				gopt.train_batch(dc, z_input, y_g);
			}
		}

		progress.stop();

		save_generated_images(e, z_test);
		_g.serialize(std::string("img/model-" + std::to_string(e) + ".bin"));
	}
}

/*************************************************************************************************************************************/

void GAN::save_generated_images(uint id, const batch& z_batch)
{
	const std::string filename = "img/g" + std::to_string(id) + ".bmp";

	const uint scale_factor = 16;
	const uint tile_wh = 28 * scale_factor;
	const uint border_sz = 24;
	const uint tile_count = 5;
	const uint total_wh = (tile_wh * tile_count) + (border_sz * (tile_count + 1));

	CImg<float> image(total_wh, total_wh);

	tensor_layout<2> img_layout(z_batch.shape(0), 28 * 28);
	std::vector<scalar> gen_image;

	auto dc = device::get().scope(execution_mode::execute, z_batch.shape(0));

	auto g = _g.forward(dc, z_batch).reshape(img_layout);
	dc.read(g, gen_image);

	uint batch_index = 0;
	for (uint y_tile = 0; y_tile < tile_count; y_tile++)
	{
		for (uint x_tile = 0; x_tile < tile_count; x_tile++)
		{
			const uint x = (x_tile * tile_wh) + ((x_tile + 1) * border_sz);
			const uint y = (y_tile * tile_wh) + ((y_tile + 1) * border_sz);

			for (uint i_g = 0; i_g < g.shape(1); i_g++)
			{
				const uint w = i_g % 28;
				const uint h = i_g / 28;

				for (uint i_sub = 0; i_sub < (scale_factor * scale_factor); i_sub++)
				{
					const uint sub_w = i_sub % scale_factor;
					const uint sub_h = i_sub / scale_factor;

					image(
						x + (w * scale_factor) + sub_w,
						y + (h * scale_factor) + sub_h
					) = ((gen_image[(size_t)batch_index*g.shape(1) + i_g] + 1) / 2) * 255;
				}
			}

			batch_index++;
		}
	}

	image.save(filename.c_str());
	std::cout << time_stamp << " saved: " << filename << std::endl;
}

/*************************************************************************************************************************************/
