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

gan::gan(model& g, model& d) :
	_g(g), _d(d)
{
	assert(tensor_shape::equals(_g.output_shape(), _d.input_shape()));
	assert(_d.output_shape()[0] == 1);
}

void gan::train(const std::vector<trainer::data>& data, uint epochs, uint batch_size)
{
	_dc.set_mode(execution_mode::training);
	_dc.set_batches(batch_size);

	composite_model gan_model(_g, _d);

	tensor_layout<2> input_layout(batch_size,  _g.input_shape().total_size());
	tensor_layout<2> img_layout(batch_size,    _g.output_shape().total_size());
	tensor_layout<2> output_layout(batch_size, _d.output_shape().total_size());

	auto z_input    = _pool.alloc(input_layout);
	auto real_input = _pool.alloc(img_layout);
	auto y_d_1      = _pool.alloc(output_layout); _dc.fill(y_d_1, 0.9f);
	auto y_d_0      = _pool.alloc(output_layout); _dc.fill(y_d_0, 0.0f);
	auto y_g        = _pool.alloc(output_layout); _dc.fill(y_g,   1.0f);
	// testing batch
	auto z_test = _pool.alloc(input_layout);
	_dc.random_uniform(z_test);

	const float alpha = 0.0002f;
	const float beta = 0.5f;

	assert(data.size() % batch_size == 0);

	trainer d_trainer(_d, adam(alpha, beta), binary_cross_entropy());
	trainer g_trainer(_g, adam(alpha, beta), mse());

	std::vector<size_t> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);

	for (uint e = 0; e < epochs; e++)
	{
		thread_local auto rng = new_random_engine();

		std::cout << time_stamp << " epoch: " << e << std::endl;

		auto first = std::chrono::system_clock::now();
		auto last = first;

		std::shuffle(indices.begin(), indices.end(), rng);

		for (uint i = 0, iters = 0; i < indices.size(); i++, iters++)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << i << "/" << data.size() << ") ";
				std::cout << std::fixed << std::setprecision(3) << (100 * (float)i / data.size()) << "% | ";
				std::cout << iters << "it/s                              \r";
				last = t;
				iters = 0;
			}

			// update batch
			uint batch_index = i % batch_size;
			_dc.update(real_input[batch_index], data[indices[i]]);
			
			if (batch_index == 0 && i > 0)
			{
				_dc.clear_allocator();
				auto dy = _dc.alloc(output_layout);

				//randomize z
				_dc.random_uniform(z_input);

				//train discriminator on real batch
				d_trainer.train_batch(real_input, y_d_1, dy);

				//train discriminator on generated batch
				auto g_output = _g.forward(_dc, z_input.flatten()).reshape(img_layout);
				d_trainer.train_batch(g_output, y_d_0, dy);

				//randomize z
				_dc.random_uniform(z_input);

				auto dy2 = d_trainer.forward_backwards(_g.forward(_dc, z_input), y_g);
				g_trainer.train_gradient(dy2);
			}
		}

		std::cout << "(" << data.size() << "/" << data.size() << ") 100%                                       ";
		std::cout << std::endl;
		
		save_generated_images(e, z_test);
		_g.serialize(std::string("img/model-" + std::to_string(e) + ".bin"));
	}
}

/*************************************************************************************************************************************/

void gan::save_generated_images(uint id, const tensor<2>& z_batch)
{
	_dc.clear_allocator();

	const std::string filename = "img/g" + std::to_string(id) + ".bmp";

	const uint scale_factor = 16;
	const uint tile_wh = 28 * scale_factor;
	const uint border_sz = 24;
	const uint tile_count = 5;
	const uint total_wh = (tile_wh * tile_count) + (border_sz * (tile_count + 1));

	CImg<float> image(total_wh, total_wh);

	tensor_layout<2> img_layout(z_batch.shape(0), 28 * 28);
	std::vector<scalar> gen_image;
	auto g = _g.forward(_dc, z_batch).reshape(img_layout);
	_dc.read(g, gen_image);

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
