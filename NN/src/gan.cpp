/*
	GAN
*/

#include <string>

#include "CImg.h"
#include "gan.h"

using namespace nn;
using namespace cimg_library;

/*************************************************************************************************************************************/

void gan::train(const std::vector<tensor>& data, size_t epochs)
{
	tensor z_input(_g->input_size());
	tensor g_value(28 * 28);
	tensor y_d_1(1); y_d_1(0) = 0.9f;
	tensor y_d_0(1); y_d_0(0) = 0;
	tensor y_g(1); y_g(0) = 1;

	const float k = 0.001f;
	//const float r = 1 - (k * 5 / data.size());
	const float r = 1;

	std::uniform_real_distribution<double> unif(-1, 1);

	for (size_t e = 0; e < epochs; e++)
	{
		std::default_random_engine rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		std::cout << time_stamp << " epoch: " << e << std::endl;

		auto first = std::chrono::system_clock::now();
		auto last = first;
		size_t c = 0;

		for (const auto& x_data : data)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << c << "/" << data.size() << ") ";
				std::cout << std::fixed << std::setprecision(3) << (float)c / data.size() << "%          " << "\r";
				last = t;
			}
			c++;

			//randomize z
			for (size_t i_z = 0; i_z < z_input.shape(0); i_z++)
				z_input(i_z) = (scalar)unif(rng);

			_d->train_batch(x_data, y_d_1);
			_d->train_batch(_g->forward(z_input), y_d_0);

			//randomize z
			for (size_t i_z = 0; i_z < z_input.shape(0); i_z++)
				z_input(i_z) = (scalar)unif(rng);

			const auto& dy = _d->forward_backwards(_g->forward(z_input), y_g);
			_g->train_from_gradient(dy);
		}

		std::cout << "(" << c << "/" << data.size() << ") 100%";
		std::cout << std::endl;
		
		save_generated_images(e);
		_g->serialize(std::string("img/model-" + std::to_string(e) + ".bin"));
	}
}

/*************************************************************************************************************************************/

void gan::save_generated_images(size_t id)
{
	const std::string filename = "img/g" + std::to_string(id) + ".bmp";
	tensor z_test(_g->input_size());

	const size_t scale_factor = 16;
	const size_t tile_wh = 28 * scale_factor;
	const size_t border_sz = 24;
	const size_t tile_count = 5;
	const size_t total_wh = (tile_wh * tile_count) + (border_sz * (tile_count + 1));

	CImg<float> image(total_wh, total_wh);

	for (size_t y_tile = 0; y_tile < tile_count; y_tile++)
	{
		for (size_t x_tile = 0; x_tile < tile_count; x_tile++)
		{
			for (size_t i_z = 0; i_z < z_test.shape(0); i_z++)
				z_test(i_z) = 2 * ((float)(y_tile * tile_count + x_tile) / (tile_count * tile_count)) - 1;

			const tensor& g = _g->forward(z_test);

			const size_t x = (x_tile * tile_wh) + ((x_tile + 1) * border_sz);
			const size_t y = (y_tile * tile_wh) + ((y_tile + 1) * border_sz);

			for (size_t i_g = 0; i_g < g.shape(0); i_g++)
			{
				const size_t w = i_g % 28;
				const size_t h = i_g / 28;

				for (size_t i_sub = 0; i_sub < (scale_factor * scale_factor); i_sub++)
				{
					const size_t sub_w = i_sub % scale_factor;
					const size_t sub_h = i_sub / scale_factor;

					image(
						x + (w * scale_factor) + sub_w,
						y + (h * scale_factor) + sub_h
					) = ((g(i_g) + 1) / 2) * 255;
				}
			}
		}
	}

	image.save(filename.c_str());
	std::cout << time_stamp << " saved: " << filename << std::endl;
}

/*************************************************************************************************************************************/
