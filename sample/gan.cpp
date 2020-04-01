/*
	GAN
*/

#include <string>

#include "CImg.h"
#include "gan.h"

using namespace nn;
using namespace cimg_library;

/*************************************************************************************************************************************/

void gan::train(const std::vector<buffer>& data, uint epochs)
{
	tensor<1> z_input(_g.input_shape());
	tensor<1> g_value(28 * 28);
	tensor<1> y_d_1(1); y_d_1[0] = 0.9f;
	tensor<1> y_d_0(1); y_d_0[0] = 0;
	tensor<1> y_g(1); y_g[0] = 1;

	const float k = 0.001f;
	//const float r = 1 - (k * 5 / data.size());
	const float r = 1;

	std::uniform_real_distribution<double> unif(-1, 1);

	for (uint e = 0; e < epochs; e++)
	{
		auto rng = new_random_engine();

		std::cout << time_stamp << " epoch: " << e << std::endl;

		auto first = std::chrono::system_clock::now();
		auto last = first;
		uint c = 0;

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
			for (uint i_z = 0; i_z < z_input.shape(0); i_z++)
				z_input[i_z] = (scalar)unif(rng);

			_d.train_batch(x_data, y_d_1.data());
			_d.train_batch(_g.forward(z_input.data()), y_d_0.data());

			//randomize z
			for (uint i_z = 0; i_z < z_input.shape(0); i_z++)
				z_input[i_z] = (scalar)unif(rng);

			const auto& dy = _d.forward_backwards(_g.forward(z_input.data()), y_g.data());
			_g.train_from_gradient(dy);
		}

		std::cout << "(" << c << "/" << data.size() << ") 100%";
		std::cout << std::endl;
		
		save_generated_images(e);
		_g.serialize(std::string("img/model-" + std::to_string(e) + ".bin"));
	}
}

/*************************************************************************************************************************************/

void gan::save_generated_images(uint id)
{
	const std::string filename = "img/g" + std::to_string(id) + ".bmp";
	tensor<1> z_test(_g.input_shape());

	const uint scale_factor = 16;
	const uint tile_wh = 28 * scale_factor;
	const uint border_sz = 24;
	const uint tile_count = 5;
	const uint total_wh = (tile_wh * tile_count) + (border_sz * (tile_count + 1));

	CImg<float> image(total_wh, total_wh);

	for (uint y_tile = 0; y_tile < tile_count; y_tile++)
	{
		for (uint x_tile = 0; x_tile < tile_count; x_tile++)
		{
			for (uint i_z = 0; i_z < z_test.shape(0); i_z++)
				z_test[i_z] = 2 * ((float)(y_tile * tile_count + x_tile) / (tile_count * tile_count)) - 1;

			auto g = _g.forward(z_test.data()).as_vector();

			const uint x = (x_tile * tile_wh) + ((x_tile + 1) * border_sz);
			const uint y = (y_tile * tile_wh) + ((y_tile + 1) * border_sz);

			for (uint i_g = 0; i_g < g.shape(0); i_g++)
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
					) = ((g[i_g] + 1) / 2) * 255;
				}
			}
		}
	}

	image.save(filename.c_str());
	std::cout << time_stamp << " saved: " << filename << std::endl;
}

/*************************************************************************************************************************************/
