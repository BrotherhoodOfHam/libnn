/*
	Generators
*/

#pragma once

#include <array>
#include <nn/model.h>

/*************************************************************************************************************************************/

const size_t LATENT_SIZE = 10;
const size_t IMAGE_SIZE = 28 * 28;

using LatentVector = std::array<float, LATENT_SIZE>;
using ImageOutput = std::array<float, IMAGE_SIZE>;

class Generator
{
	nn::model   _G;

public:

	Generator();

	void generate(const LatentVector& z, ImageOutput& img);
};

/*************************************************************************************************************************************/
