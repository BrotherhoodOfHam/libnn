
#include <vector>

#include "nn/model.h"
#include "nn/ops/dense.h"
#include "nn/ops/activations.h"
#include "nn/training.h"

int main()
{
	using namespace nn;

	model xor(2);
	xor.add<dense_layer>(4);
	xor.add<activation::sigmoid>();
	xor.add<dense_layer>(1);
	xor.add<activation::sigmoid>();
	trainer t0(xor, adam());

	std::vector<trainer::data> xdata =
	{
		{ 1, 1 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 0 },
	};
	std::vector<trainer::label> ydata =
	{
		0, 1, 1, 0
	};

	t0.train(xdata, ydata, xdata, ydata, 10, 4);

	return 0;
}