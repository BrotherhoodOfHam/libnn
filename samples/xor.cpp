
#include <vector>

#include "nn/model.h"
#include "nn/node/dense.h"
#include "nn/node/activations.h"
#include "nn/training.h"

int main()
{
	using namespace nn;

	model xor(2);
	xor.add<dense_layer>(2);
	xor.add<activation::sigmoid>();
	xor.add<dense_layer>(1);
	xor.add<activation::sigmoid>();
	trainer t(xor, sgd(), binary_cross_entropy());

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

	t.train(xdata, ydata, xdata, ydata, 10, 4);

	return 0;
}