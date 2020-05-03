/*
	Neural network model
*/

#include "nn/model.h"

using namespace nn;

/***********************************************************************************************************************/

batch model::execute(const batch& x)
{
	auto scope = device::begin(execution_mode::execute, x.shape(0));
	auto y = forward(scope, x);
	_activations.clear();
	return y;
}

batch model::forward(scope& dc, const batch& x)
{
	_activations.clear();
	_activations.push_back(x);

	for (auto& node : _nodes)
	{
		const batch& y = node->forward(dc, _activations.back());
		_activations.push_back(y);
	}

	return _activations.back();
}

batch model::backward(scope& dc, const batch& dy)
{
	if (_activations.empty())
	{
		throw std::runtime_error("A forward pass must be made before performing back propagation. Call model::forward() first");
	}

	batch dv = dy;

	for (long i = (long)_nodes.size() - 1; i >= 0; i--)
	{
		dv = _nodes[i]->backward(dc, _activations[i], _activations[i + 1], dv);
	}

	return dv;
}

/***********************************************************************************************************************/

model model::immutable() const
{
	model immutable_this(input_shape());
	immutable_this._nodes = this->_nodes;
	return immutable_this;
}

model model::compose(model& next) const
{
	if (!tensor_shape::equals(output_shape(), next.input_shape()))
		throw std::runtime_error("cannot compose models with mismatched shape");

	model composed(input_shape());
	
	for (const auto& n : _nodes)
		composed._nodes.push_back(n);
	for (const auto& p : _parameters)
		composed._parameters.push_back(p);

	for (const auto& n : next._nodes)
		composed._nodes.push_back(n);
	for (const auto& p : next._parameters)
		composed._parameters.push_back(p);

	return composed;
}

/***********************************************************************************************************************/
