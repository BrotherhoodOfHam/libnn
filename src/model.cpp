/*
	Neural network model
*/

#include "nn/model.h"

using namespace nn;

/***********************************************************************************************************************/

vector model::forward(scope& dc, const vector& x)
{
	_activations.clear();
	_activations.push_back(x);

	for (auto& node : _nodes)
	{
		const vector& y = node->forward(dc, _activations.back());
		_activations.push_back(y);
	}

	return _activations.back();
}

vector model::backward(scope& dc, const vector& dy)
{
	if (_activations.empty())
	{
		throw std::runtime_error("A forward pass must be made before performing back propagation. Call model::forward() first");
	}

	vector dv = dy;

	for (int i = (int)_nodes.size() - 1; i >= 0; i--)
	{
		auto node = _nodes[i].get();
		dv = node->backward(dc, _activations[i], dv);
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
