/*
	Neural network model
*/

#include "nn/model.h"

using namespace nn;

/***********************************************************************************************************************/

const buffer& model::forward(const buffer& x, bool is_training)
{
	auto a = std::cref(x);
	_activations.clear();
	_activations.push_back(a);

	for (auto& node : _nodes)
	{
		node->set_state(is_training);
		a = node->forward(a);
		_activations.push_back(a);
	}

	return _activations.back();
}

const buffer& model::backward(const buffer& dy, bool is_training)
{
	auto d = std::ref(dy);

	for (int i = (int)_nodes.size() - 1; i >= 0; i--)
	{
		auto node = _nodes[i].get();
		node->set_state(is_training);
		d = node->backward(_activations[i], d);
	}

	return d;
}

/***********************************************************************************************************************/
