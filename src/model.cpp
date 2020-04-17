/*
	Neural network model
*/

#include "nn/model.h"

using namespace nn;

/***********************************************************************************************************************/

vector model::forward(context& dc, const vector& x)
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

vector model::backward(context& dc, const vector& dy)
{
	vector dv = dy;

	for (int i = (int)_nodes.size() - 1; i >= 0; i--)
	{
		auto node = _nodes[i].get();
		dv = node->backward(dc, _activations[i], dv);
	}

	return dv;
}

/***********************************************************************************************************************/
