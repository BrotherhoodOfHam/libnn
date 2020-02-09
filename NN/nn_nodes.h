#pragma once

#include "nn.h"

/*************************************************************************************************************************************/

class activation_node : public nn::node
{
	activation _type;
	bool _isoutput;

public:

	activation_node(size_t input_size, activation type, bool is_output) :
		_type(type), _isoutput(is_output),
		node(input_size, input_size)
	{}

	void forward(const vector& x, vector& y) const override;

	void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const override;

	void update_params(const matrix& dw, const vector& db, float k, float r) override { }
};

class layer_node : public nn::node
{
	matrix w;
	vector b;

public:

	layer_node(size_t input_size, size_t layer_size);

	void forward(const vector& x, vector& y) const override;

	void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const override;

	void update_params(const matrix& dw, const vector& db, float k, float r) override;
};

/*************************************************************************************************************************************/
