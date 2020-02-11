/*
	Base header for nodes
*/

#pragma once

#include "base.h"

namespace nn
{
	namespace nodes
	{
		enum class node_type : uint8_t
		{
			simple = 0,
			parametric = 1,
		};

		// Base class for nodes. You should derive from either node/parametric_node
		class node_base
		{
		private:

			node_type _type;
			size_t _input_size;
			size_t _output_size;

		public:

			node_base(size_t input_size, size_t output_size, node_type type) :
				_input_size(input_size), _output_size(output_size), _type(type)
			{}

			node_base(const node_base&) = delete;

			inline node_type type() const { return _type; }
			inline size_t input_size() const { return _input_size; }
			inline size_t output_size() const { return _output_size; }

			virtual void forward(const vector& x, vector& y) const = 0;
		};

		// Simple node with no learnable parameters
		class node : public node_base
		{
		public:

			node(size_t input_size, size_t output_size) :
				node_base(input_size, output_size, node_type::simple)
			{}

			virtual void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const = 0;
		};

		// Node with learnable weight and bias parameters
		class parametric_node : public node_base
		{
		public:

			parametric_node(size_t input_size, size_t output_size) :
				node_base(input_size, output_size, node_type::parametric)
			{}

			virtual void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const = 0;

			virtual void update_params(const matrix& dw, const vector& db, float r, float k) = 0;
		};
	}
}