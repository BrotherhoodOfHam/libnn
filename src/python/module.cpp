/*
    Python bindings

    Only a limited subset of the functionality is supported.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <nn/model.h>
#include <nn/training.h>
#include <nn/node/activations.h>
#include <nn/node/dense.h>
#include <nn/node/dropout.h>

/*************************************************************************************************************************************/

namespace py = pybind11;

struct py_node
{
    std::function<void(nn::model&)> build;
};

class py_model : public nn::model
{
public:

    py_model(nn::uint input_size) : model(input_size) {}

    void add(py_node& nb) { nb.build(*this); }

    py::array exec_forward(py::array x)
    {
        if (x.ndim() != 2)
            throw std::runtime_error("Dimensionality must be a maximum of 2");
        if (x.shape(1) != _input_shape[0])
            throw std::runtime_error("Incorrect input shape");

        auto dc = nn::device::begin(nn::execution_mode::execute);
        auto dc_x = dc.alloc((nn::uint)x.shape(0), (nn::uint)x.shape(1));
        dc.update(dc_x, nn::const_span<nn::scalar>((const nn::scalar*)x.data(), (const nn::scalar*)x.data() + x.size()));
        auto dc_y = forward(dc, dc_x);

        std::vector<nn::scalar> temp;
        dc.read(dc_y, temp);

        auto y_buf = new nn::scalar[temp.size()];
        std::copy(temp.begin(), temp.end(), y_buf);
        py::capsule owner(y_buf, [](void* ptr){
            auto p = (nn::scalar*)ptr;
            if (p != nullptr)
            {
                delete[] p;
            }
        });
        return py::array_t<nn::scalar>({ dc_y.shape(0), dc_y.shape(1) }, y_buf, owner);
    }

    void summary()
    {
        py::print("abc", parameters().size());
    }
};


template<class node_type>
struct py_nodewrap : public py_node
{
    template<typename ... args_type>
    py_nodewrap(args_type ... args)
    {
        build = [=](nn::model& m) {
            m.add<node_type>(args...);
        };
    }
};

template<class node_type>
inline py::class_<py_nodewrap<node_type>, py_node> export_node(py::module& m, const char* name)
{
    return py::class_<py_nodewrap<node_type>, py_node>(m, name);
}

/*************************************************************************************************************************************/

PYBIND11_MODULE(libnn, m)
{
    m.doc() = "python bindings for libnn library";

    py::class_<py_node> node(m, "node");
    node.def_readonly("build", &py_node::build);

    // model
    py::class_<py_model> mdl(m, "model");
    mdl.def(py::init<nn::uint>(), py::arg("input_size"));
    mdl.def("add", &py_model::add);
    mdl.def("forward", &py_model::exec_forward);
    //mdl.def("backward", &py_model::exec_backward);
    mdl.def("__call__", &py_model::exec_forward);
    mdl.def("load_weights", (void(py_model::*)(const std::string&))&py_model::deserialize);
    mdl.def("summary", &py_model::summary);

    /*
        Export standard nodes
    */

    export_node<nn::dense_layer>(m, "dense")
        .def(py::init<nn::uint>(), py::arg("size"));
    
    export_node<nn::dropout>(m, "dropout")
        .def(py::init<float>(), py::arg("p"));
    
    export_node<nn::activation::leaky_relu>(m, "leaky_relu")
        .def(py::init<float>(), py::arg("k"));

    export_node<nn::activation::relu>(m, "relu")
        .def(py::init<>());

    export_node<nn::activation::sigmoid>(m, "sigmoid")
        .def(py::init<>());

    export_node<nn::activation::tanh>(m, "tanh")
        .def(py::init<>());

    export_node<nn::activation::softmax>(m, "softmax")
        .def(py::init<>());
}

/*************************************************************************************************************************************/
