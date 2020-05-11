/*
	Python interop


	We want to hijack some python functionality
*/

#include "python_interop.h"

#include <pybind11/numpy.h>

/*************************************************************************************************************************************/

Python::Python()
{
	const char code[] = R"(
import os
import time
import joblib
import sys
import numpy as np

graph = None
dataset = None
graph_file_path = "nearest_neighbours.z"

def generate_graph():
    from sklearn.neighbors import NearestNeighbors
    from keras.datasets import mnist
    global graph
    global dataset

    print("loading MNIST digits...")
    
    (xdata, _), (_, _) = mnist.load_data()
    #dataset = xdata[np.random.choice(xdata.shape[0], 20000, replace=False)]
    dataset = (xdata / 255).reshape(-1,28*28)

    print("generating nearest neighbours graph...")
    graph = NearestNeighbors(n_neighbors=1)
    t = time.time()
    graph.fit(dataset)
    joblib.dump(graph, graph_file_path)
    print("generating done", time.time() - t)

def load_graph():
    from keras.datasets import mnist
    global graph
    global dataset

    graph = joblib.load(graph_file_path)
    (xdata, _), (_, _) = mnist.load_data()
    dataset = (xdata / 255).reshape(-1,28*28)

if os.path.isfile(graph_file_path):
    load_graph()
else:
    generate_graph()

def get_nearest(data):
    #np.save(data, "testg.npz")
    #print("get_nearest:", data.shape, data.max(), data.min())
    d, idx = graph.kneighbors(data)
    return dataset[idx[0][0]]
)";
	std::cout << "Starting python interpreter..." << std::endl;
	try
	{
		py::exec(code);// py::eval_file("F:\\dev\\project\\libnn\\scripts\\mnist_generate_nearest_neighbours.py");

		auto lookup = py::globals().attr("get");
		get_nearest = lookup("get_nearest").cast<py::function>();

		std::cout << "Python done." << std::endl;
		_ok = true;
	}
	catch (std::exception e)
	{
		std::cout << "Python Error: " << e.what() << std::endl;
	}
}


void Python::nearest_neighbour(const ImageOutput& img, ImageOutput& nearest)
{
	if (_ok)
	{
		try
		{
			py::array_t<float> x({ 1, 28 * 28 }, img.data());
			auto result = get_nearest(x).cast<py::array_t<float>>();

			std::copy(result.data(), result.data() + result.size(), nearest.begin());

			//std::cout << (std::string)result.str() << std::endl;
		}
		catch (std::exception e)
		{
			std::cout << "Python Error: " << e.what() << std::endl;
		}
	}
}

/*************************************************************************************************************************************/
