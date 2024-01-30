Mini MLP for GPU or CPU

This is a Multi Layered Perecptron with two Activation functions - Tanh or Sigmoid.

It is programmed to run on the GPU with a compute as low as 5.0 (My current laptop!)

It is quick with configured for 8 hidden neurons and using Tanh it solves the XoR in 200 iterations.

This is my first GPU NN.

nvcc -arch=sm_50 -rdc=true -o miniml mini_mlp.cu

Enjoy. I thought I would share it as there are few MLP CUDA C++ implementations.