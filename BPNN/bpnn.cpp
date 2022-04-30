#include "bpnn.h"

BPNeuralNetwork::BPNeuralNetwork()
{
    srand((unsigned)time(NULL)); // 随机数种子
    error = 100.f;

    for (int i = 0; i < input_nodes; ++i)
    {
        input_layer[i] = new InputNode();
        for (int j = 0; j < hidden_nodes; ++j)
        {
            input_layer[i]->weight.push_back(get_random_num());
            input_layer[i]->w_delta_sum.push_back(0.f);
        }
    }

    for (int i = 0; i < hidden_layers; i++)
    {
        if (i == hidden_layers - 1)
        {
            /* code */
        }
        
    }
    

}

void BPNeuralNetwork::forwardPropagation() {}

void BPNeuralNetwork::backwardPropagation() {}

void BPNeuralNetwork::train() {}

void BPNeuralNetwork::predict() {}
