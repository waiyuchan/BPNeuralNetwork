#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using std::vector;

#define input_nodes 6     // 输入层节点数
#define hidden_nodes 3    // 隐含层节点数
#define hidden_layers 1   // 隐含层数量
#define output_nodes 1    // 输出节点数量
#define learning_rate 0.1 // 学习速率, alpha

// 产生 -1~1 之间的随机数
double get_random_num()
{
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

// 损失函数
double sigmoid(double x)
{
    double ans = 1 / (1 + exp(-x));
    return ans;
}

typedef struct InputNode
{
    double value;
    vector<double> weight;
    vector<double> w_delta_sum;
} InputNode;

typedef struct HiddenNode
{
    double value;
    double delta;
    double bias;
    double b_delta_sum;
    vector<double> weight;
    vector<double> w_delta_sum;
} HiddenNode;

typedef struct OutputNode
{
    double value;
    double delta;
    double right_out;
    double bias;
    double b_delta_sum;
} OutputNode;

typedef struct sample
{
    vector<double> input, output;
} sample;

class BPNeuralNetwork
{
public:
    BPNeuralNetwork();
    void forwardPropagation();
    void backwardPropagation();

    void train();
    void predict();

    void setInput(vector<double> sample_data_in);
    void setOutput(vector<double> sample_data_out);

    double error;
    InputNode *input_layer[input_nodes];
    OutputNode *output_layer[output_nodes];
    HiddenNode *hidden_layer[hidden_layers][hidden_nodes];
};

#endif