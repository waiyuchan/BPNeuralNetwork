#include "bpnn.h"

BPNeuralNetwork::BPNeuralNetwork()
{
    srand((unsigned)time(NULL)); // 随机数种子
    error = 100.f;

    // 初始化输入层
    for (int i = 0; i < input_nodes; ++i)
    {
        input_layer[i] = new InputNode();
        for (int j = 0; j < hidden_nodes; ++j)
        {
            input_layer[i]->weight.push_back(get_random_num());
            input_layer[i]->w_delta_sum.push_back(0.f);
        }
    }

    // 初始化隐藏层
    for (int i = 0; i < hidden_layers; i++)
    {
        if (i == hidden_layers - 1)
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                hidden_layer[i][j] = new HiddenNode();
                hidden_layer[i][j]->bias = get_random_num();
                for (int k = 0; k < output_nodes; k++)
                {
                    hidden_layer[i][j]->weight.push_back(get_random_num());
                    hidden_layer[i][j]->w_delta_sum.push_back(0.f);
                }
            }
        }
        else
        {
            for (int j = 0; j < output_nodes; j++)
            {
                hidden_layer[i][j] = new HiddenNode();
                hidden_layer[i][j]->bias = get_random_num();
                for (int k = 0; k < hidden_nodes; k++)
                    hidden_layer[i][j]->weight.push_back(get_random_num());
            }
        }
    }

    // 初始化输出层
    for (int i = 0; i < output_nodes; i++)
    {
        output_layer[i] = new OutputNode();
        output_layer[i]->bias = get_random_num();
    }
}

void BPNeuralNetwork::forwardPropagation()
{
    // 隐藏层前向传播
    for (int i = 0; i < hidden_layers; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                double sum = 0.f;
                for (int k = 0; k < input_nodes; k++)
                    sum += input_layer[k]->value * input_layer[k]->weight[j];
                sum += hidden_layer[i][j]->bias;
                hidden_layer[i][j]->value = sigmoid(sum);
            }
        }
        else
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                double sum = 0.f;
                for (int k = 0; k < hidden_nodes; k++)
                    sum += input_layer[k]->value * input_layer[k]->weight[j];
                sum += hidden_layer[i][j]->bias;
                hidden_layer[i][j]->value = sigmoid(sum);
            }
        }
    }

    // 输出层前向传播
    for (int i = 0; i < output_nodes; i++)
    {
        double sum = 0.f;
        for (int j = 0; j < hidden_nodes; j++)
            sum += hidden_layer[hidden_layers - 1][j]->value * hidden_layer[hidden_layers - 1][j]->weight[i];
        sum += output_layer[i]->bias;
        output_layer[i]->value = sigmoid(sum);
    }
}

void BPNeuralNetwork::backwardPropagation()
{
    // backward propagation on output layer
    // -- compute delta
    for (int i = 0; i < output_nodes; i++)
    {
        double tmpe = fabs(output_layer[i]->value - output_layer[i]->right_out);
        error += tmpe * tmpe / 2;
        output_layer[i]->delta = (output_layer[i]->value - output_layer[i]->right_out) * (1 - output_layer[i]->value) * output_layer[i]->value;
    }

    // backward propagation on hidden layer
    // -- compute delta
    for (int i = hidden_layers - 1; i >= 0; i--) // 反向计算
    {
        if (i == hidden_layers - 1)
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                double sum = 0.f;
                for (int k = 0; k < output_nodes; k++)
                    sum += output_layer[k]->delta * hidden_layer[i][j]->weight[k];
                hidden_layer[i][j]->delta = sum * (1 - hidden_layer[i][j]->value) * hidden_layer[i][j]->value;
            }
        }
        else
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                double sum = 0.f;
                for (int k = 0; k < hidden_nodes; k++)
                {
                    sum += hidden_layer[i + 1][k]->delta * hidden_layer[i][j]->weight[k];
                }
                hidden_layer[i][j]->delta = sum * (1 - hidden_layer[i][j]->value) * hidden_layer[i][j]->value;
            }
        }
    }

    // backward propagation on input layer
    // -- update weight delta sum
    for (int i = 0; i < input_nodes; i++)
    {
        for (int j = 0; j < hidden_nodes; j++)
            input_layer[i]->w_delta_sum[j] += input_layer[i]->value * hidden_layer[0][j]->delta;
    }

    // backward propagation on hidden layer
    // -- update weight delta sum & bias delta sum
    for (int i = 0; i < hidden_layers; i++)
    {
        if (i == hidden_layers - 1)
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                hidden_layer[i][j]->b_delta_sum += hidden_layer[i][j]->delta;
                for (int k = 0; k < output_nodes; k++)
                    hidden_layer[i][j]->w_delta_sum[k] += hidden_layer[i][j]->value * output_layer[k]->delta;
            }
        }
        else
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                hidden_layer[i][j]->b_delta_sum += hidden_layer[i][j]->delta;
                for (int k = 0; k < hidden_nodes; k++)
                    hidden_layer[i][j]->w_delta_sum[k] += hidden_layer[i][j]->value * hidden_layer[i + 1][k]->delta;
            }
        }
    }

    // backward propagation on output layer
    // -- update bias delta sum
    for (int i = 0; i < output_nodes; i++)
        output_layer[i]->b_delta_sum += output_layer[i]->delta;
}

void BPNeuralNetwork::train(vector<sample> sampleGroup, double threshold)
{
    int sampleNum = sampleGroup.size();

    while (error > threshold)
    {
        cout << "training error: " << error << endl;
        error = 0.f;
        // initialize delta sum
        for (int i = 0; i < input_nodes; i++)
            input_layer[i]->w_delta_sum.assign(input_layer[i]->w_delta_sum.size(), 0.f);
        for (int i = 0; i < hidden_layers; i++)
        {
            for (int j = 0; j < hidden_nodes; j++)
            {
                hidden_layer[i][j]->w_delta_sum.assign(hidden_layer[i][j]->w_delta_sum.size(), 0.f);
                hidden_layer[i][j]->b_delta_sum = 0.f;
            }
        }
        for (int i = 0; i < output_nodes; i++)
            output_layer[i]->b_delta_sum = 0.f;

        for (int iter = 0; iter < sampleNum; iter++)
        {
            setInput(sampleGroup[iter].input);
            setOutput(sampleGroup[iter].output);
            forwardPropagation();
            backwardPropagation();
        }

        // backward propagation on input layer
        // -- update weight
        for (int i = 0; i < input_nodes; i++)
        {
            for (int j = 0; j < hidden_nodes; j++)
                input_layer[i]->weight[j] -= learning_rate * input_layer[i]->w_delta_sum[j] / sampleNum;
        }

        // backward propagation on hidden layer
        // -- update weight & bias
        for (int i = 0; i < hidden_layers; i++)
        {
            if (i == hidden_layers - 1)
            {
                for (int j = 0; j < hidden_nodes; j++)
                {
                    hidden_layer[i][j]->bias -= learning_rate * hidden_layer[i][j]->b_delta_sum / sampleNum;
                    for (int k = 0; k < output_nodes; k++)
                        hidden_layer[i][j]->weight[k] -= learning_rate * hidden_layer[i][j]->w_delta_sum[k] / sampleNum;
                }
            }
            else
            {
                for (int j = 0; j < hidden_nodes; j++)
                {
                    hidden_layer[i][j]->bias -= learning_rate * hidden_layer[i][j]->b_delta_sum / sampleNum;
                    for (int k = 0; k < hidden_nodes; k++)
                        hidden_layer[i][j]->weight[k] -= learning_rate * hidden_layer[i][j]->w_delta_sum[k] / sampleNum;
                }
            }
        }

        // backward propagation on output layer
        // -- update bias
        for (int i = 0; i < output_nodes; i++)
            output_layer[i]->bias -= learning_rate * output_layer[i]->b_delta_sum / sampleNum;
    }
}

void BPNeuralNetwork::predict(vector<sample> sampleGroup)
{
    int testNum = sampleGroup.size();
    for (int iter = 0; iter < testNum; iter++)
    {
        sampleGroup[iter].output.clear();
        setInput(sampleGroup[iter].input);
        // forward propagation on hidden layer
        for (int i = 0; i < hidden_layers; i++)
        {
            if (i == 0)
            {
                for (int j = 0; j < hidden_nodes; j++)
                {
                    double sum = 0.f;
                    for (int k = 0; k < input_nodes; k++)
                        sum += input_layer[k]->value * input_layer[k]->weight[j];
                    sum += hidden_layer[i][j]->bias;
                    hidden_layer[i][j]->value = sigmoid(sum);
                }
            }
            else
            {
                for (int j = 0; j < hidden_nodes; j++)
                {
                    double sum = 0.f;
                    for (int k = 0; k < hidden_nodes; k++)
                        sum += hidden_layer[i - 1][k]->value * hidden_layer[i - 1][k]->weight[j];
                    sum += hidden_layer[i][j]->bias;
                    hidden_layer[i][j]->value = sigmoid(sum);
                }
            }
        }

        // forward propagation on output layer
        for (int i = 0; i < output_nodes; i++)
        {
            double sum = 0.f;
            for (int j = 0; j < hidden_nodes; j++)
                sum += hidden_layer[hidden_layers - 1][j]->value * hidden_layer[hidden_layers - 1][j]->weight[i];
            sum += output_layer[i]->bias;
            output_layer[i]->value = sigmoid(sum);
            sampleGroup[iter].output.push_back(output_layer[i]->value);
        }
    }
}

void BPNeuralNetwork::setInput(vector<double> sample_data_in)
{
    for (int i = 0; i < input_nodes; i++)
        input_layer[i]->value = sample_data_in[i];
}

void BPNeuralNetwork::setOutput(vector<double> sample_data_out)
{
    for (int i = 0; i < output_nodes; i++)
        output_layer[i]->right_out = sample_data_out[i];
}