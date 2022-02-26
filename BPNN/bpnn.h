#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using std::vector;

#define input_nodes 6
#define hidden_nodes 3
#define hidden_layers 1
#define output_nodes 1
#define learning_rate 0.1

// 产生 -1~1 之间的随机数
double get_random_num(){
    return ((2.0 * (double)rand()/RAND_MAX) - 1)
}

double sigmoid(double x){
    double ans = 1 / (1+exp(-x));
    return ans;
}

#endif BPNN_H