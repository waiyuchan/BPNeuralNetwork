#include "bpnn.h"

int main()
{
    BPNeuralNetwork testNet;

    // 学习样本
    vector<double> samplein[4];
    vector<double> sampleout[4];
    samplein[0].push_back(0);
    samplein[0].push_back(0);
    sampleout[0].push_back(0);
    samplein[1].push_back(0);
    samplein[1].push_back(1);
    sampleout[1].push_back(1);
    samplein[2].push_back(1);
    samplein[2].push_back(0);
    sampleout[2].push_back(1);
    samplein[3].push_back(1);
    samplein[3].push_back(1);
    sampleout[3].push_back(0);
    sample sampleInOut[4];
    for (int i = 0; i < 4; i++)
    {
        sampleInOut[i].input = samplein[i];
        sampleInOut[i].output = sampleout[i];
    }
    vector<sample> sampleGroup(sampleInOut, sampleInOut + 4);
    testNet.train(sampleGroup, 0.0001);

    // 测试数据
    vector<double> testin[4];
    vector<double> testout[4];
    testin[0].push_back(0.1);
    testin[0].push_back(0.2);
    testin[1].push_back(0.15);
    testin[1].push_back(0.9);
    testin[2].push_back(1.1);
    testin[2].push_back(0.01);
    testin[3].push_back(0.88);
    testin[3].push_back(1.03);
    sample testInOut[4];
    for (int i = 0; i < 4; i++)
        testInOut[i].input = testin[i];
    vector<sample> testGroup(testInOut, testInOut + 4);

    // 预测测试数据，并输出结果
    testNet.predict(testGroup);
    for (int i = 0; i < testGroup.size(); i++)
    {
        for (int j = 0; j < testGroup[i].input.size(); j++)
            cout << testGroup[i].input[j] << "\t";
        cout << "-- prediction :";
        for (int j = 0; j < testGroup[i].output.size(); j++)
            cout << testGroup[i].output[j] << "\t";
        cout << endl;
    }

    system("pause");
    return 0;
}
