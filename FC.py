import numpy as np


class FC:
    def __init__(self, in_features, out_features, Weight, bias):
        self.in_features = in_features
        self.out_features = out_features
        self.Weight = Weight
        self.bias = bias

    def Fc(self, input_data):
        data = np.reshape(input_data, (self.in_features,1))
        weight = np.reshape(self.Weight, (self.out_features,self.in_features))

        data_out = weight @ data + self.bias

        return self.out_features,data_out

    #激活函数
    @staticmethod
    def sigmoid(input_data):
        return 1/(1+np.exp(-input_data))

    #最后一层的激活函数
    @staticmethod
    def SoftMax(input_data):
        data = input_data.copy()
        data = np.exp(data)/np.sum(np.exp(data))
        return data



