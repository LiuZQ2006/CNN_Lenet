import numpy as np

class Convolutional_Layer:
    def __init__(self,Conbolution_Core,stride,input_length,Core_length,Padding,Bias,in_channel,out_channel):
        self.Conbolution_Core = Conbolution_Core
        self.stride = stride
        self.input_length = input_length
        self.Core_length = Core_length
        self.Padding = Padding
        self.Bias = Bias
        self.in_channel = in_channel
        self.out_channel = out_channel

    def Convolution_calculate(self, input_data):

        #1*28*28 6*1*5*5 输入
        data = np.reshape(input_data, (self.in_channel,self.input_length, self.input_length))
        core = np.reshape(self.Conbolution_Core, (self.out_channel,self.in_channel,self.Core_length, self.Core_length))
        
        # Padding 1*32*32
        if(self.Padding!=0):
            data = np.pad(data, ((0, 0),(self.Padding, self.Padding), (self.Padding, self.Padding)), 'constant', constant_values=0)
        
        input_length = self.input_length + 2*self.Padding
        
        out_length = (input_length-self.Core_length)//self.stride + 1
        data_out = np.zeros((self.out_channel,out_length, out_length))

        # Calculate
        for k in range(0,self.out_channel):
                for i in range(0,out_length):
                    for j in range(0,out_length):
                        data_out[k,i,j] += np.sum(data[:,i*self.stride:i*self.stride+self.Core_length, j*self.stride:j*self.stride+self.Core_length]*core[k,:])

        data_out += self.Bias.reshape(-1, 1, 1)

        return out_length ,data_out

    #激活函数
    @staticmethod
    def sigmoid(input_data):
        return 1/(1+np.exp(-input_data))


