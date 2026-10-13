import numpy as np

class Pooling:
    def __init__(self, Pool_size, Stride):
        self.Pool_size = Pool_size
        self.Stride = Stride


    def Pooling(self, input_data, input_length,in_channel):
        data = np.reshape(input_data, (in_channel, input_length, input_length))

        out_length = (input_length - self.Pool_size) // self.Stride +1

        data_out = np.zeros((in_channel, out_length, out_length))

        for k in range(0, in_channel):
            for i in range(0, out_length):
                for j in range(0,out_length):
                    data_out[k,i,j] = np.sum(data[k,i*self.Stride:i*self.Stride+self.Pool_size, j*self.Stride:j*self.Stride+self.Pool_size])/(self.Pool_size**2)

        return out_length,data_out
    