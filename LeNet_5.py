import numpy as np
import convolutional_layer as cl
import Pool as pl
import FC as fc


def LeNet_5(input_data,Conv1_Core,Conv1_Bias,Conv2_Core,Conv2_Bias,FC1_Weight,FC1_Bias,FC2_Weight,FC2_Bias,FC3_Weight,FC3_Bias):

    
    conv1 = cl.Convolutional_Layer(Conbolution_Core=Conv1_Core,stride=1,input_length=28,Core_length=5,Padding=2,Bias=Conv1_Bias,in_channel=1,out_channel=6)
    conv1_out_length,conv1_out = conv1.Convolution_calculate(input_data)
    conv1_out = cl.Convolutional_Layer.sigmoid(conv1_out)


    pool1 = pl.Pooling(Pool_size=2,Stride=2)
    pool1_out_length,pool1_out = pool1.Pooling(conv1_out,conv1_out_length,in_channel=6)

    
    conv2 = cl.Convolutional_Layer(Conbolution_Core=Conv2_Core,stride=1,input_length=pool1_out_length,Core_length=5,Padding=0,Bias=Conv2_Bias,in_channel=6,out_channel=16)
    conv2_out_length,conv2_out = conv2.Convolution_calculate(pool1_out)
    conv2_out = cl.Convolutional_Layer.sigmoid(conv2_out)

    
    pool2 = pl.Pooling(Pool_size=2,Stride=2)
    pool2_out_length,pool2_out = pool2.Pooling(conv2_out,conv2_out_length,in_channel=16)


    Fc1 = fc.FC(in_features=pool2_out_length*pool2_out_length*16,out_features=120,Weight=FC1_Weight,bias=FC1_Bias)
    Fc1_out_length,Fc1_out = Fc1.Fc(pool2_out)
    Fc1_out = fc.FC.sigmoid(Fc1_out)


    Fc2 = fc.FC(in_features=Fc1_out_length,out_features=84,Weight=FC2_Weight,bias=FC2_Bias)
    Fc2_out_length,Fc2_out = Fc2.Fc(Fc1_out)
    Fc2_out = fc.FC.sigmoid(Fc2_out)


    Fc3 = fc.FC(in_features=Fc2_out_length,out_features=10,Weight=FC3_Weight,bias=FC3_Bias)
    Fc3_out_length,Fc3_out = Fc3.Fc(Fc2_out)
    Fc3_out = fc.FC.SoftMax(Fc3_out)

    return Fc3_out_length , Fc3_out


