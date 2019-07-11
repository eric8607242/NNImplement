# CNN
## Introduction
This is a CNN implementation without using any framework(e.g. tensorflow, keras).
It uses Adam optimization.The data it supported is only Cifar10.if you want use other data set, you can write the load function by yourself.
You can specify the model architecture in `config.json`, and it will use default architecture if you set config value to "".
1. Can specify the kernels size, kernels number, padding size, stride size and conv layers number.
2. Can specify the max pooling layers number, stride size and pooling size.
3. Can specify the numbers and size of the hidden layers.
4. Visulize the weight and data(Not yet)
### If you want to know more how to use, you can check the `CNN_example.ipynb` file
## Architecture
![](./CNN_architecture.png)
## Instruction
```
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
pip3 install -r requirements.txt
```
