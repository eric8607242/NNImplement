# FC
## Introduction
This is a FC implementation without usingany framework(e.g. tensorflow, keras).
It uses SGD optimization.The data it supported is only Cifar.if you want use other data set, you can write the load way by yourself.
1. Can specify the numbers and size of the hidden layers.
2. Use the method **tuning\_hyperparameter** to get the best hyperparameter
3. Visulize the weight and data
### If you want to know more how to use, you can check the `FC_example.ipynb` file
## Architecture
![](./FC_architecture.png))
## Instruction
```
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
pip3 install -r requirements.txt
```
