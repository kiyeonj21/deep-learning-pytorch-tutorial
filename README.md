# Deep Learning PyTorch Tutorial

## Installation
1. Install all dependencies listed in requirements.txt
2. Download dataset into data folder 

``` bash
cd ./data
bash download-data.sh tux
bash download-data.sh action_img_trainval
bash download-data.sh action_trainval
bash download-data.sh fc_trainval
cd ../
```
## Run
(1). Solve multi-class classification problem using different last layers

train
``` python
cd ./experiments
python basic1.py -train -model regress -i_tr 10000
python basic1.py -train -model onehot -i_tr 10000
python basic1.py -train -model LL -i_tr 10000
python basic1.py -train -model L2 -i_tr 10000
cd ../
```

test
``` python
cd ./experiments
python basic1.py -test -model regress -i_te 10
python basic1.py -test -model onehot -i_te 10
python basic1.py -test -model LL -i_te 10
python basic1.py -test -model L2 -i_te 10
cd ../
```

(2) Learn quadratic classifier with simple linear and deep structure
``` python
cd ./experiments
python basic2.py -train -model linear
python basic2.py -train -model deep
python basic2.py -test -model linear
python basic2.py -test -model deep
cd ../
```

(3) Solve multi-class classification problem using Convolutional Neural Networks

train
``` python
cd ./experiments
python ccn1.py -train -i_tr 10000
python ccn2.py -train -i_tr 10000
cd ../
```

test
``` python
cd ./experiments
python ccn1.py -test -i_te 10
python ccn2.py -test -i_te 10
cd ../
```

(4) Solve multi-class classification problem using Residual Neural Networks

train / test
``` python
cd ./experiments
python resnet.py -train -i_tr 10000
python resnet.py -test -i_tr 10
cd ../
```

(5) Do segementation problem using Fully Convolutional Neural Networks

train / test
``` python
cd ./experiments
python fcnn1.py -train -i_tr 10000
python fcnn2.py -train -i_tr 10000
python fcnn1.py -test -i_te 5
python fcnn2.py -test -i_te 5
cd ../
```

(6) Do imitation learning using Recurrent Neural Networks
```
cd ./experiments
python rnn1.py -train -i_tr 2000
python rnn1.py -test -i_te 32
cd ../
```

```
cd ./experiments
python rnn2.py -train -model RNNModel1 -i_tr 2000
python rnn2.py -train -model RNNModel2 -i_tr 2000
python rnn2.py -test -model RNNModel1 -i_te 32
python rnn2.py -test -model RNNModel2 -i_te 32
cd ../
```

## Acknowledge
This tutorial is borrowed from the following class:
- [CS 342 - Neural networks](https://www.philkr.net/cs342/)