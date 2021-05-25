# `scDLC`

> a deep learning framework to classify large  sample single-cell RNA-seq data.

__Authors:__ Yan Zhou, Bin Yang, Tiejun Tong, Niansheng Tang

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
---

## Description

`scDLC` is a deep learning classifier (scDLC) for large sample scRNA-seq data, based on the long short-term memory recurrent neural networks (LSTMs). This classifier does not require a prior knowledge on the scRNA-seq data distribution and it is a scale invariant method which does not require a normalization procedure for scRNA-seq data.

<div align=center>
<img src="https://z3.ax1x.com/2021/05/25/2S9eMT.md.png" height="75%" width="75%">
</div>

---

## Requirements

It is required to install the following dependencies in order to be able to run the code of scDLC

- [Anaconda3](https://www.anaconda.com/products/individual)  
- [R>=3.5.3](https://cran.r-project.org/)  
- [python 3](https://www.python.org/downloads/)  
  [sklearn](https://pypi.org/project/sklearn/0.0/)，[numpy 1.18.5](https://pypi.org/project/numpy/1.18.5/)，[pandas 1.1.0](https://pypi.org/project/pandas/1.1.0/)，[tensorflow 1.15.0](https://pypi.org/project/tensorflow/1.15.0/)，[rpy2 2.9.5](https://pypi.org/project/rpy2/2.9.5/)
  
  In order to enable rpy2 package to be successfully imported in python, you need to add the following variables to the environment variables.  
  
  Open the environment variable setting interface:
  ```
  computer-> property -> advanced and system setting -> environment variables
  ```
  In the user variable field, add the following R path to variables path:
  ```
  C:\Program Files\R\R-3.0.2\bin\x64 (my system is windows 64bit) 
  ```
  In the system variable field add two new variables:
  ```
  R_HOME C:\program files\r\r-3.0.2  
  R_USER C:\Users\"your user name"\Anaconda\Lib\site-packages\rpy2
  ```
  
---

## Data

Four scRNA-seq datasets are from National Center for Biotechnology Information Search database [(NCBI)](https://www.ncbi.nlm.nih.gov/).
The first dataset [GSE99933](https://github.com/scDLC-code/scDLC/releases/tag/Data) has two classes, including 384 samples recombining at E12.5 and 384 samples recombining 
at E13.5. The second dataset [GSE123454](https://github.com/scDLC-code/scDLC/releases/tag/Data) includes 463 samples from single nuclei and 463 samples from matched single cells with measurements on 42003 genes. 
The third dataset [GSE113069](https://github.com/scDLC-code/scDLC/releases/tag/Data) contains three classes, each with 345, 422, 423 samples, respectively. The fourth
dataset [GSE84133 (Baron1)](https://github.com/scDLC-code/scDLC/releases/tag/Data) contains all major cell groups from the first human donor, excluding those with less than 20 cells. 
It contains nine classes, each with 110, 51, 236, 872, 214, 120, 130, 70 and 92 samples.

```
                  Datasets          |  Sample size   |  No. of classes   |  No. of genes  
                  ------------------|----------------|-------------------|---------------
                  GSE99933          |       768      |         2         |     23420       
                  GSE123454         |       926      |         2         |     42003        
                  GSE113069         |       1190     |         3         |     23218       
                  GSE84133(Baron1)  |       1895     |         9         |     20126       
```

---

## Usage

This GitHub repository contains the source code needed to build the `scDLC` classifier. selectgene.R is the code used to select differentially expressed gene. Before training the `scDLC` classifier, you need set the path of the scDLC folder to the default working directory and to modify the dataset name which will be read in selectgene.R:

```
setwd("C:\\Users\\your user name\\Desktop\\scDLC")
data <- read.table("datasetname.csv",sep=',')
```

Here is a description of some of the important parameters in the `scDLC` mode：

* `num_classes`: The number of classes in the dataset
* `num_steps`: Number of genes in the training sample
* `batch_size`: training sample size in one batch
* `lstm_size`: size of hidden state of lstm
* `num_layers`: number of lstm layers
* `n_epoch`: Number of training rounds
* `train_keep_prob`: The keep rate of neuron node during training


Open command window in the scDLC folder and execute the following command to train the `scDLC` classifier:

```
python scDLC_train.py \
--num_steps=100 \
--batch_size=11 \
--lstm_size=64 \
--num_layers=2 \
--n_epoch=30 \
--train_keep_prob=0.3 \
 
```

## Issues

If you encounter any bugs or have any specific feature requests, please [file an
issue](https://github.com/scDLC-code/scDLC/issues).

---

