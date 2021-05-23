# `scDLC`

> a deep learning framework to classify large  sample single-cell RNA-seq data

__Authors:__ Yan Zhou, Bin Yang, Tiejun Tong, Niansheng Tang

---

## Description

`scDLC` is a deep learning classifier (scDLC) for large sample scRNA-seq data, based on the long short-term memory recurrent neural networks (LSTMs). This classifier does not require a prior knowledge on the scRNA-seq data distribution and it is a scale invariant method which does not require a normalization procedure for scRNA-seq data.

---

## Usage

This GitHub repository contains the source code needed to build the `slapnap` docker image. The repository is also set up for continuous integration via Travis-CI, with built images found on [DockerHub](https://cloud.docker.com/u/slapnap/repository/docker/slapnap/slapnap). See the [Docker website](https://docs.docker.com/docker-for-windows/install/) for installation instructions.

From a terminal the image can be downloaded from DockerHub via the command line.

```{bash, eval = FALSE}
docker pull slapnap/slapnap
```

`slapnap` is executed using the docker run command. For example, the following code will instruct `slapnap` to create and evaluate a neutralization predictor for the bnAb combination VRC07-523-LS and PGT121:

```{bash, eval = FALSE}
docker run \
  -v path/to/local/save/directory:/home/output/ \
  -e nab="VRC07-523-LS;PGT121" \
  -e outcomes=”ic50;estsens” \
  -e combination_method="additive" \
  -e learners=”rf;lasso” \
  -e importance_grp=”marg” \
  -e importance_ind=”pred” \
  slapnap/slapnap:latest
```

The `–v` tag specifies the directory on the user’s computer where the report will be saved, and `path/to/local/save/directory` should be replaced with the desired target directory.  Options for the analysis are passed to the container via the `-e` tag; these options include the bnAbs to include in the analysis (`nab`), the neutralization outcomes of interest (`outcomes`), the method for predicting combination neutralization (`combination_method`), the learners to use in the analysis (`learners`), and the types of variable importance to compute (`importance_grp`, for groups of variables; `importance_ind`, for individual variables). Other output (e.g., the formatted analysis dataset and the fitted learners) can be requested via the `return` option. A full list of options and their syntax are available in the [`slapnap` documentation](https://benkeser.github.io/slapnap/3-sec-runningcontainer.html).

Complete documentation is available [here](https://benkeser.github.io/slapnap/).

## Issues

If you encounter any bugs or have any specific feature requests, please [file an
issue](https://github.com/benkeser/slapnap/issues).

---

## License

&copy; 2019- David Benkeser

The contents of this repository are distributed under the MIT license:
```
The MIT License (MIT)
Copyright (c) 2019- David Benkeser
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
