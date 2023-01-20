---
layout: page
title: kNN-Search
description: CUDA implementation of k nearest neighbor search
img: assets/img/CUDA_logo.jpeg
importance: 1
category: Machine Learning
github: https://github.com/v0rt3xh/Knn_CUDA
giscus_comments: true
---

The k-nearest neighbor (k-NN) algorithm is an important method in statistics and machine learning. We can use it to solve classification and regression problems. In a classification setting, we first find the k closest points for a given data point using a similarity measure. Then, we determine the label of the
given data point with majority vote.

This project focuses on the k-nearest neighbor search (k-NN search). Under the setting of k-NN search, we only need to find the k nearest neighbor. Further classification or regression tasks are not necessary.

We complete a CUDA implementation and a pure Cpp implementation of the plain version k-NN search. Our implementations are based on Vincent Garcia’s work. [Link](https://github.com/vincentfpgarcia/kNN-CUDA).

You can find the source code and sample usage from [here](https://github.com/v0rt3xh/Knn_CUDA/blob/main/README.md).