---
layout: distill
title: Recommender System (1) Youtube Recommender
date: 2023-01-26 17:12:00-0400
description: The first post of the Recommender Series.
tags: Recommender
categories: Youtube NeuralNet
giscus_comments: true
toc:
  - name: Introduction
  - name: System Overview
bibliography: recommender-1-distill.bib
---

## Introduction
Many of us have been using YouTube frequently. On the homepage, we can always find some videos to watch. I often watch cooking recipes and sports highlights. As a result, YouTube usually presents food or sports channels to me. Sometimes, I can also find some trending videos. So, how did YouTube's recommender system work? In this post, I will revisit a classic paper that introduces the structure of the YouTube recommender system: Deep Neural Networks for YouTube Recommendations ([Link](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)).

From the perspective of information retrieval, a recommender system consists of two major components: *candidates generation model* and *ranking model*. In the example of YouTube, the recommender model is expected to select hundreds of candidate videos from the massive corpus and assign scores to candidates for ranking.

<div class="caption">
        {% include figure.html path="assets/img/Dichotomy.png" title="dichotomy" class="img-fluid rounded z-depth-1" 
        width="360"%}
</div>
<div class="caption">
    An illustration of candidate generation and ranking, assuming that we need 26 candidates.
</div>

In the paper, the authors list three challenges of building such recommender model for YouTube.

- Scale: Youtube has a massive user base and video corpus, which demands highly specialized distributed learning algorithms and efficient serving systems. 
- Freshness: The system should be responsive to recent uploaded content as well as the latest action taken by the users. (New content vs well-established content just like exploration vs exploitation.)
- Noise: In training data, only noisy implicit feedback signals are available. Ground truth of user satisfication is not available. 

They use tensorflow to do experiments. Distributed training is involved. Billion scale. 

## System Overview

Figure 2 in the paper shows a "funnel" structure. 

Two neural networks are used for candidate generation and ranking.

- Candidate generation network
  Take events from the user's YouTube activity history as input and retrieve a small subset of videos (hundreds) from a large corpus (millions). The objective of this network is to get highly relevant (in the measure of precision) candidates for the user.

  This network only provides broad personalization via collaborative filtering. The similarity between users is expressed in terms of coarse features such as IDs of watched videos, search query tokens, and demographics.

- Ranking network
  Its objective is to get a fine-level representation of the candidates and present a few "best" recommendations in a list. The importance of candidates should be assigned with high recall. The ranking network takes in a rich set of features (describing the video and user) and assigns a score to each video according to an objective function. Highest scoring videos are presented to the user. 

**Some benefits of the two-stage framework**

- Allowing candidates retrieval from a large corpus, with some degree of guarantee of personalization.
- It's feasible to incorporate candidates generated from other sources. 

**Evaluation in development / in production**

During development, offline metrics (precision, recall, ranking loss, etc.) are used to guide iterative improvements to the system. 

In production, to determine the effectiveness of an algorithm / model, the authors rely on A/B testing via live experiments. In the experiment, one can monitor click-through rate, watch time, and other metrics that measure user's engagement. 

*Live A/B results are not always correlated with offline experiments.*

## Dive A Little Bit Deeper

### Candidate Generation Network
The generation process helps one find hundreds of videos that may be relevant to users from the enourmous corpus. Previously, people have been using a matrix factorization approach trained under rank loss. Their approach can be viewed as a non-linear generalization of factorization techniques. In the sense that shallow networks which only embeds users' previous watches are used during early iterations. 

#### Recommendation as Classification
There are many ways to approach recommendation problems. In many occasions, we are predicting whether a positive outcome would happen. For instance, we can define the positive outcome for Youtube recommendation as "User clicks the suggested video". However, this definition might not be precise enough. Since some users would click the video and lose their interest in one minute. 

The authors pose recommendation as an extreme multiclass classification problem. The problem is to accurately predict a specifi video watch $w_t$ at time $t$ among millions of videos $i$ from a corpus $V$ based on a user $U$ and context $C$

$$P(w_t = i | U, C) = \frac{e^{v_iu}}{\sum_{j \in V} e^{v_ju}}$$

where $u \in R^N$ is a high-dimensional "embedding" of the user, context pair. $v_j \in R^N$ is the embedding of a candidate video. The task of the neural network is to learn user embedding $u$ as a function of the user's history and context. In this setting, a positive example is "a
user completing a video".

(TODO: Need to modify the content here)
There are millions of "classes" in this problem. The authors relies on a technique to sample negative classes from the background distribution (“candidate sampling”) and then correct for this sampling via importance weighting.  For each example the cross-entropy loss is minimized for the true label and the sampled negative classes. In practice several thousand negatives are sampled, corresponding to more than 100 times speedup over traditional softmax.

- Why not hierarchical softmax?
  (TODO: Need to check out the method)

At serving time, one need to compute the most likely K classes (videos) to present them to the user. The latency requirement could be tens of milliseconds. An approximate scoring scheme sublinear in the number of classes is needed. Previous systems at YouTube relied on Hashing (TODO: Check the method). Calibrated likelihoods from the softmax layer are not needed. So the problem reduces to a nearest neighbor search in the dot product space. The authors claim that A/B test results were not particularly sensitive to the choice of kNN search algorithm. 

#### Model Architecture
Motivated by continuous bag of words language models, the authors learn high dimensional embeddings for each video in a fixed vocabulary and feed those embeddings into a feedforward neural network. A user's watch history is represented by a variable-length sequence of sparse video IDs which is mapped to a dense vector representation via the embeddings. 

The network requires fixed-sized dense inputs and simply averaging the embeddings performed best among several strategies (sum, component-wise max, etc.). **The embeddings are learned jointly with all other parameters in the network.** Features are concatenated into a wide first layer, followed by several layers of fully connected ReLU.

#### Heterogeneous Signals
When doing factorization on matrices, one might encounter the problem of sparsity. Using deep neural networks avoids this issue. Arbitrary continuous
and categorical features can be easily added to the model. 

Demographic features are important for providing priors so that we have reasonable recommendations to new users. The geographic region and continuous features are embedded and concatenated. Simple binary and continuous features (gender, logged-in state, and age are input directly into the network, normalized into $[0, 1]$)

**"Example Age" Feature**

A very interesting topic. Users prefer fresh content, though not at the expense of relevance. Critical for bootstrapping & viral propagation. However, machine learning model tends to exhibit an implicit bias towards the past (Trained from historical examples to predict future results.). 

The distribution of video popularity is highly non-stationary. But the multimodal distribution over the corpus produced by our recommender will reflect the average watch likelihood in the training window of several weeks. *To correct this, the authors feed the age of the training example as a feature during training.* At serving time, the age is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window. 

#### Label and Context Selection

Recommendation often involves solving a surrogate problem and transferring the result to a particular context. A classic example is the assumption that accurately predicting ratings lead to effective movie recommendations. The authors find that the choice of this surrogate problem has an outsized importance on performance in A/B testing (But is very difficult to measure with offline experiments.).

Training examples are generated using all YouTube watches (even those embedded on other sites) rather than just watches on the recommendations we produce. Otherwise, it would be hard for new content to surface and the recommender would be overly biased towards exploitation. 

Another key insight is that generating a fixed number of training examples per user improve live metrics (effectively weighting users equally in the loss function). This prevents highly active users from dominating the loss.

*TODO: More info on the following stuffs* Great care must be taken to withhold information from the classifier in order to prevent the model from exploiting the structure of the site, and overfitting the surrogate problem (**???**).

> e.g. Consider the following example: A user just issued query for "Taylor Swift". Since the recommender is posed as predicting the next watched video, a classifier given the information will predict the videos to be the ones on "Taylor Swift" search results page. Reproducing the search result page does not seem to be a good recommendation. 
> By discarding sequence information and representing the search queries with an unordered bag of tokens, the classifier is no longer directly aware of the origin of the label. 

Natural consumption patterns of videos typically lead to very asymmetric co-watch probabilities. Episodic series are usually watched sequentially and users often discover artists in a genre beginning with the most broadly popular before focusing on smaller niches. The authors find much better performance predicting the user's next match, rather than predicting a randomly held-out watch.

> Many collaborative filtering systems implicitly choose the labels and context by holding out a random item and predicting it from other items in the user’s history. This leaks future information and ignores any asymmetric consumption patterns.

In contrast, the authors rollback a user's history by choosing a random watch and only input actions the user took before the held-out label watch.

#### Experiments with Features and Depths
Adding feature and depth significantly improves precision on holdout data.  Network structure followed a common “tower” pattern in which the bottom of 
the network is widest and each successive hidden layer halves the number of units.

The depth zero network is effectively a linear factorization scheme. 

Features beyond video embeddings improve holdout Mean Average Precision (MAP) and layers of depth add expressiveness so that the model can effectively use these additional features by modeling their interaction.

### Ranking
Now we have candidate predictions. We would like to use impression data to specialize and calibrate candidates for the particular user interface. For instance, a user may watch a given video with high probability generally but is unlikely to click the homepage impression due to the thumbnail image. 

During ranking, we have access to many more features describing the video and the user's relationship to the video. Ranking is also important for ensembling different candidate sources of which scores are not directly comparable. 

The ranking network architecture is similar to candidate generation network. It assigns an independent score to each video impression using logistic regression. The list of videos is sorted by the score and returned to the user.
The final ranking objective is constantly being tuned based on live A/B testing results. But it's generally a simple function of expected watch time per impression.

> Ranking by click-through rate often promotes "click-bait" videos. Watch time can better capture user engagement.

#### Feature Rrepresentation
The features of YouTube are segregated with the traditional taxonomy of categorical and continuous / ordinal features.
- Categorical features may vary widely in their cardinality: some are binary (logged in status), some has millions of values (users' last search queries.) 
Features are further split according to whether they contribute to only a single value ("univalent") or a set of values ("multivalent").
> Example of a univalent categorical feature is the video ID of the impression being scored, while a corresponding multivalent feature might be a bag of the last N video IDs the user has watched. 
- Features can also be classified according to whether they describe properties of the item ("impression") or the properties of the user/context ("query"). Query features are computed once per request while impression features are computed for each item scored.

#### Feature Engineering
Typically, hundreds of features are used in the ranking model, roughly split evenly between categorical and continuous features. The main challenge is in representing a temporal sequence of user actions and how these actions relate
to the video impression being scored. The most important signals are those that describe a user's previous interaction with the item itself and other similar items. 
> Consider the user's past history with the channel that uploaded the video being scored - how many videos has the user watched from the channel? When was the last time the user watched a video in this topic?

These continuous features describing past user actions on related items are particularly powerful because they generalize well across disparate items. The authors also find it crucial to propagate information from candidate generation into ranking in the form of features, e.g. which sources nominated
this video candidate? What scores did they assign?

Features describing the frequency of past video impressions are critical for introducing "churn" in recommendations (successive requests do not return identical lists.) If a user was recently recommended a video but did not watch
it then the model will naturally demote this impression on the next page load.

#### Embedding Categorical Features
For categorical features, each unique ID space ("vocabulary") has a separate learned embedding with dimension that increases approximately propotional to the logarithm of the number of unique values. Very large cardinality ID spaces  are truncated by including only the top N after sorting based on their frequency in clicked impressions. Out-of-vocabulary values are simply mapped to zero embedding. Multivalent categorical features are averaged before feeding into the network. 

Importantly, categorical features in the same ID space also share underlying emeddings. For example, there exists a single global embedding of video IDs that many distinct features use: video ID of the impression, last video watched by the user, video ID that seeded the recommendation. **Despite the shared embedding, each feature is fed separately into the network so that the layers above can learn specialized representations per feature**

#### Normalizing Continuous Features
Proper normalization of continuous features was critical for convergence. 
*TODO: The normalization method: scaling the values such that the feature is equally distributed in (0,1)*

$x^2, x^{0.5}$ are included for more expressive power. (Form super- and sub-linear functions of the feature) This could improve offline accuracy.

#### Modeling Expected Watch Time
Our goal is to predict expected watch time given training examples that are either positive (the video impression was clicked) or negative (the impression was not clicked). To predict expected watch time, the authors use weighted logistic regression.

The positive (clicked) impressions are weighted by the observed watch time on
the video. Negative (unclicked) impressions all receive unit weight.

*TODO: The formula of this weighted logistic regression model. Logistic regression was modified by weighting training examples with watch time for positive examples and unity for negative examples, allowing us to learn odds that closely model expected watch time*

#### Experiments with Hidden Layers
"Weighted, per-user loss" was obtained by considering both negative and positive impressions shown to a user on a single page. If the negative impression receives a higher score than the positive impression, then we consider the positive impression’s watch time to be mispredicted watch time.

Weighted, peruser loss is then the total amount mispredicted watch time 
as a fraction of total watch time over heldout impression pairs.









