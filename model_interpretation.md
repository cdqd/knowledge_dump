# Interpretation of statistical models

This page summarizes some different techniques for interpreting the predictions of statistical models in terms of the features used to train them. i.e. for any observation, how did the inputs influence the model's prediction to be this value?

### Linear regression

The simplest models to interpret are linear regression models, because the output is constructed as the addition of inputs multiplied by a coefficient. Take for example a logistic regression (predicting a binary outcome) with a continuous input `x1` and a binary input `x2:

```
logit(yhat) = log-odds = b0 + b1 * x1 + b2 * x2
```

Here we can say that every unit increase in `x1` leads to a `b1` increase in the log-odds of the target event occuring, and that having the `x2` characteristic leads to a further `b2` increase in the log-odds. This assume that `x1` and `x2` for any observation is determined indepdently (i.e. an observation's `x1` value is not governed by its `x2` value and vice-versa), although we note that this is easily violated (usually subconciouscly) in practice.

Models that have a more complex structure than (generalized) linear models an additional layer of analysis to understand relationships between features and predictions. The additional analysis usually involves passing a sample set of observations through the model and recording the predictions, then changing the inputs slightly and recording the change in the prediction. (This is sometimes referred to as sensitivity testing in some other disciplines).

There are many ways to 'change the inputs slightly' and analyse the changes in the predictions though, so there are a few distinct established methods in this area.

### Partial dependence plots (PDP)

PDPs are quite simple to implement. Say we want to understand the relationship between `yhat` and `x1` (a continuous feature), e.g. when `x1 = 50`, what does the model produce for `yhat`? Of course this depends on the values of other features as well. The easiest way to approximate this is to set `x1 = 50` for all the observations that we have, run these synthetic observations through the model, then average the prediction values.

If we assume that our training set observations adequately approximate the joint probability distribution of the feature space, then we effectively marginalizing out `x1 = 50`. We can marginalize out all the values of `x1` in a similar manner (i.e. `x1 = 60`, `x1 = 70`, etc.), and in the end we have a table that describes the relationship between `x1` values and `yhat` values. Of course, if `x1` is a continuous feature, or even if it's an unbounded integer, we need to bucket the `x1` values (the bucket widths shouldn't be too big though, otherwise we might lose information from the model).

Another way to look at PDPs is at the individual observation level. Here we vary the `x1` values holding the other features constant for this observation. This is referred to "individual conditional expectations".

#### Drawbacks

There are two issues with PDPs:

* If the feature space is big (i.e. the model uses a lot of features, and/or feature values have large ranges), the observations in the training set may not adequately approximate the joint probability distribution. We would need a large amount of data to compensate for this, and as per the curse of dimensionality, the additional data required grows much faster than the number of features added.
* If the feature range of the feature we're trying to perform inference on (`x1` in this case) is big, it takes many `predict` calls, and this also increases with the amount of observations in our sample (you might be thinking here to reduce our sample size, but then we run into the dimensionality problem). Also, we usually want to compare the effects of different features on the target, so if we have multiple features like this it would take a long time to generate all of the partial dependency data.

Although some texts also note that this method doesn't allow for the inspection of interactions, most explanation methods don't, so we don't call it out as a disadvantage here.

This is a good reference for understanding the implementation details of PDPs: https://www.sas.com/content/dam/SAS/support/en/sas-global-forum-proceedings/2018/1950-2018.pdf

### LIME / Kernel SHAP

LIME is a framework that seeks to explain how each feature contributes to an **individual** prediction. The 'L' stands for local, and the framework aims to locally approximate (at the individual observation level) the prediction function of the model.

The question LIME seeks to answer is "what is the magnitude and direction of this feature's contribution to the predicted value for this observation?".

For a **single observation**, if we want to know the contribution of `x1` to the model's prediction, we:

* Hold `x1` constant. Create synthetic observations by sampling values for the features `x2, x3, ...`  from the other observations in the dataset (this is usually called 'permuting'). We can repeat this by holding `x2` constant and sampling `x1, x3, ...`, and so on.
* For each synthetic observation, record the difference between the predicted value and the baseline value of the model. Let's call this the `contribution target`.
* Next, consider a simple mapping of the features of the synthetic observations: where the mapping value is 1 if the feature has been held constant, and 0 if it has been permuted.
* We now have a linear regression exercise: predicting `contribution target` with these mapping values as predictors. The resulting coefficients will give us each feature's contribution to the prediction value.
* There are a couple more things to note about this regression exercise:
  * The synthetic observations used in the regression are **weighted** based on how 'plausible' they are. If a synthetic observation lies far away from the original observation, we give it less weight in the regression. Usually we need an additional kernel to calculate the distances and resulting weights.
  * Penalized regression (e.g. LASSO) is usually used, because the statistical models we apply LIME to often use a large number of features (think Random Forest, Gradient Boosted Decision Trees).

This is a good simplified explanation of LIME: https://christophm.github.io/interpretable-ml-book/lime.html
And this is the original paper introducing the method: https://arxiv.org/pdf/1602.04938.pdf

### 'Saabas' method for trees

This is another local explanation method that takes advantage of the binary-split structure of decision trees.

For a **single observation**, the method is:

* Start from the root of the tree. The prediction value at this point is the average target value of all observations, the baseline value. Say the baseline value is `100` (we're looking at a regression tree here).
* Consider the feature at the first split. Assign the increase or reduction in predicted value for the observation after going through this split to that feature.
  * For example, if the first split occurs at `x1`, then depending on the observations `x1` value, it will either go into the left or right node. If it goes to the right node, and the average target value of the observations in that node is `80` say, then it will have "lost" `100 - 80 = 20` due to splitting at `x1`.
* We can continue this logic until the observation reaches one of the leaf nodes. The idea is easily extended to an ensemble of trees -- for each feature, we take an average of its contributions across all trees (including the trees where its contribution is zero because it wasn't used as a split).

This is a simple method to understand is fairly fast to compute. It potentially has some attribution issues - more detail is explained by the writers of the SHAP package [here](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) - but for most purposes should be fine to use.

If we want at how the model is using different features as a whole, we can repeat this process a decently large number of observations from the training dataset, and plot the results (feature value, contribution to prediction) on a scatter plot. General trends / cutoffs can be picked out from this.

This is a good simplified explanation of the Saabas method for trees: http://blog.datadive.net/interpreting-random-forests/

### Tree SHAP (Shapley Additive Explanations)

Tree SHAP is an alternative to Saabas for decision trees that intends to protect against the attribution issues of Saabas.

The theory behind Tree SHAP (for a **single observation**) is as follows:

* The most accurate way to find the contribution of a feature is to consider its impact on the prediction under all subsets of feature combinations.
  * Say we had three features, `x1`, `x2`, and `x3`. There would be the base case where all features are null. Then if we wanted to find the impact of `x1` on the prediction, we would need to consider the cases where: `x1` is added to the model first, `x1` is added to the model after `x2` but before `x3`, `x1` is added after `x2` and `x3`, etc.
  * This is the idea behind Shapley values, which is a method used to determine fair allocation of profits in game theory.
* In decision trees, we can "add" features to the model in sequence using the following methodology:
  * Start with the baseline value -- the average target value for all observations. Say this is `100`.
  * Suppose we first add `x1` to the model. We run it through the tree, and whenever we have a split on a feature that is not yet included in the model (i.e. `x2` or `x3`), we take a weighted average of the target values of each of the nodes below it. The weights are the number of observations in each node.
  * This produces a value, say `90`. We say that `x1`'s marginal contribution in this case is `90 - 100 = -3`.
  * We add the next feature, say `x2`, and run it through the tree in the same manner. We then compare the resulting value, say `92`, to `90`. This is a `+2` contribution from `x2`.
  * We repeat until all features have been added, in all the possible orderings.
* The marginal contribution of a particular feature to the observation's prediction is then the average of the contributions under all permutations, across all trees (if the model is an ensemble).

We see that this is a more robust method than Saabas, but is also computationally much more expensive - the number of subsets to check grows exponentially with the number of features. However, the author of the [SHAP](https://github.com/slundberg/shap/tree/master/shap) package (which includes other algorithms alongside Tree SHAP) managed to optimize it for decision trees to run in polynomial time.

There is also a lot of theory behind why SHAP should be used over other feature interpretation methods, but I don't quite understand it myself so won't comment.

Another good reason to use the SHAP method is that it extends well to quantifying **interaction effects** between features.

Similar to the Saabas method, we can repeat this process for a decently large number of observations from the training dataset and plot the results on a scatter plot to understand how different values for a feature affect the model's prediction.

A good simplified explanation of Tree SHAP can be found here: https://medium.com/analytics-vidhya/shap-part-3-tree-shap-3af9bcd7cd9b

And this is the original paper by the authors of Tree SHAP: https://arxiv.org/pdf/1802.03888.pdf

## Summary

Overall:

* Use a linear regression model from the outset for the best interpretability.
* If using Random Forest or Gradient Boosted Decision Trees, use Tree SHAP to explain how features are being handled by the model. The original implementation of [SHAP](https://github.com/slundberg/shap/tree/master/shap) in Python is best-in-class, and works with many tree algorithms from `scikit-learn`.
* LIME or Kernel SHAP is a possible alternative for non-tree-based models.
