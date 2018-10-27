# Naive Bayes

Bayes' theorem:

`P(Y|x) = P(Y)P(x|Y) / P(x)`

A naive bayes model extends this to a classification problem where we have a categorical target value with a vector of predictors **x**.

Main assumption: predictors x1, x2, ... are independent of each other.

The model is `Yhat_i = P(Y_i | x1, x2, ...) = (1/Evidence) * P(Y_i) * product_over_j(P(x_j | Y_i))`. The last equality is a result of the independence-between-predictors assumption.

- Evidence is simply a scaling factor equal to `P(x) = sum_over_k(P(x|Y_k) * P(Y_k))`.

This model effectively uses priors to model the posterior probability of `Y_i`.

Compared to a linear regression, it:
- Is able to incorporate the effect of high-cardinality categorical predictors 
- Does not produce co-efficients which describe the effect a predictor has on the target
- Will overestimate the target if two dependent predictors are present. This is because the naive bayes model is purely multiplicative (multiplying two strong effects with each other), while linear regression will have a lower co-efficient for one of the predictors if the two predictors are correlated.
