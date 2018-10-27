# Numerical optimization, Gradient Boosting

Overview of the logic behind optimization methods like gradient descent and Newton's method. Quick overview of how this is connected to and applied to Gradient Boosting, particularly Gradient Boosted Decision Trees. Also compares the methodology used in xgboost vs. traditional gradient boosting.

## Cost function minimization

### Gradient Descent

Simple example: Equation solution. Want to find parameters `b1 b2` such that `C = (y - (b1 + b2x))^2` is a minimum.

1. Start initial guesses b1 b2.

2. Find partial C' with respect to b1 and b2. Evaluate at b1 and b2 respectively.

3. Let `(next b1 b2) = (initial b1 b2) - step * C'(initial b1 b2)`, repeat until convergence of (b1 b2) values. Step size can be optimized for each step such that the value of the cost function is the lowest.

Key points to note: It is helpful to remember that C' is always calculated using the most up-to-date guesses of b1 b2. 
Also, GD works best when all features (including the target) are on the same scale, so it may be necessary to standardize all features first.

Generalizing to a dataset (e.g. extending the above example to be a linear regression with more than one observation), gradients for each row must be calculated and summed, then the step size optimized. May be costly, so can use *stochastic* gradient descent, which only calculates cost and estimates step size based on a random subsample. SGD is widely used in ML algorithms.

### Newton's method

Instead of working off the first derivative, the line of thinking is: C' = 0 when the minimum occurs, so we need to approximate the root for that equation. This involves a first-order taylor approximation (applied to C'), which is why C'' is needed.

In summary, Newton's method uses the second derivative of C for faster convergence. But more computationally intensive, and some cost functions may not have a second derivative that is defined everywhere.

## Traditional Gradient Boosting/Additive models

### Alternate way to look at gradient descent

In the context of ML, we want to find an algorithm to feed our inputs into such that C is minimized for *any* input values.

Call the updates we make in gradient descent f. So in the previous formula, `(next b1 b2) = (initial b1 b2) - f`, and so on. 
Each `f` we have can be thought of as a "booster/learner". Of course, adding all the f's together gets us to the final answer.
This type of booster/learner is the most simple, because it is simply a numeric vector where the components are independent. However, it can't generalize outside of the training data.

### Other considerations for additive models

Building on from the above perspective, it seems a potential way to generate a good algorithm is to build it step by step. Instead of using gradient descent, we can instead initialize a learner function of any form, f0, and then aim to achieve: `yhat = f0 + f1 + f2 + ...` such that yhat is accurate. This is called additive modelling.

The learners are built one-by-one. For example, we want: `yhat = f0 + f1` at the "first" step, so we aim to minimize the cost function C(y, f0 + f1), given our results from f0. This optimisation may be possible for simple C and learners with a small number of parameters.

However, for functions with many parameters, it is difficult/computationally intensive to do this at each iteration.

### Gradient boosting combines the two ideas above

So our aim is to build learners (not simple numeric vectors) additively, but these learners need to be slightly complex to help achieve the best accuracy. Combining the two ideas above, instead of trying to optimize C directly at each iteration, we can **fit our learner to the update-values (f's) produced by gradient descent**. So we try to replicate the gradient descent update values via functions, and our final result is the sum of all the functions (analogous to how the final result in GD is the sum of the update values). This is called gradient boosting.

We can use whatever functions we want to try and fit to the gradient-descent-update values (linear functions, trees, etc.). The most popular implementation at the moment uses trees (CARTs), and has been named Gradient Boosted Decision Trees (GBDT).

### Summary - traditional GBDT

1. Choose C that best fits the task (e.g. regression = squared error loss, but classification* = logloss as we do not want to overly punish incorrect predictions). C should be differentiable. 
2. The target values at each iteration are equal to the gradient of C, evaluated at the most up-to-date predictions. The initial prediction usually starts at the average of the response for all observations in the training data (can be changed, but no reason to).
3. The learners are regression trees. In each tree, we are grouping observations into leaves, with each leaf being assigned one value.
    - In order to build the tree, we first think about the value we would assign to a leaf; it makes sense to try and minimize C *for that iteration*. So if C is squared error loss, we would assign leaf value = mean(target value). Other cost functions such as log loss do not have analytical solutions and may have to be approximated using Newton's method (C'' needs to be available here).
    - We now know the value that would be applied to a leaf given some partition of the observations. Now we apply a greedy approximation to grow trees. Starting from the top of the tree, we can search through all the possible *binary* partitions for each variable, in order to assess which split would be the "best". In traditional GBDTs, the "best" criteria is always the reduction in **squared error loss** between the target value and the leaf values. We continue to grow the tree level-by-level using this method. The tree size should be controlled through hyper-parameters such as minimum number of obs in node and max depth.
4. The most up-to-date predictions are obtained by summing the leaf values across all the tree learners.
5. This is repeated until we do not see much improvement in `C(response, most up-to-date predictions)`, i.e the *overall* C, usually measured for a hold-out dataset.

\*classification is fit via log-odds, and converted to probabilities at the end.

Traditional GBDT is implemented in the `gbm` package, or the newly-released and much faster `gbm3`.

## xgboost vs traditional gradient boosting

xgboost (and lightgbm, which is built off the same logic) take advantage of C functions that have second derivatives. Instead of only fitting to the gradient descent update values (first order approximations), we are fitting to a **second order approximation of that update value**.

There are two additional differences for GBDT under xgboost as a result of this approximation:
- The leaf values can now always be minimized analytically, but only with respect to the approximated version of C. This may result in slightly different leaf values where C is not squared error loss.
- The splitting criteria is no longer always the reduction in squared error loss between the target value and the leaf values. It is based on a "Gain" score: the reduction in the (approximated) C.

Another small difference to note is that the initial predictions are set at a constant 0.5, instead of the mean of the response values as in traditional GBDT. (For logistic regressions, this means the log-odds initiate at 0). This parameter can be changed but should have no impact on final predictions.

Note that these technical differences still produce the same results if C = squared error loss, as C'' = 1 in this case.

xgboost also introduces additional regularization parameters for the CARTs, such as tree size measured in leaves, and L2 regularization of the leaf values (to prevent predictions from being dominated by 1 tree). These are rarely used in practice though, because they add too many dimensions to the parameter tuning process.
