# Implementation details of some ML packages in R which use tree-based methods

Deep dive into some of the maths/stats/logic behind popular R packages for better tuning and understanding of algorithms.

## By method

### Treatment of factors

For rpart, ranger and gbm, documentation suggests that factor splits are determined by applying re-encoding factors as numerics and applying the same (binary) splitting method for numerics:

- Each level is replaced with the mean value of the target for that level, then the variable is ordered and the split cut-off is moved along each unique value of the variable.

#### lightgbm provides the best support for categorical features

For lightgbm, setting factors with `categorical_feature` does a similar encoding but with a slightly different summary value for each level. lightgbm also has additional parameters to control the grouping of categories **before** it applies its encoding method. These parameters are `cat_smooth` (unclear what this actually does) and `min_data_per_group` (controls the minimum number data points a factor level can have).

### Tree growing method

Traditionally, trees are grown "depth-wise", or horizontally, such that the algorithm attempts to split at the current level for as many of the previous nodes as possible, before moving onto the next level.

lightgbm and xgboost both have the option to split their individual trees based on loss reduction instead. For example, if the left-most branch of a tree constantly has the best loss reduction compared to other possible splits, the tree will continue splitting there first. This means the tree can grow to be "unbalanced" (more nodes and leaves on one side compared to the other). The tree structure can then be controlled with `num_leaves`, which limits the final number of leaves. This would improve the fit of the tree and reduce noise in splits: if it can split more on one side because it gives better loss reduction, it will ignore the other side but is still able to end up with a conservative structure (same number of leaves) as a traditional tree.

### Missing value handling

lightgbm and xgboost:
- By default, only treats `NA` values as missing, but can change parameter to include other values (e.g. 0)
- For each split, observations with missing values for that variable are first **excluded** from the data. Then, after the best possible split is chosen, the missing values for that variable are allocated (all together) to the node which results in the better loss reduction.

## By package

### rpart

[Reference](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf)

- Splitting criterion: default set to gini index, can change to information gain in `rpart.control`.
  
### RandomForest

### ranger

### gbm

### xgboost

#### Tree-by-tree diagnostics

xgboost has useful functions for deep diving into the produced tree structure. These are `xgb.model.dt.tree` and `xgb.plot.tree` (which is a visualisation of the former). We can pull some key statistics from the tree structure:

- `Quality` is the reduction in objective achieved by splitting.
- `Cover` is the sum of hessians in each node. This can be compared to `min_child_weight` to ensure the algorithm has behaved properly.
- `Value` is the leaf value, **after adjusting for `eta`**. For binary logistic regression, it is shown as the log-odds.

#### What is min_child_weight and what values should we consider for it when doing grid search?

`min_child_weight` is documented as the sum of hessians within node. For squared error loss (`reg:linear`), hessian = 1, so this simply corresponds to the minimum observations in a node. This is straightforward to set grid search values for.  

For logloss (`binary:logistic`), hessian = p(1-p), where p is the predicted probability. It is important to remember that p at each iteration is always the **most up-to-date prediction of the final probability**. Also, xgboost always begins p at 0.5. So, if we have to sum this over all observations in a leaf, how do we translate desired number of obs in leaf to `min_child_weight`?

Say we want a minimum number of obs in a node, n. The absolute max `min_child_weight` is n * 0.25; avoid going over this value otherwise we'll get nonsensical results because the tree won't split.

Otherwise, in a dataset where the features values are **balanced**, we expect the average values of the hessians to be approx. the mean response of the training data. So another value we can try is n * (event_rate) * (1 - event_rate).

Otherwise, we might have a tolerance for purity -- this means the leaf should not contain too much of observations with the same sign -- to prevent overfitting. When the leaf contains all observations belonging to the same sign, the average hessian in the leaf will be very low. So, say we determine a threshold heuristically (perhaps half the event_rate), the absolute minimum we might try is n * (event_rate / 2) * (1 - event_rate / 2)

Note that these are the possible values we might want to include in a grid search for one given n. We may want to test many n's. We can repeat the above steps for different n, and from all our calculated values, come up with a range of ~4 values to incorporate into our tuning. If we have a wide range of values, we may consider splitting them into two groups (high and low), running the validation through each to see which performs better, then do a finer tuning within the values of the better group.

Finally, don't forget the effect of `subsample`. The estimates for `min_child_weight` should always be multiplied by this parameter.
 
#### lambda and gamma, parameters that are usually left alone

Page 25 of [this link](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf) explains these two parameters well. `gamma` penalizes a high number of leaves, while `lambda` is an L2 regularization co-efficient that penalizes large leaf values. These two hyper-parameters integrate nicely into the tree-learner process, as they become a part of the leaf values, as well as the split criteria. This may be seen as a more direct/scientific approach compared to manually stopping trees growing at a certain depth (`max_depth`). However, they are harder to tune (more possible search values) compared to `max_depth`, and generally do not have a big impact if `max_depth` is set.
