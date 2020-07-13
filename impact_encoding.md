# Impact/target encoding

References:

- [Blog post 1](http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/?bcsi-ac-0e612311b79015a1=2991BF3F00000102jBCX/jEQkmRKZDI7bgnDzW+MzLSPCAAAAgEAAJxtagGEAwAAAAAAAJxNBAA=)
- [Blog post by the creator of the R package vtreat](http://www.win-vector.com/blog/2014/12/a-comment-on-preparing-data-for-classifiers/?bcsi-ac-0e612311b79015a1=2991BF3F00000102ivT9mr8xzJ1Kss38+Fgg1xHH+3p3CwAAAgEAAFDIeAGEAwAAAAAAALNhBQA=)
- Owen Zhang's "How to win Kaggle competitions" presentation

Used to re-encode high-cardinality categoricals into numeric predictors.

At its simplest, impact encoding involves stacking a naive bayes model within the overarching model: a naive bayes model is constructed with the dummified categorical variable as predictors, and the target as is. The (probability) results of the classifier are then used as a numeric predictor representing the original categorical variable.

Leakage: Standard naive bayes would incorporate information from all rows of the data when calculating priors and likelihoods. This may introduce leakage (one of our predictors would already incorporate data from the training target), which will exercabate the strength of the predictor and will not generalise well to unseen data (as the target won't be known beforehand). Owen Zhang suggests using a leave-one-out approach to calculate the impact encoding for each row.

Owen Zhang also suggests adding noise to the impact encoding values in order to avoid over-fitting. Although this has been shown to work in practice, there is no real mathematical/statistical basis for it. The distribution of the random noise added is arbitrary.

Other encoding methods (e.g. linear/quadratic/cubic smoothing, binary representation):
- May aid predictability with some algorithms
- Explainability of variable effects on the target is lost. Encoding method may not properly capture the order and relative value of each of the levels of the categorical variable.
- Harder to explain what the predictor is/represents
