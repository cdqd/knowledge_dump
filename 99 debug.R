### xgboost test

library(xgboost)

set.seed(242)
bst <- xgb.train(data = xgb.DMatrix(data.matrix(train.x.dummy),
                                    label = as.numeric(train.y == "Died")),
                 booster = "gbtree",
                 objective = "binary:logistic",
                 # objective = "reg:linear",  # min_child_weight is number of obs
                 nround = 25,
                 max_depth = 1,
                 # eta = 0.01,  # eta affects value shown in leaf, but doesn't affect cover (min_child_weight)
                 eta = 1,
                 min_child_weight = 15,
                 colsample_bytree = 0.7,
                 # subsample = 0.8,  # subsample is not exact - just assigns random values and picks above threshold, so obs selected may deviate from x% of total obs
                 subsample = 1)

# check min_child_weight:
tree <- xgb.model.dt.tree(bst$feature_names, bst)
# cover0 <-unique(tree$Cover)  # min child weight = 20....20.772
# cover1 <- unique(tree$Cover)  # min child weight = 30....30.2187

match(min(cover1), cover0)  # NA - so a new/different split has been made!

# First tree -----------------------
# Gradient boosting is initialized with f_0(x) = 0.5 for all x
# Tree 0:
xgb.plot.tree(bst$feature_names, bst, trees = 0)  # splits at Sex
# Cover (node) = 222.75
891 * (0.5) * (1 - 0.5)  # 222.75
# so it looks like the Covers (sum of hessians) for the first tree (f_1(x)) are based on the predicted values in f_0(x)

# predicted values from Tree 0: sex_female = 0 -> value = 1.02926
#                               sex_female = 1 -> value = -0.578616

1/(1 + exp(-0.578616))  # transform log odds, 0.6407489
mean(as.numeric(train.y == "Survived")[train.x.dummy$sex_female == 1])  # 0.6464968

preds1 <- predict(bst, xgb.DMatrix(data.matrix(train.x.dummy)), ntreelimit = 1)

# Second tree ------------------------------
xgb.plot.tree(bst$feature_names, bst, trees = 1)  # splits at deck_T
# Cover (node) = 184.182
sum(preds1*(1-preds1))  # 184.1823, consistent with previous observation

# predicted values from Tree 0: deck_T = 0 -> value = -1.02925
#                               deck_T = 1 -> value = 0.414787

1/(1 + exp(-1.02925))  # 0.7367705
mean(as.numeric(train.y == "Survived")[train.x.dummy$deck_T == 0], na.rm = T)  # 0.679803

preds2 <- predict(bst, xgb.DMatrix(data.matrix(train.x.dummy)), ntreelimit = 2)

1 / (exp(-(0 + 1.02926 + 0.414787)) + 1)  #0.8090806
unique(preds2)  # 0.8090807
# 0.5 base prediction score means that the base log-odds = 0.

# Check Tree 3 to make sure that that hessian includes all predictions up to current tree.
sum(preds2*(1-preds2))  # 166.0389
xgb.plot.tree(bst$feature_names, bst, trees = 2)  # splits at p.class 
# Cover (node) = 166.039