# Bayesian Optimisation

## Why BO

This is a technique that can be used to optimise machine learning algorithms with numerous parameters. It is usually used in deep learning (for neural networks), but can be applied to other ML methods such as gradient boosting to improve performance. For example, `xgboost` has the following optimisable paramters:

	1. Individual tree depth / number of leaves 
	2. Minimum number of observations / minimum hessian in a leaf
	3. Row subsampling percentage for each tree / for each split
	4. Column subsampling percentage for each tree / for each split
	5. Combination of shrinkage and number of trees
	6. Gamma (affects split criteria) 
	7. Lambda and Alpha, individual tree regularisation parameters

Usually, items 1 to 5 are optimised via **grid search**, which involves tracking model performance under all the different possible combinations of a chosen subset of values for each parameter. The number of required searches blows up quickly even for the first 4 items. Trialling the following:

	1. 4, 6, 8, 10
	2. 50, 500, 1000, 2000
	3. 0.5, 0.6, 0.7, 0.8
	4. 0.4, 0.6, 0.8

seems like a pretty high-level, "first-go" search, but already requires 4 * 4 * 4 * 3 = 192 searches. Since gradient boosting algorithms like `lightgbm` and `xgboost` are actually quite quick to train / validate, this particular example isn't too bad in terms of computing time for small-medium datasets. The real limitations are:

- What about the parameter values that occur in between the selected ones? Trying to interpolate between the values doesn't sound like a good idea for something as complex as gradient boosting, especially when the parameters interact with each other.
- By choosing the search with the best result and claiming it is the "best", there is a chance that those parameters were only chosen by chance, and may be "overfit" to the data you use for validation.
- What about the parameters we're not testing? How can we be comfortable that items 5 to 7 won't have an impact on the performance? Traditionally, the reason items 5 to 7 are not tested is because they are all real-valued parameters: it is too hard to choose a suitable subset to grid search over, and if we attempt to search a wider range of values the number of searches required will be very large.

 This is where Bayesian Optimisation (BO) comes in. The entire algorithm, including the underpinning maths, is quite heavy. This knowledge dump will outline the main concepts so that we understand how BO works at a high level, and what each of the arguments in the BO functions available in Python and R are for.

## High-level Bayesian Optimisation concepts
 
We can break BO into two components, the optimisation part, and the bayesian part.
 
### The Bayesian part

BO requires the assumption that there is a (smooth) functional relationship between the parameters and the validation score. 

If we were just considering things on a 2-D plane, this just means that values of x that are close together should give a similar value for y. 

In the case of higher dimensions (like the 7 dimension case we have with gradient boosting, with y being the validation score), this means that points that are close together in "space" should give a similar output for the validation score. This helps us to be comfortable with what we decide is the best combination of parameters - this combination is the point that maximises the function.

The functional relationship assumed in BO is a **Gaussian Process**. If you would like a deep dive into the mathematical underpinnings of a Gaussian process, check out [this link](http://katbailey.github.io/post/gaussian-processes-for-dummies/) for a medium-level overview and [this link](http://www.cs.ubc.ca/~nando/540-2013/lectures/l6.pdf) ([lecture link](https://www.youtube.com/watch?v=4vGiHC35j9s)) for a rigorous explanation.

#### Gaussian Processes (GP)
At a high level, we need to note the following points about the GP:

- The GP is our "prior" of the relationship between the parameters and the validation score. 
- The GP is parameterised by a **mean** function and a **covariance function**, similar to a multivariate Gaussian distribution.
- The covariance function can be based on different types of **kernels**, and these kernels can be **paramaterised**.
- Usually, we can just opt for the default definition of the covariance function and its parameterisation.

### The Optimisation part

[This article](http://krasserm.github.io/2018/03/21/bayesian-optimization/) outlines the optimisation part well. The optimisation process can be summarised at a high level as the following:
1. Get the validation score for a few random searches over the hyperparameter space. These are the initial points that will be used to define the initial mean function and covariance function of the GP (the prior). The number of initial evaluations to do is a parameter that could be chosen, but the defaults are probably a reasonable number so it should be fine to leave.
2. The optimisation process uses an **acquisition function** to determine what area of points to trial next. The acquisition function is a mathematical metric based on the GP that can be computed very quickly on all possible parameter combinations. 

	Basing it on the GP here is useful because: since we are assuming a prior distribution for our underlying function, we can compute probabilities and confidence intervals on the validation score for all the possible paramater (given our known points). 
	
	The acquisition function usually incorporates a trade-off between the expected value and the variance of the validation score to determine where to trial next. Research by academics have found some robust and well-performing acquisition functions; whatever package we are working with with will probably have one of these as the default, so we should not have to change this argument from its default.

3. Once the acquisition function determines the next best hyper-parameter combination, this combination is evaluated for the true validation score.
4. Now we have another real point. Our prior (the GP) is then updated (i.e. the mean and covariance will change) to reflect this additional knowledge.
5. Steps 1-4 are repeated until the algorithm determines that convergence has been reached. This is usually via a criteria placed on the acquisition function; for example, *if the probability of improvement of trialling the next best point is below 5%, we have reached the best*. 

	This criteria is an argument that can potentially be tweaked if the first full run of the BO isn't convincing (e.g. the validation score doesn't look like it has converged to the highest possible).

A final note: the above process is done under the over-arching assumption that whenever we evaluate the validation score on a set of hyper-parameters, the evaluation is **noiseless**. 

This is also an argument in most BO optimisers, and it should be set to TRUE (noiseless) if we are using the same validation data each time (and random seed if cross-validation is being employed). There are techniques to account for noisyness when evaluating the validation score, but these may require additional computation time and the gain from using the method may not be significant.

## Concluding thoughts

BO is essentially an algorithm-aided grid search: it starts with a random combination of parameters, then directs itself to the spots where it is mathematically likely to get a better score (relying on the "prior" - the Gaussian Process - to do this). It continues to do this until it hits an area where it consistently gets high scores, and other areas are not likely to be any better. This not only saves time as less computations of the validation score are required (i.e. fewer full trainings of the model are required), but also allows us to try and optimise real-valued parameters with wide ranges (e.g. items 5 to 7 for gradient boosting).

Luckily, most of the arguments that need to be passed to BO functions/libraries can be left as default values. 

## Additional resources for learning: 

- [Noisy vs Noiseless GP regression](https://www.cs.princeton.edu/~bee/courses/scribe/lec_10_16_2013.pdf) 
- [Bayesian optimisation, focusing on different acquisition functions](https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf)
- [End to end view of the optimisation part of BO](https://arxiv.org/pdf/1012.2599.pdf)
- [Covariance functions](https://en.wikipedia.org/wiki/Covariance_function)