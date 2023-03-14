# Notes on GPR

## GPR is a cognitive model for learning in large-ish spaces
As far as I understand, Wu et al. use a Gaussian Process Regression (GPR) as a cognitive model for the process of learning the latent reward distribution structure. The grid-bandit is an environment from which participants can extract rewards. The objective grid structure is more or less smooth, so that cells that are spatially close provide similar payouts compared to distant cells. The size of the grid makes it hard to explore, but learners who can generalize can infer the underlying reward structure.

## **GPR has prior (uncertain) expectations**
Initially, the participant is assumed to know nothing about a given grid. In the model, this is expressed as a highly uncertain distribution over expected reward values on each cell. More precisely, GPR starts with high prior uncertainty over possible functions. These functions map grid coordinates $c = (x, y)$ onto a real reward values, $f : c = [x, y] \in \mathbb{R}^2 \rightarrow r \in \mathbb{R}$. Note that when we talk about *uncertainty over functions* in GPR, we are not talking about uncertainty over function parameters or its form (like in normal regression). Rather, we refer to the uncertainty of predicting $r$ from $c$ -- the predictive uncertainty. Thus, before receiving any information about the grid, the learner (and the model) expect different mappings from $r$ to $c$ with equally uncertainty. In other words, they do not expect any reward value with high certainty from any cell. In other words, they are oblivious about the reward structure of the grid.

## **GPR learns and generalizes**
When a user first samples from a cell (in no-memory condition), the cell payout value is revealed and the uncertainty about this cell is reduced. The equivalent of this *learning* event in GPR is the conditioning of expectations on observed data. In practice, the observed value of the probed cell constrains the set of possible functions, so that they have to be such that they produce the observed reward $r$ at the coordinate $c$. The prior knowledge about reward distributions becomes posterior knowledge. However, by revealing the value of a cell, the GPR learns more than a value of a single cell. A crucial feature of GPR is its ability to generalize, formalized by the radial basis kernel function (RBF). The RBF kernel is another aspect of prior knowledge. It is an inductive bias that tells the model to expect similar rewards for close-by coordinates. Thus, by revealing a single cell on an otherwise unknown grid, RBF updates reward expectations not only for the revealed cell, but for the all other cells on the grid. Values for these unobserved cells are updated depending on how far they are from the observed one.

## **Generalization in GPR is quantifiable**
One quantitative feature of this spatial-generalization bias is its strength. If the bias is very strong, the model expects closely located cells to have very similar rewards. On the other hand, a model with a weak spatial inductive bias would not expect nearby cells to have similar rewards (it is not biased like that). This variability in strength of the spatial generalization bias is captured by the so-called *length* parameter of the RBF kernel. Wu et al., fit this parameter to behavioral data consisting only of observed choices.

## **GPR guides decision-making**
Notably, while GPR does not model choice probabilities, it is not unrelated to decision-making. Specifically, GPR serves as a basis for decision-making, as it is assumed that decisions in the grid-bandit are based on the inferred reward values of grid cells. In  Wu et al., GPR is the basis for UCB sampling, as is provides inferences about reward expectations as well as uncertainties about these expectations -- the two components of the UCB utility function.

## **GPR can be forgetful**
It is possible to like the predictive uncertainty parameter of the GPR with "forgetfulness". If forgetting signifies the deterioration of knowledge, then it can be formalized as increasing uncertainty about a previously observed cell. Wu et al., assume that forgetting is a function of time since the cell was last visited, possible moderated by surprise -- quantified as expectation violation. That is, all else being equal, a cell observed a long time ago has more uncertainty compared to a cell observed recently. At the same time, a cell observed a long time ago that was surprising (yielded much more than expected) has less uncertainty than a cell observed a long time ago that was not surprising.

# Implementing a GPR-UCB explorer
## 1. Generating grid-bandits
The first step in implementing a GPR-UCB explorer is to generate an environment to explore. A particular grid environment is described by a multivariate Gaussian of $N \times N$ dimensions. For example, a $4 \times 4$ grid is a set of 16 (Gaussian) random variables that can be described by $\mu \in \mathbb{R}^{16}$ and $\Sigma \in \mathbb{R}^{4 \times 4}$. In code [env_generation/samplePrior2D.py], this is done by doing:
```python
s = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
```
where `K` is the covariance matrix created with an RBF kernel:
```python
k = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=1)
K = k.K(X)
```
The variable `X` is a 2-dimensional array with $N$ rows and 2 columns. Each 2-element row contains a pair of coordinates from the bandit grid e.g.,:
```python
X = [[0, 0], [0, 1], [0, 2],..., [0, N],  
	 [1, 0], [1, 1], [1, 2],..., [1, N],
	 ...,
	 [N, 0], [N, 1], [N, 2],..., [N, N]]
```
 `K = k.K(X)`  creates the covariance matrix based on given parameters `variance=1` and `lengthscale=1`.  The resulting covariance matrix is $N \times N$, where each $i,j$-th element is equal to the covariance between coordinates $i$ and $j$.  The generated environments are stored in json format:
```json
{"0": {
	"0": {
		"x2": 0, 
		"y": 0.0, 
		"x1": 0
	}, 
	... 
	"63": {
		"x2": 7, 
		"y": 0.067173192882018229, 
		"x1": 7
	},
 "39": {...}
}
```
There are several versions, or samples, of an environment with certain parameters (grid size, kernel lengthscale) etc.
## 2. Simulating choices with known parameters
Next, we want to simulate choices using a GPR-UCB agent with known parameters. A few steps are required. First, we define sampling distributions to sample model parameters from:
```R
lambda   <- rlnorm(1,.1,.5)      # Random value from Log Normal
beta     <- rlnorm(1,-1.55,.5)   # Random value from Log Normal
tau      <- rexp(1,33)           # Random value from Exponential
memory_w <- rgamma(1, 25, 1)     # Random value from Gamma

# Define parameter vector for the gpr function
parVec_gpr <- c(lambda, lambda, 1, 0.001)
```
The parameter vector `parVec_gpr` later enters the `gpr()` function that outputs the predicted expected values (and variance) of each cell on a grid:
```R
post <- gpr(
	X.test = bandit_grid,
	theta = parVec_gpr,
	X = cbind(obs$x1,obs$x2),
	y = ((obs$y-50)/100), k = rbf) # scale y observations to zero mean and variance of 1
```
It is important to understand what the function does at least at the high level, so let's spend some time on it. Starting with the parameters: 
- `X.test` is an `R` list with 2 vector elements: coordinates for `x1` and `x2`, created by `bandit_grid <- expand.grid('x1'=0:(GRIDSIZE-1), 'x2'=0:(GRIDSIZE-1))`. Effectively, `bandit_grid` is a table where each row contains a unique pair of coordinates from the grid bandit.
- Parameter `theta` is a vector of GPR parameters. Note that `lambda` is repeated twice in this vector. This is because the RBF function assumes separate lengthscale parameters for each dimension, and the grid has two dimensions. In practice, the same value is used for each dimension. The next parameter whose value is set to `1` is observational variance (kept constant), and the last one (0.001) is noise variance (changes with time).
- The variable `X` (not to be confused with `X` from the previous section) is a pair of coordinates, and `y` is the reward observed for these coordinates. On the first iteration of the simulation, `X` is sampled randomly. On subsequent iterations, `X` and `y` expand with new values from new observations and new coordinates (outputted by the agent). 

Using these parameters, `gpr()` does the following:
```R
# 1. Calculate covariance between visited locations
K <- k(X, X, theta)
# 2. Create a diagonal matrix containing added noise consisting of error variance and default variance
D <- (diag(rep(1, num_obs)) * err_var) + diag(rep(1,num_obs)) * d_default
# 3. Add noise to observations
KD <- K + D
# 4. Perform matrix inversion
KK.inv <- inverse(KD)
# 5. Infer mean and variance
post <- apply(X.test, 1, inference_function, inv.cov = KK.inv)
```

It is not strictly necessary to understand the details of how the `gpr()` function does what it does (I don't understand all of it), but more details can be found in this [distill article](https://distill.pub/2019/visual-exploration-gaussian-processes/#FurtherReading), and [this blog](https://peterroelants.github.io/posts/gaussian-process-tutorial/).

The `post` variable is a data frame that contains posterior means and variances for each cell of the grid, inferred from data supplied by `X` and `y` and an inductive bias in the form of `theta` that was randomly sampled previously. These predictions then enter a UCB function that returns utility for each cell based on the `post` predictions. The UCB function takes in an additional free parameter `beta` that controls how much influence posterior prediction variance has on utility. The resulting utility values are then converted to sampling probabilities using the softmax function, which uses another free parameter `tau` that controls how much utility scores influence choice probabilities that are otherwise uniformly random:
```R
# Upper confidence bound (UCB): expected value plus beta-scaled SD
ucb_utils <- post$mu + beta * sqrt(post$sig) 

# Compute softmax choice probabilities
p <- exp(ucb_utils/tau)
p <- p/sum(p)
```
The new `location` is then used to draw a new observation and the cycle continues until the horizon is reached:

```R
# Sample next choice
location <- sample(1:GRIDSIZE^2,1, prob=p, replace=TRUE)

# Generate reward
reward <- env.gen_reward(location)

# Update history of observations for posterior inference on the next iteration
x1 <- c(x1, bandit_grid[location, 'x1'])
x2 <- c(x2, bandit_grid[location, 'x2'])
y <- c(y, reward)
```

The novel part of Sebastian's work is the addition of uncertainty scaling. Previous work assumed that posterior uncertainty changes only with observations. The novel work assumes that uncertainty varies with time: expectation of previously visited cells become more uncertain with time, even if these cells have been previously sampled. This is implemented with a `gp_error_variance_exponential()` function:
```R
# Update value of the variance parameter (last element of `parVec_gpr`)
parVec_gpr[-1] <- gp_error_variance_exponential(
	obs = data.frame(x1,x2,y),
	theta = parVec_var,
	clicks = NUM_CLICKS
)
```
Where `parVec_var` is a vector that contains parameters that control how variance is scaled. To implement forgetfulness, randomly drawn parameter `memory_w` is added to this vector. Here is how  variance scaling happens:
```R
gp_error_variance_exponential <- function(obs, theta, clicks, prior_mean=0.5, default_noise=0.0001) {
	# Recency ranging between [0, 1]
	time <- 1:nrow(obs)
	f_recency <- (max(time) - time) / clicks

	# Linear surprise ranging between [0, 1]
	f_surprise <- 1 - abs((obs$y - prior_mean) * 2)
	
	# Combine recency and surprise 
	feature_mat <- cbind(c(f_recency),c(f_surprise))
	features_x_weights <- feature_mat %*% theta
	error_var <- exp(features_x_weights)
	return(c(error_var))
}
```
First, we calculate the `f_recency` vector that has an entry for each observation (this vector grows by 1 value with each iteration). Recent observations are closer to 0 and more distant ones are closer to 1. These `f_recency` values are proportional to how much uncertainty we want to add to posterior predictions, and we want to add more uncertainty to predictions about cells that we visited farther in the past.

Then, we calculate the `f_surprise` vector. In this current implementation, observer rewards `obs$y` are compared to the prior mean of the entire grid, which is equal to 0.5. This reference point is assumed to stay the same throughout exploration. 

Finally, the two vectors are combined into a single matrix and each is scaled (element-wise) by its respective free parameter. The scaled values are then exponentiated and returned. The returned values

## 3. Parameter recovery
Parameter recovery is a model validation procedure that fits parameters to simulated data. Since the parameters that simulated the data are known, we can try to *recover* them by inferring their most likely values given the data.

In practice, the recovery is done through evolutionary optimization `DEoptim` of the negative log likelihood function of model parameters. Recall that likelihood is a function of model parameters given data, $\mathcal{L}(\theta \mid d)$. The `DEOoptim` algorithm repeatedly evaluates a generation of randomly mutated "guess" parameters until it finds a set of parameters that are locally optimal. Thus, the basic idea is perform the following steps:
1. Define a fitness function that maps model parameters to likelihood (given a dataset) 
2. Select data simulated by a certain known model
3. Initialize a set (generation) of initial parameter values
4. Then, itiratively:
	4.1. Evaluate fitness of the current generation of parameters
	4.2. Mutate and select best fitting parameters

Excluding nonessential details, the fitness function looks as follows:
```R
modelFit <- function(
	par,			# current generation of parameters
	acquisition,    # function that outputs selecton probabilities (UCB)
	k,				# covariance function for GPR (RBF)
	subjD,			# data from a subject (contains multiple rounds)
	rounds,			# set of rounds to fit parameters to
) {
	# Define parameter vectors for memory function and GPR
	lambda <- exp(par[1])
	beta <- exp(par[2])
	tau <- exp(par[3])
	recency <- (par[4])
	surprise <- (par[5])
	
	parVec_gpr <- c(lambda, lambda, 1, DEFAULT_ERR_VAR)
	parVec_mem <- c(resency, surprise)
	parVec_gpr[-1] <- gp_error_variance_exponential(obs, theta=parVec_mem, clicks=NUM_CLICKS)
	
	# For each round (grid) find likelihood
	for (r in rounds) {
		post <- gpr(X.test=Xnew,  theta=parVec_gpr, X=X_t, y=y1, k=k)
		u <- acquisition(post, c(beta))
		p <- softmax(u, tau)
		
		# Likelihood of parameters given one round of choices
		choiceData = subset(subjD, round==r)
		negLogLik <- -sum(
			log(p[, choiceData])
		)
	}

	# Sum of negative log likelihoods over multiple rounds
	return (sum(nLL))
}
```

Then the `modelFit` function is fed into the optimization algorithm that returns a list object from which we extract fitted parameters:
```R
fit <- DEoptim(modelFit, lower, upper, DEoptim.control(itermax=MAX_ITER_FITTING),
	k = rbf,
	acquisition = ucb,
	subjD = subjectData,
	rounds = trainingSetSize
)
paramEstimates <- fit$optim$bestmem
```
Note that, `DEoptim` takes care of parameter initialization so nothing is passed to the `par` parameter of `modelFit`. Instead, we specify `lower` and `upper` bounds for parameter initialization.

Also note that `rounds` is set to `trainingSetSize`. This is because we only use a subset of all available rounds to fit model parameters. The rest of the data is used to evaluate the fit:
```R
nLL <- modelFit(
	par = paramEstimates,
	subjD = subjectData,
	acquisition = ucb,
	k = rbf,
	rounds = testSet
)
```
We then store the resulting `nLL` value for model comparisons.

One important detail that I have omitted for simplicity is model versioning. The code above assumes that we are dealing with a "full model" that includes both the surprise and recency parameters. We can modify the code to fit the given data to different model forms. Model forms differ from each other by which of the possible free parameters are kept constant (e.g., 0):
```R
if (model_version ==  "full") {
	recency <- (par[4])
	surprise <- (par[5])
} else  if (model_version ==  "recency") {
	recency <- (par[4])
	surprise <-  0
} else  if (model_version ==  "surprise") {
	recency <-  0
	surprise <- (par[4])
}
```

During model validation, it is important to make sure that best the best fitting model corresponds to the model that simulated the data. By fitting parameters from different model forms to data simulated by a particular model, we can evaluate whether we can recover not only the parameter values, but also the model form.

## 4. Fitting parameters to human data

As it should be, the fitting procedure is the same for simulated and human data. The same fitting routine is defined to fit parameters to human data.