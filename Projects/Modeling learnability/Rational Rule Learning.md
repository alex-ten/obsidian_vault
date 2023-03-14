LP was modeled as the change in % correct. We are planning to investigate a more explicit learning model.

If we assume that learning is a process in which the agent observes data to decrease entropy of posterior distribution over hypotheses, we can imagine. When this entropy gets reduced, we experience LP. 

---
Irrationality could be explained by irrational priors, rather than irrational updating.

---

# MCMC (Metropolis-Hastings) posterior estimation

Starting from a randomly sampled hypothesis $H_0$, we use the [Metropolis-Hastings](https://joa.sh/posts/2016-08-21-metropolis.html) algorithm to traverse the infinite space of hypotheses defined by the generative grammar. The reason we can do this at all is the inductive bias introduced by the generation process. Otherwise, how can you even think of traversing an infinite space? The reason we can do this effectively is that different hypotheses (i.e., rules) can be meaningfully compared by their weighted likelihoods (likelihoods weighte by prior probabilities).

Here's how the algorithm works:
1. For `steps` number of iterations do:
	1. Sample a new hypothesis $H^*$ (called the proposal hypothesis)
	2. Given the data, compare the weighted likelihoods, $\eta(H_0|D) = \mathcal{L}(H_0|D)\pi(H_0)$ and $\eta(H^*|D) = \mathcal{L}(H^*|D)\pi(H^*)$. This can be done by looking at the ratio $\mathcal{R} = \frac{\eta(H^*|D)}{\eta(H_0|D)}$
		1. If $\mathcal{R}$ is > 1, we "accept" $H^*$ and it becomes our new working hypothesis
		2. If $\mathcal{R}$ is < 1, we may or may not accept $H^*$, depending on how much less likely it is from $H_0$. In fact, $H^*$ will be accepted despite it being less likely with probability equal to $\mathcal{R}$

After running all iterations, we can count how many times different hypotheses were accepted. The resulting distribution is supposed to approximate the posterior. 