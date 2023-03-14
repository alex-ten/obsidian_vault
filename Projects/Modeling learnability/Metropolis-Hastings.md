Metropolis-Hastings (M-H) is an iterative algorithm that allows us to draw random samples from proposal distributions whose parameters are Markovian states.

# In the `LOTLib` library
Iteration on M-H algorithm in the `LOTLib` library is implemented using Python's [iterator types](https://docs.python.org/3/library/stdtypes.html#typeiter). The library's implementation defines a `MetropolisHastingsSampler` class that defines `__iter__` and `__next__` methods, making it a python [*iterator*](https://stackoverflow.com/questions/2776829/difference-between-pythons-generators-and-iterators). Thus, `MetropolisHastingsSampler` is an object that iterates through a Markovian trace, drawing a sample on every iteration. To work as an iterator, the sampling class must define some state variables, so that it can do something new (but not random) on each iteration and knows when to stop. It must also define the `__next__` method to know what to do on each iteration depending on its state.

## Sampler state
The sampler state is defined upon initialization in the `__init__` method:

```python
def __init__(self, current_sample, data, steps=Infinity, proposer=None, skip=0, prior_temperature=1.0, likelihood_temperature=1.0, acceptance_temperature=1.0, trace=False, shortcut_likelihood=True):
	self_update(self, locals()) # defines attributes such as data and current sample
	
	if proposer is None:
		self.proposer = lambda x: x.propose()
	
	self.samples_yielded = 0
	self.set_state(
		current_sample, compute_posterior = (current_sample is not None)
	)
	self.reset_counters()
```

First, a proposer attribute gets defined as a function that returns a proposal from its input, which is expected to be a `Hypothesis` class that defines a `propose` method.

> [!info] Proposals
> Hypotheses instances define a `propose` method to propose new productions. To use this method, a grammar must be defined. The `propose` method supplies two values: `ret` and `fb` with the help of the `regeneration_proposal` function. The `regeneration_proposal` function takes in (1) the current grammar (`LOTLib.Grammar`) and (2) the current tree (`LOTLib.FunctionNode`) and a new tree along with the difference in log probability between the current and the newly proposed tree. The routine also ensures that the new proposal is not the same as the current one.

Next, the sampler initializes the counter attribute `self.samples_yielded` to 0.

Then, the sampler's state is updated. The sampler's state is equal to the current sample (i.e., old) sample of a M-H iteration. Since in our context, sample are hypotheses, the state of the sampler iterator is a `Hypothesis`. By default, whenever a hypothesis is set as a sampler state, it's posteror is computed. This is possible because data is provided to the sampler upon initialization.

How these samples are drawn and how the sampler's state is updated on each iteration is defined by the `__next__` method (see [source code](https://github.com/piantado/LOTlib3/blob/master/Samplers/MetropolisHastings.py) and the next section).

## Iteration routine

First, the iterator checks if the number of `samples_yielded` does not exceed the designated limit:

```python
if self.samples_yielded >= self.steps:
	raise StopIteration
```

Then a for loop is started on a `range(self.skip + 1)`. Normally, `self.skip = 0` which basically means that the code under the for loop is run once. If, however, skip is set to some $N > 1$, the iteration routine $N$ times, but the iterator will still perform `__next__()` only once.

With these nuances out of the way, here is how the iteration routine performs a single M-H step:
1. Sample new proposal and calculate its probability ratio (fb ratio) relative to the current proposal
```python
self.proposal, fb = self.proposer(self.current_sample)
```
2. Update the proposal's posterior given the current data
```python
self.compute_posterior(self.proposal, self.data)
```
3. Taking into account the temperatur parameter, get "final" posterior scores from current and new proposals:
```python
prop = (self.proposal.prior/self.prior_temperature +
		self.proposal.likelihood/self.likelihood_temperature)
cur = (self.current_sample.prior/self.prior_temperature +
	   self.current_sample.likelihood/self.likelihood_temperature)
```
4. Accept or reject proposal
```python
if MH_acceptance(cur, prop, fb):
	self.current_sample = self.proposal
	self.was_accepted = True
	self.acceptance_count += 1
else:
	self.was_accepted = False

self.proposal_count += 1
self.samples_yielded += 1
return self.current_sample
```
## Acceptance function
The function `MH_acceptance` above returns a boolean, telling the calling script whether to accept the new proposal.
1. It first checks if the current proposal is `nan` or if both posteriors are negative infinity. In that case, acceptance will be random. 
2. It checks if the posterior score of the proposal is `nan` or negative infinity or the `fb` ratio is positive infinity. In that case, always return `False`.

**Basically**: always accept the proposal if it has higher likelihood. Accept with probability equal to `r` if the proposal has lower likelihood; where `r` depends on how much less likely the new proposal is. 

```
F0 > 3 -28.19633492498771 
F0 = 3 -28.19633492498771 
F0 = 5 -28.19633492498771 

F1 < 2 -3.7841996339432615 
F1 = 1 -3.7841996339432615 
```
