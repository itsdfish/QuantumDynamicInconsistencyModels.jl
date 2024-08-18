# Overview

This page provides an overview of the API along with examples. 


# Setup Two-Stage Gamble 

To begin, we will create six two-stage gambles. The first and corresponding second stage gambles are the same. The vector `win_gamble1` indicates whether the outcome from the first gamble is a win or a loss. Each two-stage gamble will be repeated 10 times, as indicated by `ns`.

```@example  basic_usage
outcomes1 = [[2,-1],[6,-3],[3,-1],[7,-2],[.50,-.75], [2,-3]]
outcomes2 = [[2,-1],[6,-3],[3,-1],[7,-2],[.50,-.75], [2,-3]]
win_gamble1 = [true, false, true, true, false, true]
ns = fill(10, 6)
``` 

# API 

## Create Model 

In the code block below, we will create a model object for QDIM.
```@example  basic_usage
using QuantumDynamicInconsistencyModels 

model = QDIM(;  
    α = .9, 
    λ = 2,
    w₁ = .5,
    m = .30,
    γ = 2.5
)
```

## Simulate Model

The code block below demonstrates how to generate simulated data from the model using `rand`. In the example, we will generate 10 simulated responses for each gamble. 
```@example  basic_usage
data = rand.(model, outcomes1, outcomes2, win_gamble1, ns)
```

The variable `data` is a vector of vectors in which each sub-vector is a sample of response frequencies for each two-stage gamble. The frequencies follow a multinomial distribution where elements of each sub-vector correspond to the following

1. frequency of planning to accept second gamble and accepting second gamble
2. frequency of planning to accept second gamble and rejecting second gamble
3. frequency of planning to reject second gamble and accepting second gamble
4. frequency of planning to reject second gamble and rejecting second gamble

## Evaluate Log Likelihood

The log likelihood of data can be evaluated using `logpdf`. In the code block below, we valuate the loglikelihood for each two-stage gamble: 
```@example  basic_usage
logpdf.(model, outcomes1, outcomes2, win_gamble1, ns, data)
```