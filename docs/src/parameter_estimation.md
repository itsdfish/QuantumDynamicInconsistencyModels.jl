# Parameter Estimation

This brief tutorial explains how to performance Bayesian parameter estimation of the QPDM using [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl). One complication in estimating the parameters of the QPDM is that the posterior distributions may have multiple modes, which leads to convergence problems with most MCMC algorithms. Pigeons.jl uses a special type of parallel tempering to overcome this challenge. An additional advantage of using Pigeons.jl is the ability to compute Bayes factors from the log marginal likelihood using the function `stepping_stone`.

## Load Packages

First, we will load the required packages below. 

```julia
using Pigeons
using QuantumDynamicInconsistencyModels
using Random
using StatsPlots
using Turing
```

## Generate Simulated Data

The next step is to generate some simulated data from which the parameters can be estimated. In the code block below, the utility parameter $\mu_d$ is set to one and the entanglement parameter is set to $\gamma = 2$.  A total of 50 trials is generated for each of the three conditions. The resulting values represent the number of defections per condition out of 50.
```julia
Random.seed!(8744)
 parms = (
    α = .9, 
    λ = 2,
    w_win = .5,
    γ = 2.5
)

outcomes1 = [[2,-1],[5,-3],[.5,-.25],[2,-2],[5,-5],[.5,-.50]]
outcomes2 = [[2,-1],[5,-3],[.5,-.25],[2,-2],[5,-5],[.5,-.50]]

n = 10
model = QDIM(; parms...)
data = rand(model, outcomes1, outcomes2, n)
```

## Define Turing Model

The next step is to define a Turing model with the `@model` macro. For simplicity, we will fix the utility parameter $\mu_d=1$ and set the prior of the entanglement parameter to $\gamma \sim \mathrm{normal}(0,3)$. 

```julia 
@model function turing_model(data, outcomes1, outcomes2, n, parms)
    γ ~ Normal(0, 3)
    data ~ QDIM(; parms..., γ)
end
_data = (outcomes1,outcomes2,data,n)
sampler = turing_model(_data, outcomes1, outcomes2, n, parms)
```

## Estimate Parameters

To estimate the parameters, we need to pass the Turing model to `pigeons`. The second command converts the output to an `MCMCChain` object, which can be used for plotting
```julia
pt = pigeons(
    target=TuringLogPotential(sampler), 
    record=[traces])
samples = Chains(sample_array(pt), ["γ","LL"])
```
The trace of the `pigeon`'s sampler is given below:
```julia
────────────────────────────────────────────────────────────────────────────
  scans        Λ      log(Z₁/Z₀)   min(α)     mean(α)    min(αₑ)   mean(αₑ) 
────────── ────────── ────────── ────────── ────────── ────────── ──────────
        2       1.57      -32.7      0.443      0.826          1          1 
        4      0.622      -31.7      0.692      0.931          1          1 
        8       1.07      -31.4      0.574      0.881      0.923      0.991 
       16        0.9      -31.7      0.757        0.9       0.96      0.996 
       32      0.837      -31.7      0.719      0.907       0.99      0.998 
       64       1.06      -31.7       0.72      0.882       0.99      0.998 
      128       1.02      -31.9      0.811      0.887       0.99      0.997 
      256      0.986      -31.9       0.86       0.89       0.99      0.997 
      512      0.968      -31.9      0.873      0.892      0.992      0.998 
 1.02e+03      0.998      -31.9      0.874      0.889      0.994      0.998 
────────────────────────────────────────────────────────────────────────────
```

## Plot Posterior Distribution 

Now we can plot the posterior distribution of $\gamma$ with `plot`. The posterior distribution of $\gamma$ has a primary mode around 1 and secondary modes around 2 and 3.5.
```julia 
plot(samples)
```

![](resources/posterior_gamma.png)