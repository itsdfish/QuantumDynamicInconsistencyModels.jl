using SafeTestsets

@safetestset "predict" begin
    using QuantumDynamicInconsistencyModels
    using Test

    model = QDIM(; α = 1.0, λ = 1, m = 0.5, w₁ = 0.75, γ = 1.74)
    outcomes1 = [2, -1]
    outcomes2 = [2, -1]
    preds = predict(model, outcomes1, outcomes2)

    @test preds ≈ [0.70, 0.56, 0.38] atol = 0.01
end

@safetestset "rand" begin
    @safetestset "rand 1" begin
        using QuantumDynamicInconsistencyModels
        using Test
        using Random

        Random.seed!(7878)

        n = 100_000
        n = 100_000

        parms = (α = 0.9, λ = 1, w₁ = 0.5, γ = -1.74)

        model = QDIM(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]

        data = rand(model, outcomes1, outcomes2, n)
        preds = predict(model, outcomes1, outcomes2)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end

    @safetestset "rand 2" begin
        using QuantumDynamicInconsistencyModels
        using Test
        using Random

        Random.seed!(665)

        n = 100_000

        parms = (α = 0.9, λ = 2, w₁ = 0.5, γ = 2.5)

        model = QDIM(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]

        data = rand(model, outcomes1, outcomes2, n)
        preds = predict(model, outcomes1, outcomes2)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end
end

@safetestset "H1" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: make_H1
    using Test

    H1 = make_H1(1.5, 2)
    @test H1[1, 1] == -H1[2, 2]
    @test H1[3, 3] == -H1[4, 4]
    @test H1[1, 2] == H1[2, 1]
    @test H1[4, 3] == H1[3, 4]
    @test H1 == H1'
end

@safetestset "H2" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: make_H2
    using Test

    H2 = make_H2(√(2) * 2)
    @test H2[1, 1] == -2
    @test H2[2, 2] == 2
    @test H2[2, 4] == -2
    @test H2[3, 1] == -2
    @test H2[3, 3] == 2
    @test H2[4, 2] == -2
    @test H2[4, 4] == -2
    @test H2 == H2'
end

@safetestset "logpdf" begin
    using QuantumDynamicInconsistencyModels
    using Test
    using Random

    Random.seed!(410)

    parms = (α = 0.9, λ = 2, w₁ = 0.5, γ = 2.5)

    outcomes1 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]
    outcomes2 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]

    n = 20_000
    model = QDIM(; parms...)
    data = rand(model, outcomes1, outcomes2, n)

    γs = range(0.8 * parms.γ, 1.2 * parms.γ, length = 100)
    LLs = map(γ -> logpdf(QDIM(; parms..., γ), outcomes1, outcomes2, data, n), γs)
    _, mxi = findmax(LLs)
    @test γs[mxi] ≈ parms.γ rtol = 1e-2

    αs = range(0.8 * parms.α, 1.2 * parms.α, length = 100)
    LLs = map(α -> logpdf(QDIM(; parms..., α), outcomes1, outcomes2, data, n), αs)
    _, mxi = findmax(LLs)
    @test αs[mxi] ≈ parms.α rtol = 1e-2

    λs = range(0.8 * parms.λ, 1.2 * parms.λ, length = 100)
    LLs = map(λ -> logpdf(QDIM(; parms..., λ), outcomes1, outcomes2, data, n), λs)
    _, mxi = findmax(LLs)
    @test λs[mxi] ≈ parms.λ rtol = 1e-1

    w₁s = range(0.8 * parms.w₁, 1.2 * parms.w₁, length = 100)
    LLs = map(w₁ -> logpdf(QDIM(; parms..., w₁), outcomes1, outcomes2, data, n), w₁s)
    _, mxi = findmax(LLs)
    @test w₁s[mxi] ≈ parms.w₁ rtol = 1e-2
end

# @safetestset "pdf" begin
#     using QuantumDynamicInconsistencyModels
#     using Distributions
#     using Test
#     using Random 

#     Random.seed!(11214)

#     n = 5

#     model = QDIM(;μd=.5, γ=2)
#     data = rand(model, n)

#     for _ ∈ 1:10
#         μd = rand(Uniform(-1, 1))
#         γ = rand(Uniform(-2, 2))
#         LL1 = logpdf(QDIM(;μd, γ), n, data)
#         LL2 = log(pdf(QDIM(;μd, γ), n, data))
#         @test LL1 ≈ LL2
#     end
# end

@safetestset "get_utility_diffs" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: get_utility_diffs
    using Test

    model = QDIM(; α = 0.65, λ = 1.6, w₁ = 0.5, γ = 2.2)
    outcomes1 = [2, -1]
    outcomes2 = [2, -1]
    d = get_utility_diffs(model, outcomes1, outcomes2)

    @test d[1] ≈ 0.162 atol = 0.001
    @test d[2] ≈ 0.8447 atol = 0.001
end

@safetestset "get_utility" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: get_utility
    using Test

    utility = get_utility(4, 0.5, 2)
    @test utility ≈ 2

    utility = get_utility(-4, 0.5, 2)
    @test utility ≈ -4
end
