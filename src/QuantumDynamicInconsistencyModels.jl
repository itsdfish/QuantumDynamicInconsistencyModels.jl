module QuantumDynamicInconsistencyModels

    using Distributions: Multinomial
    using Distributions: ContinuousUnivariateDistribution
    using LinearAlgebra
    using PrettyTables

    import Distributions: logpdf
    import Distributions: pdf
    import Distributions: rand
    import Distributions: loglikelihood

    export get_expected_utility
    export predict
    export loglikelihood
    export logpdf
    export pdf
    export QDIM
    export rand

    include("structs.jl")
    include("function.jl")
    include("utilities.jl")

end
