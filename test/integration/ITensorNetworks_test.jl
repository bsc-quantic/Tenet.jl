@testset "ITensorNetworks" begin
    using Tenet
    using Graphs
    using ITensors: ITensors, ITensor, Index, array
    using ITensorNetworks: ITensorNetwork

    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(4, "k")
    l = Index(5, "l")
    m = Index(6, "m")

    a = ITensor(rand(2, 3), i, j)
    b = ITensor(rand(3, 4, 5), j, k, l)
    c = ITensor(rand(5, 6), l, m)
    itn = ITensorNetwork([a, b, c])

    tn = convert(TensorNetwork, itn)
    @test tn isa TensorNetwork
    @test issetequal(arrays(tn), array.([a, b, c]))

    itn = convert(ITensorNetwork, tn)
    @test itn isa ITensorNetwork
    @test issetequal(map(v -> array(itn[v]), vertices(itn)), array.([a, b, c]))

    tn = convert(Quantum, itn)
    @test tn isa Quantum
    @test issetequal(arrays(tn), array.([a, b, c]))

    itn = convert(ITensorNetwork, tn)
    @test itn isa ITensorNetwork
    @test issetequal(map(v -> array(itn[v]), vertices(itn)), array.([a, b, c]))

    tn = convert(Ansatz, itn)
    @test tn isa Ansatz
    @test issetequal(arrays(tn), array.([a, b, c]))

    itn = convert(ITensorNetwork, tn)
    @test itn isa ITensorNetwork
    @test issetequal(map(v -> array(itn[v]), vertices(itn)), array.([a, b, c]))
end
