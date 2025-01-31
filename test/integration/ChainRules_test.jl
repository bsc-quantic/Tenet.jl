using Test
using Tenet
using ChainRulesTestUtils

@testset "Tensor" begin
    test_frule(Tensor, ones(), Symbol[])
    test_rrule(Tensor, ones(), Symbol[])

    test_frule(Tensor, ones(2), Symbol[:i])
    test_rrule(Tensor, ones(2), Symbol[:i])

    test_frule(Tensor, ones(2, 3), Symbol[:i, :j])
    test_rrule(Tensor, ones(2, 3), Symbol[:i, :j])
end

@testset "TensorNetwork" begin
    # TODO it crashes
    # test_frule(TensorNetwork, Tensor[])
    # test_rrule(TensorNetwork, Tensor[])

    @testset "equal ndims" begin
        a = Tensor(ones(4, 2), (:i, :j))
        b = Tensor(ones(2, 3), (:j, :k))

        test_frule(TensorNetwork, Tensor[a, b])
        test_rrule(TensorNetwork, Tensor[a, b])
    end

    @testset "different ndims" begin
        a = Tensor(ones(4, 2), (:i, :j))
        b = Tensor(ones(2, 3, 5), (:s, :k, :l))

        test_frule(TensorNetwork, Tensor[a, b])
        test_rrule(TensorNetwork, Tensor[a, b])

        a = Tensor(ones(4, 2), (:i, :j))
        b = Tensor(ones(2, 3, 5), (:j, :k, :l))

        test_frule(TensorNetwork, Tensor[a, b])
        test_rrule(TensorNetwork, Tensor[a, b])
    end
end

@testset "conj" begin
    a = Tensor(rand(4, 2), (:i, :j))
    b = Tensor(rand(2, 3), (:j, :k))

    tn = TensorNetwork([a, b])

    @testset "Tensor" begin
        test_frule(Base.conj, a)
        test_rrule(Base.conj, a)
    end

    @testset "TensorNetwork" begin
        test_frule(Base.conj, tn)
        test_rrule(Base.conj, tn)
    end
end

@testset "merge" begin
    a = Tensor(rand(4, 2), (:i, :j))
    b = Tensor(rand(2, 3), (:j, :k))

    test_frule(merge, TensorNetwork([a]), TensorNetwork([b]))
    test_rrule(merge, TensorNetwork([a]), TensorNetwork([b]); check_inferred=false)
end

@testset "contract" begin
    @testset "unary" begin
        @testset "real" begin
            x = Tensor(fill(1.0), Symbol[])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(ones(2), Symbol[:i])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(ones(2, 3), Symbol[:i, :j])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(ones(2, 3), Symbol[:i, :j])
            test_frule(contract, x; fkwargs=(dims=[:i],))
            test_rrule(contract, x; fkwargs=(dims=[:i],), check_inferred=false)
        end

        @testset "complex" begin
            x = Tensor(fill(1.0 + 1.0im), Symbol[])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(fill(1.0 + 1.0im, 2), Symbol[:i])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(fill(1.0 + 1.0im, 2, 3), Symbol[:i, :j])
            test_frule(contract, x)
            test_rrule(contract, x; check_inferred=false)

            x = Tensor(fill(1.0 + 1.0im, 2, 3), Symbol[:i, :j])
            test_frule(contract, x; fkwargs=(dims=[:i],))
            test_rrule(contract, x; fkwargs=(dims=[:i],), check_inferred=false)
        end
    end

    @testset "binary" begin
        @testset "real" begin
            # scalar-scalar product
            a = Tensor(ones(), Symbol[])
            b = Tensor(2.0 * ones(), Symbol[])
            test_frule(contract, a, b; check_inferred=false, testset_name="scalar-scalar product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="scalar-scalar product - rrule")

            # vector-vector inner product
            a = Tensor(ones(2), (:i,))
            b = Tensor(2.0 .* ones(2), (:i,))
            test_frule(contract, a, b; check_inferred=false, testset_name="vector-vector inner product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="vector-vector inner product - rrule")

            # vector-vector outer product
            a = Tensor(ones(2), (:i,))
            b = Tensor(2.0 .* ones(3), (:j,))
            test_frule(contract, a, b; check_inferred=false, testset_name="vector-vector outer product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="vector-vector outer product - rrule")

            # matrix-vector product
            a = Tensor(ones(2, 3), (:i, :j))
            b = Tensor(2.0 .* ones(3), (:j,))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-vector product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-vector product - rrule")

            # matrix-matrix product
            a = Tensor(ones(4, 2), (:i, :j))
            b = Tensor(2.0 .* ones(2, 3), (:j, :k))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-matrix product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-matrix product - rrule")

            # matrix-matrix inner product
            a = Tensor(ones(3, 4), (:i, :j))
            b = Tensor(ones(4, 3), (:j, :i))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-matrix inner product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
        end

        @testset "complex" begin
            # scalar-scalar product
            a = Tensor(fill(1.0 + 1.0im), Symbol[])
            b = Tensor(2.0 * fill(1.0 + 1.0im), Symbol[])
            test_frule(contract, a, b; check_inferred=false, testset_name="scalar-scalar product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="scalar-scalar product - rrule")

            # vector-vector inner product
            a = Tensor(fill(1.0 + 1.0im, 2), (:i,))
            b = Tensor(2.0 .* fill(1.0 + 1.0im, 2), (:i,))
            test_frule(contract, a, b; check_inferred=false, testset_name="vector-vector inner product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="vector-vector inner product - rrule")

            # vector-vector outer product
            a = Tensor(fill(1.0 + 1.0im, 2), (:i,))
            b = Tensor(2.0 .* fill(1.0 + 1.0im, 3), (:j,))
            test_frule(contract, a, b; check_inferred=false, testset_name="vector-vector outer product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="vector-vector outer product - rrule")

            # matrix-vector product
            a = Tensor(fill(1.0 + 1.0im, 2, 3), (:i, :j))
            b = Tensor(2.0 .* fill(1.0 + 1.0im, 3), (:j,))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-vector product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-vector product - rrule")

            # matrix-matrix product
            a = Tensor(fill(1.0 + 1.0im, 4, 2), (:i, :j))
            b = Tensor(2.0 .* fill(1.0 + 1.0im, 2, 3), (:j, :k))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-matrix product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-matrix product - rrule")

            # matrix-matrix inner product
            a = Tensor(fill(1.0 + 1.0im, 3, 4), (:i, :j))
            b = Tensor(fill(1.0 + 1.0im, 4, 3), (:j, :i))
            test_frule(contract, a, b; check_inferred=false, testset_name="matrix-matrix inner product - frule")
            test_rrule(contract, a, b; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
        end
    end
end

@testset "Quantum" begin
    test_frule(Quantum, TensorNetwork([Tensor(ones(2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
    test_rrule(Quantum, TensorNetwork([Tensor(ones(2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
end

@testset "Ansatz" begin
    tn = Quantum(TensorNetwork([Tensor(ones(2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
    lattice = Lattice([lane"1"])
    test_frule(Ansatz, tn, lattice)
    test_rrule(Ansatz, tn, lattice)
end

@testset "Product" begin
    tn = Product([ones(2), ones(2), ones(2)])
    test_frule(Product, Ansatz(tn))
    test_rrule(Product, Ansatz(tn))
end

@testset "MPS" begin
    @testset "Open" begin
        tn = MPS([ones(2, 2), ones(2, 2, 2), ones(2, 2)])
        # test_frule(MPS, Ansatz(tn), form(tn))
        test_rrule(MPS, Ansatz(tn), form(tn))
    end
end

@testset "MPO" begin
    @testset "Open" begin
        tn = MPO([ones(2, 2, 2), ones(2, 2, 2, 2), ones(2, 2, 2)])
        # test_frule(MPO, Ansatz(tn), form(tn))
        test_rrule(MPO, Ansatz(tn), form(tn))
    end
end
