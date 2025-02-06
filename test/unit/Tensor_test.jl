
@testset "Constructors" begin
    @testset "Number" begin
        tensor = Tensor(1.0)
        @test isempty(inds(tensor))
        @test parent(tensor) == fill(1.0)
    end

    @testset "Array" begin
        data = ones(2, 2, 2)
        tensor = Tensor(data, [:i, :j, :k])

        @test inds(tensor) == (:i, :j, :k)
        @test parent(tensor) === data

        @test_throws DimensionMismatch Tensor(zeros(2, 3), (:i, :i))
    end
end

@testset "copy" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test parent(tensor) == parent(copy(tensor))
    @test inds(tensor) == inds(copy(tensor))

    @test copy(view(tensor, :i => 1)) isa Tensor
    @test parent(copy(view(tensor, :i => 1))) isa Array
end

@testset "eltype" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(rand(T, 2), [:i])
    @test eltype(tensor) == T
end

@testset "elsize" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(rand(T, 2), [:i])
    @test Base.elsize(tensor) == sizeof(T)
end

@testset "similar" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))

    @test eltype(similar(tensor)) == eltype(tensor)
    @test size(similar(tensor)) == size(tensor)
    @test parent(similar(tensor)) !== parent(tensor)
    @test inds(similar(tensor)) == inds(tensor)

    @test inds(similar(tensor; inds=[:a, :b, :c])) == (:a, :b, :c)

    @test eltype(similar(tensor, Bool)) == Bool
    @test size(similar(tensor, Bool)) == size(tensor)
    @test inds(similar(tensor, Bool)) == inds(tensor)

    @test eltype(similar(tensor, 2, 2, 4)) == eltype(tensor)
    @test size(similar(tensor, 2, 2, 4)) == (2, 2, 4)
    @test inds(similar(tensor, 2, 2, 4)) == inds(tensor)

    @test eltype(similar(tensor, Bool, 2, 2, 4)) == Bool
    @test size(similar(tensor, Bool, 2, 2, 4)) == (2, 2, 4)
    @test inds(similar(tensor, Bool, 2, 2, 4)) == inds(tensor)

    @test_throws DimensionMismatch similar(tensor, 2, 2)
end

@testset "zero" begin
    tensor = Tensor(ones(2, 2, 2), (:i, :j, :k))
    @test parent(zero(tensor)) == zeros(size(tensor)...)
    @test inds(zero(tensor)) == inds(tensor)
end

@testset "isequal" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test tensor == copy(tensor)
    @test tensor != zeros(size(tensor)...)
    @test zeros(size(tensor)...) != tensor

    @test tensor ∈ [tensor]
    @test copy(tensor) ∈ [tensor]
    @test tensor ∈ [copy(tensor)]
    @test zeros(size(tensor)...) ∉ [tensor]

    @test tensor ∈ Set([tensor])
    @test zeros(size(tensor)...) ∉ Set([tensor])

    @test tensor == permutedims(tensor, (3, 1, 2))
end

@testset "isapprox" begin
    data = rand(2, 3, 4, 5)
    tensor = Tensor(data, (:i, :j, :k, :l))

    @test tensor ≈ copy(tensor)
    @test tensor ≈ permutedims(tensor, (3, 1, 2, 4))
    @test tensor ≈ permutedims(tensor, (2, 4, 1, 3))
    @test tensor ≈ permutedims(tensor, (4, 3, 2, 1))
    @test tensor ≈ tensor .+ 1e-14

    @test !(tensor ≈ Tensor(data, (:i, :m, :n, :l)))
    @test !(tensor ≈ Tensor(rand(2, 2, 2), (:i, :j, :k)))
    @test !(tensor ≈ data)
end

@testset "strides" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test strides(tensor) == (1, 2, 4)
    @test stride(tensor, 1) == stride(tensor, :i) == 1
    @test stride(tensor, 2) == stride(tensor, :j) == 2
    @test stride(tensor, 3) == stride(tensor, :k) == 4
end

@testset "unsafe_convert" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test Base.unsafe_convert(Ptr{Float64}, tensor) == Base.unsafe_convert(Ptr{Float64}, parent(tensor))
end

@testset "Indexing" begin
    data = [1 2; 3 4]
    tensor = Tensor(copy(data), (:i, :j))

    @testset "getindex" begin
        @test tensor[1, 1] == 1
        @test tensor[1, 2] == 2
        @test tensor[2, 1] == 3
        @test tensor[2, 2] == 4

        @test tensor[1, :] == [1, 2]
        @test tensor[2, :] == [3, 4]
        @test tensor[:, 1] == [1, 3]
        @test tensor[:, 2] == [2, 4]

        @test tensor[:, :] isa Matrix{Int} && tensor[:, :] == data
        @test tensor[:] isa Vector{Int} && tensor[:] == data[:]
    end

    @testset "setindex!" begin
        tensor[1, 1] = 0
        @test tensor[1, 1] == 0

        tensor[1, :] = [5, 5]
        @test tensor[1, :] == [5, 5]

        tensor[:, 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]

        tensor[:, :] = data * 5
        @test tensor[:, :] == data * 5

        tensor[:] = data[:] * 10
        @test tensor[:] == data[:] * 10
    end
end

@testset "Base.replace" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test inds(replace(tensor, :i => :u, :j => :v, :k => :w)) == (:u, :v, :w)
    @test parent(replace(tensor, :i => :u, :j => :v, :k => :w)) === parent(tensor)

    @test inds(replace(tensor, :a => :u, :b => :v, :c => :w)) == (:i, :j, :k)
    @test parent(replace(tensor, :a => :u, :b => :v, :c => :w)) === parent(tensor)
end

@testset "dim" begin
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    @test dim(tensor, 1) == 1
    for (i, label) in enumerate(inds(tensor))
        @test dim(tensor, label) == i
    end

    @test_throws BoundsError dim(tensor, :_)
end

@testset "Broadcasting" begin
    data = rand(2, 2, 2)
    @test begin
        tensor = Tensor(data, (:a, :b, :c))
        tensor = tensor .+ one(eltype(tensor))

        parent(tensor) == data .+ one(eltype(tensor))
    end

    @test begin
        tensor = Tensor(data, (:a, :b, :c))
        tensor = sin.(tensor)

        parent(tensor) == sin.(data)
    end
end

@testset "selectdim" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))

    @test parent(selectdim(tensor, :i, 1)) == selectdim(data, 1, 1)
    @test parent(selectdim(tensor, :j, 2)) == selectdim(data, 2, 2)
    @test issetequal(inds(selectdim(tensor, :i, 1)), (:j, :k))
    @test issetequal(inds(selectdim(tensor, :i, 1:1)), (:i, :j, :k))
end

@testset "view" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))

    @test parent(view(tensor, 2, :, :)) == view(data, 2, :, :)
    @test parent(view(tensor, :i => 1)) == view(data, 1, :, :)
    @test parent(view(tensor, :j => 2)) == view(data, :, 2, :)
    @test parent(view(tensor, :i => 2, :k => 1)) == view(data, 2, :, 1)
    @test :i ∉ inds(view(tensor, :i => 1))

    @test parent(view(tensor, :i => 1:1)) == view(data, 1:1, :, :)
    @test :i ∈ inds(view(tensor, :i => 1:1))
end

@testset "permutedims" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))
    perm = (3, 1, 2)

    @test inds(permutedims(tensor, perm)) == (:k, :i, :j)
    @test parent(permutedims(tensor, perm)) == permutedims(data, perm)

    newtensor = Tensor(similar(data), (:a, :b, :c))
    permutedims!(newtensor, tensor, perm)
    @test parent(newtensor) == parent(permutedims(tensor, perm))
end

@testset "indexing" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))

    @test firstindex(tensor) == 1
    @test lastindex(tensor) == 8
    @test all(firstindex(tensor, i) == 1 for i in 1:ndims(tensor))
    @test all(lastindex(tensor, i) == 2 for i in 1:ndims(tensor))

    @test axes(tensor) == axes(data)
    @test first(tensor) == first(data)
    @test last(tensor) == last(data)
    @test tensor[1, :, 2] == data[1, :, 2]
    @test tensor[i=1, k=2] == data[1, :, 2]

    tensor[1] = 0
    @test tensor[1] == data[1]

    for i in [0, -1, length(tensor) + 1]
        @test_throws BoundsError tensor[i]
    end
end

@testset "iteration" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))

    @test Base.IteratorSize(tensor) == Base.HasShape{3}()
    @test Base.IteratorEltype(tensor) == Base.HasEltype()
    @test all(x -> ==(x...), zip(tensor, data))
end

@testset "conj/adjoint" begin
    @testset "scalar" begin
        tensor = Tensor(fill(1.0 + 1.0im), Symbol[])

        @test inds(conj(tensor)) == ()
        @test isapprox(conj(tensor), 1.0 - 1.0im)
        @test adjoint(tensor) == conj(tensor)
    end

    @testset "Vector" begin
        data = fill(1.0 + 1.0im, 2)
        tensor = Tensor(data, (:i,))

        @test inds(conj(tensor)) == (:i,)
        @test all(isapprox.(conj(tensor), fill(1.0 - 1.0im, size(data)...)))
        @test adjoint(tensor) == conj(tensor)
    end

    @testset "Matrix" begin
        data = fill(1.0 + 1.0im, 2, 2)
        tensor = Tensor(data, (:i, :j))

        @test inds(adjoint(tensor)) == (:i, :j)
        @test all(isapprox.(conj(tensor), fill(1.0 - 1.0im, size(data)...)))
        @test adjoint(tensor) == conj(tensor)
    end
end

@testset "transpose" begin
    @testset "Vector" begin
        data = rand(Complex{Float64}, 2)
        tensor = Tensor(data, (:i,))

        @test inds(transpose(tensor)) == (:i,)
        @test ndims(transpose(tensor)) == 1
        @test all(isapprox.(transpose(tensor), data))
    end

    @testset "Matrix" begin
        data = rand(Complex{Float64}, 2, 2)
        tensor = Tensor(data, (:i, :j))

        @test inds(transpose(tensor)) == (:j, :i)
        @test ndims(transpose(tensor)) == 2
        @test all(isapprox.(transpose(tensor), transpose(data)))
    end
end

@testset "expand" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, (:i, :j, :k))

    let new = expand(tensor; label=:x, axis=1)
        @test inds(new) == (:x, :i, :j, :k)
        @test size(new, :x) == 1
        @test selectdim(new, :x, 1) == tensor
    end

    let new = expand(tensor; label=:x, axis=3)
        @test inds(new) == (:i, :j, :x, :k)
        @test size(new, :x) == 1
        @test selectdim(new, :x, 1) == tensor
    end

    let new = expand(tensor; label=:x, axis=1, size=2, method=:zeros)
        @test inds(new) == (:x, :i, :j, :k)
        @test size(new, :x) == 2
        @test selectdim(new, :x, 1) == tensor
        @test selectdim(new, :x, 2) == Tensor(zeros(size(data)...), inds(tensor))
    end

    let new = expand(tensor; label=:x, axis=1, size=2, method=:repeat)
        @test inds(new) == (:x, :i, :j, :k)
        @test size(new, :x) == 2
        @test selectdim(new, :x, 1) == tensor
        @test selectdim(new, :x, 2) == tensor
    end
end

@testset "groupinds" begin
    tensor = Tensor(rand(2, 3), (:i, :j))
    grouped = Tenet.groupinds(tensor, [:i, :j])
    @test vec(tensor) ≈ parent(grouped)

    grouped = Tenet.groupinds(tensor, [:j, :i])
    @test vec(transpose(parent(tensor))) ≈ parent(grouped)

    tensor = Tensor(rand(2, 3, 4), (:i, :k, :j))
    grouped = Tenet.groupinds(tensor, [:i, :j])
    @test reshape(permutedims(parent(tensor), [1, 3, 2]), 8, 3) ≈ parent(grouped)

    tensor = Tensor(rand(2, 3, 4), (:i, :k, :j))
    grouped = Tenet.groupinds(tensor, [:j, :i])
    @test reshape(permutedims(parent(tensor), [2, 3, 1]), 3, 8) ≈ parent(grouped)
end
