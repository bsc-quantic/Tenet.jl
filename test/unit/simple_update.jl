using Test
using Tenet

@testset "apply X" begin
    tn = ProductState([[1, 0], [0, 1]])

    op = Tensor([0 1; 1 0], [Index(plug"1"), Index(plug"1'")])
    simple_update!(tn, op)
    @test parent(tensor_at(tn, site"1")) == [0, 1]

    op = Tensor([0 1; 1 0], [Index(plug"2"), Index(plug"2'")])
    simple_update!(tn, op)
    @test parent(tensor_at(tn, site"2")) == [1, 0]
end

@testset "apply CX" begin
    tn = convert(MPS, ProductState([[0, 1], [0, 1]]))

    op = Tensor(
        reshape(
            [
                1 0 0 0
                0 1 0 0
                0 0 0 1
                0 0 1 0
            ],
            2,
            2,
            2,
            2,
        ),
        [Index(plug"2"), Index(plug"1"), Index(plug"2'"), Index(plug"1'")],
    )
    simple_update!(tn, op)
    a = tensor_at(tn, site"1")
    b = tensor_at(tn, site"2")

    @test view(a, Index(bond"1-2") => 1) |> parent .|> abs ≈ [0, 1]
    @test view(a, Index(bond"1-2") => 2) |> parent .|> abs ≈ zeros(2)

    @test view(b, Index(bond"1-2") => 1) |> parent .|> abs ≈ [1, 0]
    @test view(b, Index(bond"1-2") => 2) |> parent .|> abs ≈ zeros(2)
end
