using Tenet: letter

@testset "letter" begin
    # NOTE probabilitic testing due to time taken by `letter`. refactor when `letter` is optimized.
    @test all(isletter ∘ only ∘ String, Iterators.map(letter, rand(1:(Tenet.NUM_UNICODE_LETTERS), 1000)))
end
