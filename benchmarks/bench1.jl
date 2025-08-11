using BenchmarkTools

using Tenet

function overlap_tenet(L)

    psi = rand(MPS, n=L, maxdim=30)
    phi = rand(MPS, n=L, maxdim=50)
    o = rand(MPO, n=L, maxdim=1)

    #psic = copy(psi)
    evolve!(psi, o)

    return overlap(psi,phi)
end

bench_1 = "<phi|O|psi>"
bench_sub1 = "L=10"
bench_sub2 = "L=20"

suite = BenchmarkGroup()

suite[bench_1] = BenchmarkGroup(["overlaps"])

suite[bench_1][bench_sub1]["x"]["Tenet"] = @benchmarkable overlap_tenet(10)
suite[bench_1][bench_sub2]["x"]["Tenet"] = @benchmarkable overlap_tenet(20)

tune!(suite)
results1 = run(suite, verbose = true)

r1 = median(results1)

BenchmarkTools.save("output.json", r1)
