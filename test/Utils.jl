using Test

# from https://discourse.julialang.org/t/skipping-a-whole-testset/65006/4
"""
Skip a testset

Use `@testset_skip` to replace `@testset` for some tests which should be skipped.

## Usage

Replace `@testset` with `@testset "reason"` where `"reason"` is a string saying why the
test should be skipped (which should come before the description string, if that is
present).
"""
macro testset_skip(args...)
    isempty(args) && error("No arguments to @testset_skip")
    length(args) < 2 && error("First argument to @testset_skip giving reason for " * "skipping is required")

    skip_reason = args[1]

    desc, testsettype, options = Test.parse_testset_args(args[2:(end - 1)])

    ex = quote
        # record the reason for the skip in the description, and mark the tests as
        # broken, but don't run tests
        local ts = Test.DefaultTestSet(string($desc, " - ", $skip_reason))
        push!(ts.results, Test.Broken(:skipped, "skipped tests"))
        local ret = Test.finish(ts)
        ret
    end

    return ex
end

"""
    @testimpl expr

Test `expr` but return `false` if a `MethodError` is thrown.
"""
macro testimpl(expr)
    return Base.remove_linenums!(:(
        try
            @test $(esc(expr))
        catch e
            if e isa MethodError
                return false
            else
                rethrow(e)
            end
        end
    ))
end

"""
    @testif pred=predexpr testexpr

Test `expr` if `predexpr` is true.
"""
macro testif(pred, expr)
    @assert Meta.isexpr(pred, :(=), 2)
    @assert pred.args[1] == :pred
    predexpr = pred.args[2]
    return quote
        if $predexpr
            @test $expr
        else
            # TODO it should skip here but let's just return true for now
            @test $expr skip = true
        end
    end |> esc
end
