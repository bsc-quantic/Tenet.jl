using SparseArrays

struct IncidenceMatrix{T} <: AbstractSparseArray{Bool,T,2}
    rows::Dict{T,Vector{T}}
    cols::Dict{T,Vector{T}}
end

IncidenceMatrix(args...; kwargs...) = IncidenceMatrix{Int}(args...; kwargs...)
IncidenceMatrix{T}() where {T} = IncidenceMatrix{T}(Dict{T,Vector{T}}(), Dict{T,Vector{T}}())

# NOTE `i ∈ arr.cols[j]` must be equivalent
Base.getindex(arr::IncidenceMatrix, i, j) = j ∈ arr.rows[i]
Base.getindex(arr::IncidenceMatrix, i, ::Colon) = arr.rows[i]
Base.getindex(arr::IncidenceMatrix, ::Colon, j) = arr.cols[j]

function Base.setindex!(arr::IncidenceMatrix{T}, v, i, j) where {T}
    row = get!(arr.rows, i, T[])
    col = get!(arr.cols, j, T[])

    if v
        j ∉ row && push!(row, j)
        i ∉ col && push!(col, i)
    else
        filter!(==(j), row)
        filter!(==(i), col)
    end

    return arr
end

insertrow!(arr::IncidenceMatrix{T}, i) where {T} = get!(arr.rows, i, T[])
insertcol!(arr::IncidenceMatrix{T}, j) where {T} = get!(arr.cols, j, T[])

function deleterow!(arr::IncidenceMatrix, i)
    for j in arr.rows[i]
        filter!(==(i), arr.cols[j])
    end
    delete!(a.rows, i)
    return arr
end

function deletecol!(arr::IncidenceMatrix, j)
    for i in arr.cols[j]
        filter!(==(j), arr.rows[i])
    end
    delete!(a.cols, j)
    return arr
end

Base.size(arr::IncidenceMatrix) = (length(arr.rows), length(arr.cols))

SparseArrays.nnz(arr::IncidenceMatrix) = mapreduce(length, +, arr.rows)
function SparseArrays.findnz(arr::IncidenceMatrix)
    I = Iterators.flatmap(enumerate(values(arr.rows)) do (i, row)
        Iterators.repeated(i, length(row))
    end) |> collect
    J = collect(Iterators.flatten(values(arr.rows)))
    V = trues(length(I))
    return (I, J, V)
end
