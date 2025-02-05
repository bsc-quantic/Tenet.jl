function lanes end

function tensorat end
function laneat end

nlanes(tn) = length(lanes(tn))
haslane(tn, lane) = lane ∈ lanes(tn)

# sugar
Base.in(lane::Lane, tn::AbstractTensorNetwork) = haslane(tn, lane)
