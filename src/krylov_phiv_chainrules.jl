function ChainRulesCore.frule((_, Δt, ΔA, Δb), ::typeof(expv), t, A, b; kwargs...)
    w = expv(t, A, b; kwargs...)
    ∂w = similar(w)
    mul!(∂w, A, w)
    ∂w .*= Δt
    if !isa(Δb, AbstractZero)
        ∂w .+= expv(t, A, Δb; kwargs...)
    end
    # TODO: handle ΔA
    ΔA isa AbstractZero || error("ΔA currently cannot be pushed forward")
    return w, ∂w
end

function ChainRulesCore.rrule(::typeof(expv), t, A, b; kwargs...)
    w = expv(t, A, b; kwargs...)
    function expv_pullback(Δw)
        ∂t = Thunk() do
            t̄ = A isa AbstractMatrix ? conj(dot(Δw, A, w)) : dot(mul!(similar(w), A, w), Δw)
            return t isa Real ? real(t̄) : t̄
        end
        # TODO: handle ∂A
        ∂A = @thunk error("Adjoint wrt A not yet implemented")
        ∂b = Thunk() do
            # using similar is necessary to ensure type-stability
            b̄ = similar(b)
            _copyto!(b̄, expv(t', A', Δw; kwargs...))
            return b̄
        end
        return (NO_FIELDS, ∂t, ∂A, ∂b)
    end
    expv_pullback(::Zero) = (NO_FIELDS, Zero(), Zero(), Zero())
    return w, expv_pullback
end

function _copyto!(x, y)
    if eltype(x) <: Real && !(eltype(y) <: Real)
        x .= real.(y)
    else
        copyto!(x, y)
    end
    return x
end
