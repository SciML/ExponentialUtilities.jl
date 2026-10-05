module ForwardDiffExt

using ForwardDiff: Dual
import ExponentialUtilities

# The Padé coefficients are constants, so they are computed in the value type of a `Dual`
# (recursively, for nested `Dual`s) rather than as `Dual`s with zero partials.
function ExponentialUtilities._coef_type(::Type{<:Dual{Tag, V}}) where {Tag, V}
    return ExponentialUtilities._coef_type(V)
end

end
