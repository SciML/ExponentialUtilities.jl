# Utility functions

"""
    coeff(::Type,α)

Helper functions that returns the real part if that is what is
required (for Hermitian matrices), otherwise returns the value
as-is.
"""
coeff(::Type{T},α::T) where {T} = α
coeff(::Type{U},α::T) where {U<:Real,T<:Complex} = real(α)

"""
    @diagview(A,d) -> view of the `d`th diagonal of `A`.
"""
macro diagview(A,d::Integer=0)
    s = d<=0 ? 1+abs(d) : :(m+$d)
    quote
        m = size($(esc(A)),1)
        @view($(esc(A))[($s):m+1:end])
    end
end

"""
    realview(::Type, V) -> real view of `V`
"""
realview(::Type{R}, V::AbstractVector{C}) where {R,C<:Complex} =
    @view(reinterpret(R, V)[1:2:end])
realview(::Type{R}, V::AbstractVector{R}) where {R} = V
