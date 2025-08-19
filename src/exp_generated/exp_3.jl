using LinearAlgebra

# See exp_gen!(cache, A, ::Val{1}) for documentation
# This implements the Padé approximation of order 3
function exp_gen!(cache, A, ::Val{3})
    T = promote_type(eltype(A), Float64) # Make it work for many 'bigger' types (matrices and scalars)
    # max_memslots=6
    n = size(A, 1)
    # The first slots are precomputed nodes [:A]
    memslots2 = getmem(cache, 2)
    memslots3 = getmem(cache, 3)
    memslots4 = getmem(cache, 4)
    memslots5 = getmem(cache, 5)
    memslots6 = getmem(cache, 6)
    # Assign precomputed nodes memslots 
    memslots1 = A # overwrite A
    # Uniform scaling is exploited.
    # No matrix I explicitly allocated.
    # Computation order: A2 A4 A6 Ua U V Z X P
    # Computing A2 with operation: mult
    mul!(memslots2, memslots1, memslots1)
    # Computing A4 with operation: mult
    mul!(memslots3, memslots2, memslots2)
    # Computing A6 with operation: mult
    mul!(memslots4, memslots2, memslots3)
    # Computing Ua = x*I+x*A2+x*A4+x*A6
    coeff1 = 8.64864e6
    coeff2 = 277200.0
    coeff3 = 1512.0
    coeff4 = 1.0
    memslots5 .= coeff2 .* memslots2 .+ coeff3 .* memslots3 .+ coeff4 .* memslots4
    inplace_add!(memslots5, I * coeff1)
    # Computing U with operation: mult
    mul!(memslots6, memslots5, memslots1)
    # Deallocating Ua in slot 5
    # Deallocating A in slot 1
    # Computing V = x*I+x*A2+x*A4+x*A6
    coeff1 = 1.729728e7
    coeff2 = 1.99584e6
    coeff3 = 25200.0
    coeff4 = 56.0
    # Smart lincomb recycle A2
    memslots2 .= coeff2 .* memslots2 .+ coeff3 .* memslots3 .+ coeff4 .* memslots4
    inplace_add!(memslots2, I * coeff1)
    # Deallocating A4 in slot 3
    # Deallocating A6 in slot 4
    # Computing Z = x*V+x*U
    coeff1 = 1.0
    coeff2 = -1.0
    memslots1 .= coeff1 .* memslots2 .+ coeff2 .* memslots6
    # Computing X = x*V+x*U
    coeff1 = 1.0
    coeff2 = 1.0
    # Smart lincomb recycle V
    memslots2 .= coeff1 .* memslots2 .+ coeff2 .* memslots6
    # Deallocating U in slot 6
    # Computing P with operation: ldiv
    ldiv_for_generated!(memslots3, memslots1, memslots2)
    # Deallocating Z in slot 1
    # Deallocating X in slot 2
    copyto!(A, memslots3) # Returning P
end
