using LinearAlgebra

function exp_gen!(cache, A, ::Val{2})
    T = promote_type(eltype(A), Float64) # Make it work for many 'bigger' types (matrices and scalars)
    # max_memslots=5
    n = size(A, 1)
    # The first slots are precomputed nodes [:A]
    memslots2 = getmem(cache, 2)
    memslots3 = getmem(cache, 3)
    memslots4 = getmem(cache, 4)
    memslots5 = getmem(cache, 5)
    # Assign precomputed nodes memslots 
    memslots1 = A # overwrite A
    # Uniform scaling is exploited.
    # No matrix I explicitly allocated.
    # Computation order: A2 A4 Ua U V Z X P
    # Computing A2 with operation: mult
    mul!(memslots2, memslots1, memslots1)
    # Computing A4 with operation: mult
    mul!(memslots3, memslots2, memslots2)
    # Computing Ua = x*I+x*A2+x*A4
    coeff1 = 15120.0
    coeff2 = 420.0
    coeff3 = 1.0
    memslots4 .= coeff2 .* memslots2 .+ coeff3 .* memslots3
    inplace_add!(memslots4, I * coeff1)
    # Computing U with operation: mult
    mul!(memslots5, memslots4, memslots1)
    # Deallocating Ua in slot 4
    # Deallocating A in slot 1
    # Computing V = x*I+x*A2+x*A4
    coeff1 = 30240.0
    coeff2 = 3360.0
    coeff3 = 30.0
    # Smart lincomb recycle A2
    memslots2 .= coeff2 .* memslots2 .+ coeff3 .* memslots3
    inplace_add!(memslots2, I * coeff1)
    # Deallocating A4 in slot 3
    # Computing Z = x*V+x*U
    coeff1 = 1.0
    coeff2 = -1.0
    memslots1 .= coeff1 .* memslots2 .+ coeff2 .* memslots5
    # Computing X = x*V+x*U
    coeff1 = 1.0
    coeff2 = 1.0
    # Smart lincomb recycle V
    memslots2 .= coeff1 .* memslots2 .+ coeff2 .* memslots5
    # Deallocating U in slot 5
    # Computing P with operation: ldiv
    ldiv_for_generated!(memslots3, memslots1, memslots2)
    # Deallocating Z in slot 1
    # Deallocating X in slot 2
    copyto!(A, memslots3) # Returning P
end
