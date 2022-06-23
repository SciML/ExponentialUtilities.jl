using LinearAlgebra

@inline function exp_gen!(cache, A, ::Val{1})
    T = promote_type(eltype(A), Float64) # Make it work for many 'bigger' types (matrices and scalars)
    # max_memslots=4
    n = size(A, 1)
    # The first slots are precomputed nodes [:A]
    memslots2 = getmem(cache, 2)
    memslots3 = getmem(cache, 3)
    memslots4 = getmem(cache, 4)
    # Assign precomputed nodes memslots 
    memslots1 = A # overwrite A
    # Uniform scaling is exploited.
    # No matrix I explicitly allocated.
    # Computation order: A2 Ua U V Z X P
    # Computing A2 with operation: mult
    mul!(memslots2, memslots1, memslots1)
    # Computing Ua = x*I+x*A2
    coeff1 = 60.0
    coeff2 = 1.0
    memslots3 .= coeff2 .* memslots2
    inplace_add!(memslots3, I * coeff1)
    # Computing U with operation: mult
    mul!(memslots4, memslots3, memslots1)
    # Deallocating Ua in slot 3
    # Deallocating A in slot 1
    # Computing V = x*I+x*A2
    coeff1 = 120.0
    coeff2 = 12.0
    # Smart lincomb recycle A2
    memslots2 .= coeff2 .* memslots2
    inplace_add!(memslots2, I * coeff1)
    # Computing Z = x*V+x*U
    coeff1 = 1.0
    coeff2 = -1.0
    memslots1 .= coeff1 .* memslots2 .+ coeff2 .* memslots4
    # Computing X = x*V+x*U
    coeff1 = 1.0
    coeff2 = 1.0
    # Smart lincomb recycle V
    memslots2 .= coeff1 .* memslots2 .+ coeff2 .* memslots4
    # Deallocating U in slot 4
    # Computing P with operation: ldiv
    ldiv_for_generated!(memslots3, memslots1, memslots2)
    # Deallocating Z in slot 1
    # Deallocating X in slot 2
    copyto!(A, memslots3) # Returning P
end
