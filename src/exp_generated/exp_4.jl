using LinearAlgebra


@inline function exp_gen!(cache,A,::Val{4})
    T=promote_type(eltype(A),Float64) # Make it work for many 'bigger' types (matrices and scalars)
    # max_memslots=6
    n=size(A,1)
    # The first slots are precomputed nodes [:A]
    memslots2 = getmem(cache,2)
    memslots3 = getmem(cache,3)
    memslots4 = getmem(cache,4)
    memslots5 = getmem(cache,5)
    memslots6 = getmem(cache,6)
    # Assign precomputed nodes memslots 
    memslots1=A # overwrite A
    # Uniform scaling is exploited.
    # No matrix I explicitly allocated.
    # Computation order: A2 A4 A6 A8 V Ua U Z X P
    # Computing A2 with operation: mult
    mul!(memslots2,memslots1,memslots1)
    # Computing A4 with operation: mult
    mul!(memslots3,memslots2,memslots2)
    # Computing A6 with operation: mult
    mul!(memslots4,memslots2,memslots3)
    # Computing A8 with operation: mult
    mul!(memslots5,memslots2,memslots4)
    # Computing V = x*I+x*A2+x*A4+x*A6+x*A8
    coeff1=1.76432256e10
    coeff2=2.0756736e9
    coeff3=3.027024e7
    coeff4=110880.0
    coeff5=90.0
    memslots6 .= coeff2.*memslots2 .+ coeff3.*memslots3 .+ coeff4.*memslots4 .+ coeff5.*memslots5
    inplace_add!(memslots6,I*coeff1)
    # Computing Ua = x*I+x*A2+x*A4+x*A6+x*A8
    coeff1=8.8216128e9
    coeff2=3.027024e8
    coeff3=2.16216e6
    coeff4=3960.0
    coeff5=1.0
    # Smart lincomb recycle A2
    memslots2 .= coeff2.*memslots2 .+ coeff3.*memslots3 .+ coeff4.*memslots4 .+ coeff5.*memslots5
    inplace_add!(memslots2,I*coeff1)
    # Deallocating A4 in slot 3
    # Deallocating A6 in slot 4
    # Deallocating A8 in slot 5
    # Computing U with operation: mult
    mul!(memslots3,memslots2,memslots1)
    # Deallocating Ua in slot 2
    # Deallocating A in slot 1
    # Computing Z = x*V+x*U
    coeff1=1.0
    coeff2=-1.0
    memslots1 .= coeff1.*memslots6 .+ coeff2.*memslots3
    # Computing X = x*V+x*U
    coeff1=1.0
    coeff2=1.0
    # Smart lincomb recycle V
    memslots6 .= coeff1.*memslots6 .+ coeff2.*memslots3
    # Deallocating U in slot 3
    # Computing P with operation: ldiv
    ldiv_for_generated!(memslots2, memslots1, memslots6)
    # Deallocating Z in slot 1
    # Deallocating X in slot 6
    copyto!(A,memslots2) # Returning P
end

