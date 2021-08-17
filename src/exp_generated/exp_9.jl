using LinearAlgebra

@inline function exp_9(A)
    T=promote_type(eltype(A),Float64)
    A_copy=similar(A,T); A_copy .= A;
    return exp_9!(cache,A_copy)
end

@inline function exp_9!(cache,A)
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
    # Computation order: C A2 A4 A6 Ub3 Ub Uc U Vb3 Vb V Z X P S1 S2 S3 S4
    # Computing C = x*A+x*I
    coeff1=0.0625
    coeff2=0.0
    # Smart lincomb recycle A
    memslots1 .= coeff1.*memslots1
    mul!(memslots1,true,I*coeff2,true,true)
    # Computing A2 with operation: mult
    mul!(memslots2,memslots1,memslots1)
    # Computing A4 with operation: mult
    mul!(memslots3,memslots2,memslots2)
    # Computing A6 with operation: mult
    mul!(memslots4,memslots2,memslots3)
    # Computing Ub3 = x*A2+x*A4+x*A6
    coeff1=4.08408e7
    coeff2=16380.0
    coeff3=1.0
    memslots5 .= coeff1.*memslots2 .+ coeff2.*memslots3 .+ coeff3.*memslots4
    # Computing Ub with operation: mult
    mul!(memslots6,memslots5,memslots4)
    # Deallocating Ub3 in slot 5
    # Computing Uc = x*I+x*A2+x*A4+x*A6+x*Ub
    coeff1=3.238237626624e16
    coeff2=1.1873537964288e15
    coeff3=1.05594705216e13
    coeff4=3.352212864e10
    coeff5=1.0
    # Smart lincomb recycle Ub
    memslots6 .= coeff2.*memslots2 .+ coeff3.*memslots3 .+ coeff4.*memslots4 .+ coeff5.*memslots6
    mul!(memslots6,true,I*coeff1,true,true)
    # Computing U with operation: mult
    mul!(memslots5,memslots1,memslots6)
    # Deallocating C in slot 1
    # Deallocating Uc in slot 6
    # Computing Vb3 = x*A2+x*A4+x*A6
    coeff1=1.32324192e9
    coeff2=960960.0
    coeff3=182.0
    memslots1 .= coeff1.*memslots2 .+ coeff2.*memslots3 .+ coeff3.*memslots4
    # Computing Vb with operation: mult
    mul!(memslots6,memslots1,memslots4)
    # Deallocating Vb3 in slot 1
    # Computing V = x*I+x*A2+x*A4+x*A6+x*Vb
    coeff1=6.476475253248e16
    coeff2=7.7717703038976e15
    coeff3=1.29060195264e14
    coeff4=6.704425728e11
    coeff5=1.0
    # Smart lincomb recycle A2
    memslots2 .= coeff2.*memslots2 .+ coeff3.*memslots3 .+ coeff4.*memslots4 .+ coeff5.*memslots6
    mul!(memslots2,true,I*coeff1,true,true)
    # Deallocating A4 in slot 3
    # Deallocating A6 in slot 4
    # Deallocating Vb in slot 6
    # Computing Z = x*V+x*U
    coeff1=1.0
    coeff2=-1.0
    memslots1 .= coeff1.*memslots2 .+ coeff2.*memslots5
    # Computing X = x*V+x*U
    coeff1=1.0
    coeff2=1.0
    # Smart lincomb recycle V
    memslots2 .= coeff1.*memslots2 .+ coeff2.*memslots5
    # Deallocating U in slot 5
    # Computing P with operation: ldiv
    LAPACK.gesv!(memslots1, memslots2); memslots3=memslots2
    # Deallocating Z in slot 1
    # Deallocating X in slot 2
    # Computing S1 with operation: mult
    mul!(memslots1,memslots3,memslots3)
    # Deallocating P in slot 3
    # Computing S2 with operation: mult
    mul!(memslots2,memslots1,memslots1)
    # Deallocating S1 in slot 1
    # Computing S3 with operation: mult
    mul!(memslots1,memslots2,memslots2)
    # Deallocating S2 in slot 2
    # Computing S4 with operation: mult
    mul!(memslots2,memslots1,memslots1)
    # Deallocating S3 in slot 1
    return memslots2 # Returning S4
end

