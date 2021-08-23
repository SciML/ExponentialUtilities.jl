

"""
    ExpMethodHigham2005(A::AbstractMatrix);
    ExpMethodHigham2005(b::Bool=true);

Computes the matrix exponential using the algorithm Higham, N. J. (2005). "The scaling and squaring method for the matrix exponential revisited." SIAM J. Matrix Anal. Appl.Vol. 26, No. 4, pp. 1179–1193" based on generated code. If a matrix is specified, balancing is determined automatically.
"""
struct ExpMethodHigham2005
    do_balancing::Bool
end
ExpMethodHigham2005(A::AbstractMatrix)=ExpMethodHigham2005(A isa StridedMatrix)
ExpMethodHigham2005()=ExpMethodHigham2005(true)

function alloc_mem(A,::ExpMethodHigham2005)
    return [similar(A) for i=1:5];
end


# Import the generated code
for i=1:13
    include("exp_generated/exp_$i.jl")
end

function getmem(cache,k) # Called from generated code
    return cache[k-1];
end
function ldiv_for_generated!(C,A,B) # C=A\B. Called from generated code
    F= lu!(A);
    ldiv!(F,B); # Result stored in B
    if (pointer_from_objref(C) != pointer_from_objref(B)) # Aliasing allowed
        copyto!(C,B)
    end
    return C
end

# Inplace add of a UniformScaling object (support julia 1.6.2)
@inline function inplace_add!(A,B::UniformScaling) # Called from generated code
    s = B.λ
    @inbounds for i in diagind(A)
        A[i] += s
    end
end
"""
    _exp!(A, caches=nothing, do_balancing = A isa StridedMatrix) -> A

Computes the matrix exponential using the algorithm Higham, N. J. (2005). "The scaling and squaring method for the matrix exponential revisited." SIAM J. Matrix Anal. Appl.Vol. 26, No. 4, pp. 1179–1193" based on generated code. The function does not allocate matrices if the `cache` is provided.
"""
function _exp!(A,method::ExpMethodHigham2005,cache=alloc_mem(A,method))
    n = LinearAlgebra.checksquare(A)
    nA = opnorm(A,1);

    # Maybe to balancing
    if method.do_balancing
        ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    end

    # Select how many multiplications to use
    rhov=    [0.015; 0.25; 0.95; 2.1; 5.4];
    # Number of memslots needed (beside A)
    memslots=[3   ;    4;    5;   5;   5];
    for s=1:7 # Only 8 since exp(5.4*2^8)=Inf
        push!(rhov, rhov[end]*2);
        push!(memslots,memslots[end]);
    end
    i = findfirst(nA .< rhov)



    # Make the call to the appropriate exp_gen! function
    X = Base.Cartesian.@nif 13 d -> begin
        nA < rhov[d]
    end d -> begin # if condition
        exp_gen!(cache, A, Val(d))
    end d -> begin # fallback (d == 13)
        exp_gen!(cache, A, Val(d))
    end

    # Undo the balancing
    if method.do_balancing
        for j = ilo:ihi
            scj = scale[j]
            for i = 1:n
                X[j,i] *= scj
            end
            for i = 1:n
                X[i,j] /= scj
            end
        end

        if ilo > 1       # apply lower permutations in reverse order
            for j in (ilo-1):-1:1; LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
        end
        if ihi < n       # apply upper permutations in forward order
            for j in (ihi+1):n;    LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
        end
    end

    return X

end
