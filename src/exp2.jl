# Import the generated code
for i=1:13
    include("exp_generated/exp_$i.jl")
end

function getmem(cache,k)
    return cache[k-1];
end

function _exp2!(A; caches=nothing, do_balancing = A isa StridedMatrix)
    n = LinearAlgebra.checksquare(A)
    nA = opnorm(A,1);

    # Maybe to balancing
    if do_balancing
        ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    end

    # Select how many multiplications to use
    rhov=    [0.015; 0.25; 0.95; 2.1; 5.4];
    # Number of memslots needed (beside A)
    memslots=[3   ;    4;    5;   6;   5];
    for s=1:7 # Only 8 since exp(5.4*2^8)=Inf
        push!(rhov, rhov[end]*2);
        push!(memslots,memslots[end]);
    end
    i = findfirst(nA < rho for rho in rhov)

    # Take care of caching
    if caches == nothing
        # No cache. Allocate Matrix objects
        TT=eltype(A);
        thiscache=[Matrix{TT}(undef,n,n) for i=1:memslots[i]];
    else
        thiscache = caches;
    end


    # Evaluate
    if (nA < rhov[1])
        X=exp_1!(thiscache,A)
    elseif (nA < rhov[2])
        X=exp_2!(thiscache,A)
    elseif (nA < rhov[3])
        X=exp_3!(thiscache,A)
    elseif (nA < rhov[4])
        X=exp_4!(thiscache,A)
    elseif (nA < rhov[5])
        X=exp_5!(thiscache,A)
    elseif (nA < rhov[6])
        X=exp_6!(thiscache,A)
    elseif (nA < rhov[7])
        X=exp_7!(thiscache,A)
    elseif (nA < rhov[8])
        X=exp_8!(thiscache,A)
    elseif (nA < rhov[9])
        X=exp_9!(thiscache,A)
    elseif (nA < rhov[10])
        X=exp_10!(thiscache,A)
    elseif (nA < rhov[11])
        X=exp_11!(thiscache,A)
    elseif (nA < rhov[12])
        X=exp_12!(thiscache,A)
    else # X will become very close to Inf
        X=exp_13!(thiscache,A)
    end

    # Undo the balancing
    if do_balancing
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
