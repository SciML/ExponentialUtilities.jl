# Import the generated code
for i=1:15
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
    for s=1:8
        push!(rhov, rhov[end]*2);
        push!(memslots,memslots[end]);
    end
    i = findfirst(nA .< rhov)

    # Take care of caching
    if caches == nothing
        # No cache. Allocate Matrix objects
        TT=eltype(A);
        thiscache=[Matrix{TT}(undef,n,n) for i=1:memslots[i]];
    else
        thiscache = caches;
    end


    # Evaluate
    exp_fun=Symbol("exp_$(i)!")
    s=:($(exp_fun)($thiscache,$A));
    X=eval(s)::typeof(A) # Evaluate and maintain type stability

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
