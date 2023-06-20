@setup_workload begin
    function precomp_gen_mat(; Tx, n)
        mx = rand(Tx, n, n)
        m = rand(Tx, n, n)
        v = rand(Tx, n)
        (mx, m, v)
    end

    function precomp_fx(; method, Tx, n = 5)
        (mx, m, v) = precomp_gen_mat(; Tx, n)

        # mat exp
        cache = ExponentialUtilities.alloc_mem(mx, method)
        exm = exponential!(mx, method, cache)

        # exp v
        [expv(ts, f(m), v)
         for ts in (1.0, 1.0im)
         for f in (copy, Hermitian)]

        # Subspace exponential caches not yet available for non-Hermitian matrices
        (Tx <: Real) && expv(1.0, Hermitian(m), v; mode = :error_estimate)

        # phi v
        # Subspace exponential caches not yet available for non-Hermitian matrices
        if Tx <: Complex
            [phiv(ts, Hermitian(m), v, 1)
             for ts in (1.0, 1.0im)]
        else
            phiv(1.0, m, v, 1)
        end
    end

    precomp_ms = [
        #ExpMethodHigham2005(),
        ExpMethodHigham2005Base(),
        #ExpMethodGeneric(),
        #ExpMethodNative(),
        #ExpMethodDiagonalization(),
    ]

    Txs = [Float64]
    #Txs = [Float64, ComplexF64]

    @compile_workload begin
        [precomp_fx(; method, Tx) for method in precomp_ms
         for Tx in Txs]
    end
end
