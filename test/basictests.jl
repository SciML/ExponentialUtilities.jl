using Test, LinearAlgebra, Random, SparseArrays, ExponentialUtilities
using ExponentialUtilities: getH, getV, exponential!, ExpMethodNative,
    ExpMethodDiagonalization, ExpMethodHigham2005, ExpMethodGeneric,
    ExpMethodHigham2005Base, alloc_mem
using ForwardDiff, StaticArrays, DoubleFloats

@testset "exp!" begin
    n = 100
    A1 = randn(n, n)
    expA1 = exp(A1)
    A2 = 0.5 * A1 / opnorm(A1, 1) # Test small norm as well
    expA2 = exp(A2)

    methodlist = [
        ExpMethodNative(),
        ExpMethodDiagonalization(),
        ExpMethodHigham2005(),
        ExpMethodHigham2005Base(),
        ExpMethodGeneric(),
    ]

    for m in methodlist
        @testset "$(typeof(m))" begin
            E1 = exponential!(copy(A1), m)
            @test E1 ≈ expA1
            E2 = exponential!(copy(A2), m)
            @test E2 ≈ expA2

            # With preallocation
            mem = alloc_mem(A1, m)
            E1 = exponential!(copy(A1), m)
            @test E1 ≈ expA1
        end
    end
end

@testset "Exp generated" begin
    # Test all the cases for coverage of all generated code in exp
    method = ExpMethodHigham2005()
    A0 = [3.0 2.0; 0.0 1.0]
    A0 = A0 / opnorm(A0, 1)
    rhov = [0; 0.015; 0.25; 0.95; 2.1; 5.4]
    for s in 1:7
        push!(rhov, rhov[end] * 2)
    end
    for (i, _) in enumerate(rhov)
        if (i + 1 < size(rhov, 1))
            r = (rhov[i] + rhov[i + 1]) / 2
        else
            r = rhov[i] + 0.5
        end

        A = A0 * r
        expA = exp(A)
        A = exponential!(A, method)
        @test A ≈ expA
    end
end

#
#@testset "Exp" begin
#    n = 100
#    A = randn(n, n)
#    expA = exp(A)
#    exponential!(A)
#    @test A ≈ expA
#    A2 = randn(n, n)
#    A2 ./= opnorm(A2, 1) # test for small opnorm
#    expA2 = exp(A2)
#    exponential!(A2)
#    @test A2 ≈ expA2
#
#
#
exp_generic(A) = exponential!(copy(A), ExpMethodGeneric())
@testset "exp_generic" begin
    for n in [5, 10, 30, 50, 100, 500]
        M = rand(n, n)
        @test exp(M) ≈ exp_generic(M)

        M′ = M / 10opnorm(M, 1)
        @test exp(M′) ≈ exp_generic(M′)

        N = randn(n, n)
        @test exp(N) ≈ exp_generic(N)

        exp(n) ≈ exp_generic(n)
    end

    @testset "Inf" begin
        @test exp_generic(Inf) == Inf
        @test exp_generic(NaN) === NaN
        @test exp_generic(1.0e20) === Inf
        @test all(isinf, exp_generic([1 Inf; Inf 1]))
        @test all(isnan, exp_generic([1 Inf; Inf 0]))
        @test all(isnan, exp_generic([1 Inf 1 0; 1 1 1 1; 1 1 1 1; 1 1 1 1]))
    end
end

@testset "exponential! on immutable (static) matrices" begin
    for (n, T) in ((3, Float64), (4, Float64), (3, Float32), (4, Float32))
        A = rand(SMatrix{n, n, T})
        E = exponential!(A)
        @test E isa SMatrix{n, n}
        @test E ≈ exp(Matrix(A))
        @test !any(isnan, E)

        J = ForwardDiff.jacobian(exponential!, A)
        Jref = ForwardDiff.jacobian(exp_generic, Matrix(A))
        @test !any(isnan, J)
        @test J ≈ Jref
    end
end

@testset "ExpMethodGeneric preserves element type (immutable matrices)" begin
    # https://discourse.julialang.org/t/137880 : ExpMethodGeneric silently promoted
    # Float32 static matrices to Float64. The (13,13) Padé path must keep the input type.
    for (n, T) in ((2, Float32), (3, Float32), (2, Float64), (3, Float64))
        A = rand(SMatrix{n, n, T})
        E = exponential!(A, ExpMethodGeneric())
        @test E isa SMatrix{n, n, T}
        @test Float64.(E) ≈ exp(Float64.(Matrix(A))) rtol = sqrt(eps(T))

        J = ForwardDiff.jacobian(x -> exponential!(x, ExpMethodGeneric()), A)
        @test eltype(J) === T
    end

    # ComplexF32 must stay ComplexF32
    Ac = rand(SMatrix{2, 2, ComplexF32})
    @test exponential!(Ac, ExpMethodGeneric()) isa SMatrix{2, 2, ComplexF32}

    # Scalar path likewise preserves precision
    @test exponential!(0.5f0, ExpMethodGeneric()) isa Float32
    @test ForwardDiff.derivative(x -> exponential!(x, ExpMethodGeneric()), 0.3f0) isa Float32

    # The (13,13) Padé coefficients are exact rationals, so an immutable Double64 matrix
    # must reach full Double64 accuracy. Hardcoded Float64 coefficients capped this at
    # eps(Float64) (~1e-16); with exact rationals it reaches eps(Double64) (~5e-32).
    Ad = SMatrix{2, 2, Double64}(
        Double64(9 // 10), Double64(2 // 10),
        Double64(3 // 10), Double64(7 // 10)
    )
    Rd = exponential!(Ad, ExpMethodGeneric())
    setprecision(BigFloat, 300) do
        Ab = BigFloat[9 // 10 3 // 10; 2 // 10 7 // 10]
        sc = Ab ./ BigFloat(2)^50
        ref = sum(sc^n / factorial(big(n)) for n in 0:80)
        for _ in 1:50
            ref = ref * ref
        end
        @test maximum(abs.(BigFloat.(Rd) .- ref)) < 1.0e-28
    end
end

@testset "exponential! sparse" begin
    A = sparse([1, 2, 1], [2, 1, 1], [1.0, 2.0, 3.0])
    @test_throws ErrorException exponential!(A)
end

@testset "Issue 41" begin
    @test ForwardDiff.derivative(exp_generic, 0.1) ≈ exp_generic(0.1) atol = 1.0e-15
end

@testset "Issue 42" begin
    @test exp_generic(0.0) == 1
    @test ForwardDiff.derivative(exp_generic, 0.0) == 1
    @test ForwardDiff.derivative(t -> ForwardDiff.derivative(exp_generic, t), 0.0) == 1
end

@testset "Issue 143" begin
    ts = collect(0:0.1:1)

    out = Pipe()
    res = redirect_stdout(out) do
        expv_timestep(ts, reshape([1], (1, 1)), [1.0]; verbose = true)
    end
    close(Base.pipe_writer(out))

    @test occursin("Completed after 1 time step(s)", read(out, String))

    @test vec(res) ≈ exp.(ts)
end

@testset "Issue 44 - BigFloat precision" begin
    using ExponentialUtilities: pade_order_for_type

    # Test that pade_order_for_type returns reasonable values
    @test pade_order_for_type(Float64) == 13
    @test pade_order_for_type(BigFloat) >= 24  # depends on precision

    # Test scalar BigFloat accuracy
    _x = range(-10, stop = 10, length = 50)
    bigfloat_x = big.(_x)
    max_err = maximum(
        abs,
        (x -> exp_generic(x) / exp(x) - 1).(bigfloat_x)
    ) / eps(BigFloat)
    @test max_err < 100  # Should be within ~100 eps

    # Test with different precisions
    for bits in [128, 256, 512]
        setprecision(bits) do
            x = big"0.5"
            result = exp_generic(x)
            exact = exp(x)
            rel_err = abs(result - exact) / exact
            @test rel_err < 100 * eps(BigFloat)
        end
    end
end

@testset "naive_matmul" begin
    A = Matrix(reshape((1.0:(23.0^2)) ./ 700, (23, 23)))
    @test exp_generic(A) ≈ exp(A)
    # FiniteDiff.finite_difference_gradient(sum ∘ exp, A)
    @test ForwardDiff.gradient(sum ∘ exp_generic, A) ≈
        [
        1057.6079767302406 1061.5362688188497 1065.4645657136762 1069.39286741472 1073.3211571002205 1077.249468413699 1081.1777677116343 1085.1060742188956 1089.0343494857439 1092.9626752178747 1096.890964903375 1100.8192569919843 1104.747561096137 1108.6758820220502 1112.6041596920072 1116.5324493775076 1120.460763094095 1124.3890623920304 1128.3173592868568 1132.245658584792 1136.173974704488 1140.1022643899885 1144.030546866163;
        1546.2686015296754 1551.8428067925008 1557.4170336833045 1562.9912269305864 1568.5654249840857 1574.1396398593458 1579.7138523314973 1585.2880503849965 1590.8622628571482 1596.4364681199734 1602.0106853983423 1607.584888258059 1613.159081505341 1618.7332915743837 1624.3075064496438 1629.881714115578 1635.455931393947 1641.0301270443374 1646.6043419195976 1652.1785327637708 1657.752740429705 1663.3269745298346 1668.9011653740079;
        2034.9292407477624 2042.1493471692606 2049.3694608000847 2056.5895816402353 2063.8097024803856 2071.02981611121 2078.2499273389253 2085.4700457759673 2092.6901642130088 2099.910270634507 2107.130393877766 2114.350493089938 2121.570616333197 2128.79073236713 2136.010845997954 2143.230966838105 2150.4510684533857 2157.6711844873184 2164.8913077305774 2172.1114261676194 2179.331554217096 2186.5516558323766 2193.771774269418;
        2523.5898631440887 2532.4558707242595 2541.3219119479522 2550.187929140558 2559.0539511393813 2567.9199659288784 2576.78599033081 2585.6520195389594 2594.518051150217 2603.384075552149 2612.2500975509724 2621.116121952904 2629.9821487579447 2638.848151531898 2647.7141975618083 2656.5802195606316 2665.446239156346 2674.3122491396257 2683.178283153992 2692.044300346598 2700.9103199423125 2709.776363569114 2718.642368746176;
        3012.2504783310887 3022.7624255196715 3033.27436309582 3043.786281447098 3054.298197395268 3064.8101421807423 3075.3220869662164 3085.8340173330384 3096.3459476998605 3106.8578804697913 3117.3698108366134 3127.881736397218 3138.393659554714 3148.905589921536 3159.4175154821405 3169.9294602676146 3180.441381022002 3190.9533210012587 3201.4652561742982 3211.9771937504465 3222.489109698616 3233.0010544840907 3243.512987254021;
        3500.911105533632 3513.0689562839966 3525.226797421926 3537.384628947421 3549.542472488459 3561.7003256419325 3573.8581523612097 3586.015995902248 3598.1738226215257 3610.331658953238 3622.489526525363 3634.647350841532 3646.805189576353 3658.9630114894126 3671.120879061538 3683.2787177963587 3695.4365589342883 3707.5943808473485 3719.7522267914956 3731.9100511076645 3744.067906664246 3756.225767027045 3768.3836033587577;
        3989.5717399455016 4003.3754942576475 4017.1792317480326 4030.9829932695047 4044.786757194085 4058.590473056492 4072.3942201593118 4086.197972068349 4100.001714364951 4113.805480692641 4127.609222989243 4141.412986913824 4155.216714791774 4169.02046910392 4182.824213803631 4196.627960906451 4210.431710412379 4224.235455112091 4238.039211827345 4251.8429445115125 4265.646701226768 4279.450453135804 4293.254188223081;
        4478.232371954263 4493.6820346344075 4509.1316828959 4524.581340769827 4540.030991434429 4555.480649308356 4570.930309585392 4586.379962653102 4601.829615720812 4617.279259176087 4632.728936274884 4648.1785749239425 4663.628242410305 4679.077907493558 4694.527553351942 4709.977206419651 4725.426871502905 4740.876519764398 4756.326184847651 4771.775830706035 4787.225483773745 4802.675148856998 4818.124801924709;
        4966.892999156807 4983.988577414276 5001.084124431332 5018.179695479476 5035.275256915185 5052.370806335351 5069.466386995929 5086.56194362542 5103.657507464238 5120.753083318599 5137.848637544982 5154.944198980691 5172.039770028834 5189.135324255217 5206.230890497143 5223.3264591421785 5240.422022980996 5257.517596432248 5274.613138643088 5291.708714497448 5308.804278336266 5325.899844578193 5342.99540841701;
        5455.553635971784 5474.29509135684 5493.036573176091 5511.778040576691 5530.519532008377 5549.26098979654 5568.002459600249 5586.743934210173 5605.485399207664 5624.226888236241 5642.968336411971 5661.70980381257 5680.451290438038 5699.1927674510725 5717.934239657889 5736.675716670922 5755.417176862195 5774.158646665904 5792.900109260285 5811.641574257776 5830.383070495679 5849.124537896278 5867.866010103095;
        5944.214258368111 5964.601636539817 5984.989024323959 6005.376400092557 6025.763761442502 6046.151166048405 6066.53852739835 6086.9259175856005 6107.313293354198 6127.70068113834 6148.088042488285 6168.475437481753 6188.862810847242 6209.250198631384 6229.637569593764 6250.024959781014 6270.412342758938 6290.7997161244275 6311.187099102352 6331.574472467841 6351.961841027113 6372.349233617471 6392.736621401613;
        6432.874883167546 6454.908169707251 6476.941465859391 6498.974752399096 6521.008041341911 6543.0413302847255 6565.074602405779 6587.10789615481 6609.141180291407 6631.17447163733 6653.20776778947 6675.241054329175 6697.2743288533375 6719.307610586826 6741.340925963836 6763.374205294215 6785.407489430812 6807.440792792278 6829.474062510223 6851.50734664682 6873.540640395851 6895.573917323122 6917.607213475262;
        6921.535507966981 6945.214717293336 6968.893909797933 6992.573109511854 7016.252299613341 7039.931506536589 7063.610687025642 7087.289881933346 7110.969084050376 7134.648286167407 7158.327473865786 7182.006666370381 7205.685868487411 7229.365068201333 7253.0442607059285 7276.723453210525 7300.402640908903 7324.081835816607 7347.761042739855 7371.440235244451 7395.119427749047 7418.798627462968 7442.477822370673;
        7410.1961447819585 7435.521238445227 7460.846356139583 7486.171457012178 7511.496562690989 7536.821680385345 7562.1467620330695 7587.471877324317 7612.796975793803 7638.122062247745 7663.447172732774 7688.772283217803 7714.097386493507 7739.422499381645 7764.747607463565 7790.072689111291 7815.397801999428 7840.722898065806 7866.048015760161 7891.37312624519 7916.698215102242 7942.023323184162 7967.348433669192;
        7898.856784000046 7925.8277980468565 7952.798792868798 7979.769828543587 8006.740820962419 8033.711822993687 8060.682839443606 8087.653858296635 8114.624865134119 8141.595881584039 8168.566883615306 8195.537909677661 8222.50890930582 8249.479918546413 8276.450947011876 8303.42195384936 8330.392951074411 8357.363974733657 8384.33498157114 8411.305986005516 8438.277016874088 8465.248026114683 8492.219037758385;
        8387.517403993263 8416.134326408073 8444.751251225991 8473.368171237693 8501.98507683074 8530.601992036225 8559.218914451034 8587.835829656518 8616.452759280653 8645.069684098573 8673.686587288512 8702.303519315758 8730.920446536784 8759.53735693605 8788.154281753968 8816.771192153235 8845.38812177737 8874.005029773529 8902.621940172796 8931.238874603148 8959.855797017957 8988.47271702966 9017.089646653794;
        8876.178026389589 8906.440864381724 8936.70368314899 8966.966497110037 8997.229342311497 9027.492173094306 9057.755011086441 9088.017827450598 9118.28065342719 9148.54347700067 9178.806319799023 9209.069126550745 9239.331950124228 9269.594790519472 9299.857633317823 9330.12043045711 9360.383270852353 9390.646101635162 9420.908927611754 9451.171768006996 9481.43458917737 9511.697424766397 9541.96024593677;
        9364.838670413894 9396.747399952266 9428.656122281312 9460.56486143212 9492.473617404688 9524.382334927519 9556.291074078326 9588.199798810481 9620.108533155071 9652.017279515205 9683.926013859795 9715.834753010602 9747.743477742757 9779.652214490456 9811.56096085059 9843.469690388963 9875.378434345988 9907.28718551234 9939.195893422733 9971.10465179841 10003.013395755435 10034.922110875155 10066.830857235289;
        9853.499285600894 9887.053928313482 9920.60857583229 9954.163220947987 9987.717863660575 10021.272501566948 10054.827149085753 10088.381801410778 10121.93643691404 10155.491101254607 10189.045729548545 10222.600367454916 10256.155007764397 10289.70966008942 10323.264307608226 10356.818945514598 10390.373585824078 10423.928238149103 10457.482880861691 10491.03752117117 10524.592161480652 10558.146825821219 10591.701454115157;
        10342.15992722209 10377.36047109335 10412.561017367721 10447.761582866962 10482.962136350658 10518.162670609485 10553.363216883856 10588.563779979988 10623.764331060574 10658.964884544272 10694.165430818643 10729.365986705447 10764.566544995361 10799.767081657297 10834.967647156536 10870.168195834016 10905.368749317713 10940.569285979649 10975.769858688214 11010.970392947042 11046.170956043174 11081.371499914434 11116.572070219892;
        10830.820552021523 10867.667009067003 10904.513456500046 10941.359930367285 10978.206382606546 11015.052854070675 11051.899306309937 11088.745756146089 11125.592222804002 11162.438677446371 11199.285141701175 11236.131596343545 11272.978065404566 11309.824522450044 11346.670989107955 11383.517436541 11420.363895989587 11457.210350631956 11494.056812483652 11530.903281544674 11567.749738590152 11604.596198038738 11641.442674309084;
        11319.48117441785 11357.973542234437 11396.465905244804 11434.958285076933 11473.450640877976 11511.94301109767 11550.435376511146 11588.927746730842 11627.420116950536 11665.91248717023 11704.404838165055 11742.897220400293 11781.389576201334 11819.881958436572 11858.374326253159 11896.866686860418 11935.359049870785 11973.851427299805 12012.343804728827 12050.836150917434 12089.328530749563 12127.820886550606 12166.313261576517;
        11808.141794411067 11848.280085014305 11888.418346780238 11928.556642189691 11968.694903955624 12008.833187349534 12048.971465937228 12089.109730106267 12129.248003887744 12169.386292087873 12209.524549047586 12249.662842053933 12289.801132657169 12329.939380004449 12370.07766339836 12410.215941986053 12450.354210961312 12490.492489549004 12530.630772942915 12570.769034708848 12610.907310893432 12651.045589481126 12691.183868068818
    ]

    for n in 1:48
        A = rand(n, n)
        B = rand(n, n)
        C = similar(A)
        AB = A * B
        @test ExponentialUtilities.naivemul!(
            C, A, B, axes(C, 1), axes(C, 2), Val(2),
            Val(1)
        ) ≈ AB
        @test ExponentialUtilities.naivemul!(
            C, A, B, axes(C, 1), axes(C, 2), Val(4),
            Val(2)
        ) ≈ AB
        @test ExponentialUtilities.naivemul!(
            C, A, B, axes(C, 1), axes(C, 2), Val(4),
            Val(3)
        ) ≈ AB
        if n ≤ 16
            Am = MMatrix{n, n}(A)
            Bm = MMatrix{n, n}(B)
            Cm = MMatrix{n, n}(A)
            @test ExponentialUtilities.naivemul!(
                Cm, Am, Bm, axes(Cm, 1), axes(Cm, 2),
                Val(2), Val(2)
            ) ≈ AB
            @test ExponentialUtilities.naivemul!(
                Cm, Am, Bm, axes(Cm, 1), axes(Cm, 2),
                Val(4), Val(2)
            ) ≈ AB
            @test ExponentialUtilities.naivemul!(
                Cm, Am, Bm, axes(Cm, 1), axes(Cm, 2),
                Val(4), Val(3)
            ) ≈ AB
        end
    end
    A = @SMatrix rand(7, 7)
    Am = MMatrix(A)
    Aa = Matrix(A)
    @test exp(Aa) ≈ exp_generic(Aa) ≈ exp_generic(Am) ≈ exp_generic(A)
end

@testset "Phi" begin
    # Scalar phi
    K = 4
    z = 0.1
    P = fill(0.0, K + 1)
    P[1] = exp(z)
    for i in 1:K
        P[i + 1] = (P[i] - 1 / factorial(i - 1)) / z
    end
    @test phi(z, K) ≈ P

    # Matrix phi (dense)
    A = [0.1 0.2; 0.3 0.4]
    P = Vector{Matrix{Float64}}(undef, K + 1)
    P[1] = exp(A)
    for i in 1:K
        P[i + 1] = (P[i] - 1 / factorial(i - 1) * I) / A
    end
    @test phi(A, K) ≈ P

    # Matrix phi (Diagonal)
    A = Diagonal([0.1, 0.2, 0.3, 0.4])
    Afull = Matrix(A)
    P = phi(A, K)
    Pfull = phi(Afull, K)
    for i in 1:(K + 1)
        @test Matrix(P[i]) ≈ Pfull[i]
    end
end

@testset "Phi (Al-Mohy--Liu vs reference)" begin
    # Independent reference: phi_0..phi_p are the first block row of exp(W) for
    # the augmented block matrix W = [A E; 0 J] (Al-Mohy--Liu, Theorem 2.1), with
    # E = [I 0 ... 0] and J the nilpotent shift. This is unrelated to the
    # scaling-and-recovering algorithm under test.
    function phi_block_reference(A, p)
        # `local`: without it these assignments capture and clobber the
        # enclosing testset's variables of the same name (closures in a local
        # scope assign to existing outer locals).
        local n = size(A, 1)
        local T = eltype(A)
        W = zeros(T, n * (p + 1), n * (p + 1))
        W[1:n, 1:n] .= A
        for j in 0:(p - 1)
            rb = j * n
            W[(rb + 1):(rb + n), (rb + n + 1):(rb + 2n)] .= Matrix{T}(I, n, n)
        end
        eW = exp(W)
        return [eW[1:n, (j * n + 1):((j + 1) * n)] for j in 0:p]
    end

    # Force the legacy basis-vector path (Sidje/`phiv_dense`) for cross-checking.
    function phi_legacy(A, p)
        local n = size(A, 1)
        local T = eltype(A)
        local caches = (
            Vector{T}(undef, n), Matrix{T}(undef, n, p + 1),
            Matrix{T}(undef, n + p, n + p),
        )
        local out = [Matrix{T}(undef, n, n) for _ in 1:(p + 1)]
        return phi!(out, A, p; caches = caches)
    end

    Random.seed!(20250701)
    testmats = Dict(
        "small 2x2" => [0.1 0.2; 0.3 0.4],
        "random 8x8" => randn(8, 8) ./ 4,
        "nonnormal 10x10" => 3 * (triu(randn(10, 10), 1) + Diagonal(randn(10))),
        "large-norm 6x6" => 6 * randn(6, 6),
        "complex 5x5" => randn(ComplexF64, 5, 5),
        "hessenberg 12x12" => Matrix(hessenberg(randn(12, 12)).H),
        "zero 4x4" => zeros(4, 4),
    )
    for (name, A) in testmats, p in (1, 2, 3, 5)
        ref = phi_block_reference(A, p)
        new = phi(A, p)                 # default path == Al-Mohy--Liu
        leg = phi_legacy(A, p)
        for j in 1:(p + 1)
            @test new[j] ≈ ref[j] rtol = 1.0e-8 atol = 1.0e-12
            @test new[j] ≈ leg[j] rtol = 1.0e-7 atol = 1.0e-10
        end
    end

    # phi_0 must equal the matrix exponential.
    A = randn(9, 9) ./ 3
    @test phi(A, 4)[1] ≈ exp(A)

    # Large p exercises the recurrence form of the backward-error coefficient
    # (naive factorial ratios overflow long before their quotient does).
    A = randn(3, 3)
    P30 = phi(A, 30)
    ref30 = phi_block_reference(A, 30)
    for j in 1:31
        @test P30[j] ≈ ref30[j] rtol = 1.0e-8 atol = 1.0e-13
    end

    for normTs in (1.0e-8, 1.0e-2, 0.5, 1.0, 1.9), p in (60, 200, 500)
        m = 1
        phat = 0
        delta = (p - 1) * (p - phat) / p + 1
        K = 2m + p + 1
        logcoeff_big = -sum(log(big(m + p + j)) for j in 1:m) -
            sum(log(big(m + j)) for j in 1:(m + p + 1))
        ref = Float64(exp((logcoeff_big - big(delta) * log(big(normTs))) / big(K)))
        c = ExponentialUtilities._phi_be_scale(m, p, normTs, delta, K)
        @test isfinite(c)
        @test c ≈ ref rtol = 5.0e-14
    end

    # Preallocated in-place entry point returns phi_j in out[j+1].
    n = 7
    A = randn(n, n) ./ 3
    out = [Matrix{Float64}(undef, n, n) for _ in 1:4]
    phi!(out, A, 3)
    @test out ≈ phi_block_reference(A, 3)

    # Reusable workspace: same result, and reuse is allocation-free (the linear
    # solve runs through the workspace's LinearSolve cache, which owns the LU
    # pivot storage).
    cache = PhiPadeCache(A, 3)
    out2 = [Matrix{Float64}(undef, n, n) for _ in 1:4]
    phi!(out2, A, 3; caches = cache)
    @test out2 ≈ out
    run_phi!(o, M, c) = phi!(o, M, 3; caches = c)
    run_phi!(out2, A, cache)                           # warm up / compile
    @test (@allocated run_phi!(out2, A, cache)) == 0

    # Complex-matrix workspace path, also allocation-free on reuse.
    Ac = randn(ComplexF64, 6, 6) ./ 3
    cachec = PhiPadeCache(Ac, 2)
    outc = [Matrix{ComplexF64}(undef, 6, 6) for _ in 1:3]
    phi!(outc, Ac, 2; caches = cachec)
    @test outc ≈ phi_block_reference(Ac, 2)
    @test cachec.info[] == 0
    run_phic!(o, M, c) = phi!(o, M, 2; caches = c)
    run_phic!(outc, Ac, cachec)
    @test (@allocated run_phic!(outc, Ac, cachec)) == 0

    # Numerical failure must not throw: NaN input yields NaN-filled outputs and
    # a nonzero return code in cache.info[], so adaptive integrators can reject
    # the step instead of aborting.
    Anan = fill(NaN, 5, 5)
    Pnan = phi(Anan, 2)
    @test all(P -> any(!isfinite, P), Pnan)
    cachenan = PhiPadeCache(Anan, 2)
    outnan = [Matrix{Float64}(undef, 5, 5) for _ in 1:3]
    phi!(outnan, Anan, 2; caches = cachenan)
    @test cachenan.info[] != 0
    @test all(P -> all(isnan, P), outnan)
    # ... and a subsequent good call resets the code.
    phi!(outnan, randn(5, 5), 2; caches = cachenan)
    @test cachenan.info[] == 0

    # Wrongly sized workspaces are programmer errors, not numerical failures,
    # and do throw.
    A3 = randn(3, 3)
    out3 = [Matrix{Float64}(undef, 3, 3) for _ in 1:4]
    @test_throws DimensionMismatch phi!(out3, A3, 3; caches = cache)
    cache_p1 = PhiPadeCache(A, 1)
    out11 = [Matrix{Float64}(undef, n, n) for _ in 1:11]
    @test_throws ArgumentError phi!(out11, A, 10; caches = cache_p1)
end

@testset "Phi static arrays (container-preserving)" begin
    Random.seed!(4321)
    function phi_block_reference(A, p)
        local n = size(A, 1)
        local T = eltype(A)
        W = zeros(T, n * (p + 1), n * (p + 1))
        W[1:n, 1:n] .= A
        for j in 0:(p - 1)
            rb = j * n
            W[(rb + 1):(rb + n), (rb + n + 1):(rb + 2n)] .= Matrix{T}(I, n, n)
        end
        eW = exp(W)
        return [eW[1:n, (j * n + 1):((j + 1) * n)] for j in 0:p]
    end
    for N in (3, 5), p in (1, 2, 3)
        A = SMatrix{N, N}(randn(N, N) ./ 3)
        Rm = ExponentialUtilities._phi_almohy_generic(A, p)
        @test all(r -> r isa SMatrix, Rm)        # static in, static out
        ref = phi_block_reference(Matrix(A), p)
        for j in 1:(p + 1)
            @test Matrix(Rm[j]) ≈ ref[j] rtol = 1.0e-8 atol = 1.0e-12
        end
    end
end

@testset "Static Arrays" begin
    Random.seed!(0)
    for N in (3, 4, 6, 8), t in (0.1, 1.0, 10.0)

        A = I + randn(SMatrix{N, N, Float64}) / 3
        b = randn(SVector{N, Float64})
        # Reference via the dense LAPACK matrix exponential rather than
        # `exp(t * A)` on the SMatrix. StaticArrays' own SMatrix `exp` uses an
        # unbalanced scaling-and-squaring Padé path that loses ~7-9 digits for
        # the larger non-normal cases on some platforms (observed: macOS +
        # Julia prerelease, relerr ~1e-7..1e-5), whereas LAPACK's balanced
        # `exp` is accurate everywhere. Checked against a 512-bit BigFloat
        # `exp(t*A)*b`, the `expv` output here is correct to ~1e-16 on every
        # platform; it was the StaticArrays reference, not `expv`, that drifted.
        # Comparing against the dense reference keeps this a machine-precision
        # (default-tolerance) assertion that still catches real `expv` regressions.
        @test expv(t, A, b) ≈ exp(t * Matrix(A)) * Vector(b)
    end
end

@testset "Arnoldi & Krylov" begin
    Random.seed!(0)
    n = 20
    m = 5
    K = 4
    A = randn(n, n)
    t = 1.0e-2
    b = randn(n)
    direct = exp(t * A) * b
    @test direct ≈ expv(t, A, b; m = m)
    @test direct ≈ expv(t, A, b; m = m, expmethod = ExpMethodHigham2005(false))
    @test direct ≈ kiops(t, A, b)[1]
    P = phi(t * A, K)
    W = fill(0.0, n, K + 1)
    for i in 1:(K + 1)
        W[:, i] = P[i] * b
    end
    Ks = arnoldi(A, b; m = m)
    W_approx = phiv(t, Ks, K)
    @test W ≈ W_approx
    W_approx_kiops3 = kiops(t, A, hcat([b * inv(t)^i for i in 0:(K - 1)]...))
    @test sum(W[:, 1:K], dims = 2) ≈ W_approx_kiops3[1]
    @test_skip begin
        W_approx_kiops4 = kiops(t, A, hcat([b * inv(t)^i for i in 0:K]...))
        @test_broken sum(W[:, 1:(K + 1)], dims = 2) ≈ W_approx_kiops4[1]
        @test sum(W[:, 1:(K + 1)], dims = 2) ≈ W_approx_kiops4[1] atol = 1.0e-2
    end

    # Happy-breakdown in Krylov
    v = normalize(randn(n))
    A = v * v' # A is Idempotent
    Ks = arnoldi(A, b)
    @test Ks.m == 2

    # Test Arnoldi with zero input
    z = zeros(n)
    Ksz = arnoldi(A, z)
    wz = expv(t, A, z; m = m)
    @test norm(wz) == 0.0

    # Arnoldi vs Lanczos
    A = Hermitian(randn(n, n))
    Aperm = A + 1.0e-10 * randn(n, n) # no longer Hermitian
    w = expv(t, A, b; m = m)
    wperm = expv(t, Aperm, b; m = m, opnorm = opnorm)
    wkiops = kiops(t, A, b; m = m)[1]
    @test w ≈ wperm
    @test w ≈ wkiops

    # Test Lanczos with zero input
    wz = expv(t, A, z; m = m)
    @test norm(wz) == 0.0

    # Test matrix version of phiv
    n = 30
    A = diagm(-1 => ones(n - 1), 0 => 30 * ones(n), 1 => ones(n - 1))
    t = 0.1
    Q = phiv(t, A, ones(n), 10)
    @test Q[:, 2] ≈ (t * A) \ (exp(t * A) - I) * ones(n)
end

@testset "PhivCache reuse (issue: cache reallocated every call)" begin
    Random.seed!(0)
    n, m, k = 40, 10, 3
    A = randn(n, n) / 5
    b = randn(n)
    Ks = arnoldi(A, b; m = m)
    w = Matrix{Float64}(undef, n, k + 1)
    cache = ExponentialUtilities.PhivCache(b, m, k + 1)
    phiv!(w, 0.1, Ks, k; cache = cache)
    mem = cache.mem
    phiv!(w, 0.1, Ks, k; cache = cache)
    # A correctly-sized cache must not be reallocated on repeated calls
    @test cache.mem === mem
    # Correctness against the uncached path
    w2 = Matrix{Float64}(undef, n, k + 1)
    phiv!(w2, 0.1, Ks, k)
    @test w ≈ w2
    # An undersized cache must still grow to fit
    smallcache = ExponentialUtilities.PhivCache(b, 2, 1)
    phiv!(w, 0.1, Ks, k; cache = smallcache)
    @test w ≈ w2
end

@testset "exponential! workspace reuse in phiv!/expv!" begin
    Random.seed!(0)
    n, m = 40, 10
    A = randn(n, n) / 5
    b = randn(n)
    Ks = arnoldi(A, b; m = m)
    cache = ExponentialUtilities.PhivCache(b, m, 5)
    # Alternating phi orders (as the ETDRK/EPIRK integrators do) must reuse
    # one workspace per extended-matrix size and stay correct
    for k in (1, 3, 1, 3)
        w = Matrix{Float64}(undef, n, k + 1)
        phiv!(w, 0.1, Ks, k; cache = cache)
        @test w ≈ phiv(0.1, Ks, k)
    end
    entries = cache.expcache
    @test entries isa Vector{Any}
    @test length(entries) == 2 # one per distinct size, not one per call
    # expv! with an ExpvCache reuses its workspace and stays correct
    ecache = ExponentialUtilities.ExpvCache{Float64}(m)
    w1 = zeros(n)
    expv!(w1, 0.1, Ks; cache = ecache)
    entries1 = ecache.expcache
    expv!(w1, 0.1, Ks; cache = ecache)
    @test ecache.expcache === entries1
    @test w1 ≈ expv(0.1, Ks)
end

@testset "Complex Value" begin
    n = 20
    m = 10
    for A in [
            Hermitian(rand(ComplexF64, n, n)),
            Hermitian(rand(n, n)),
            rand(ComplexF64, n, n),
            rand(n, n),
        ]
        for b in [rand(ComplexF64, n), rand(n)], t in [1.0e-2, 1.0e-2im, 1.0e-2 + 1.0e-2im]

            @test exp(t * A) * b ≈ expv(t, A, b; m = m)
        end
    end
end

@testset "Adaptive Krylov" begin
    # Internal time-stepping for Krylov (with adaptation)
    n = 100
    K = 4
    t = 5.0
    tol = 1.0e-7
    A = spdiagm(-1 => ones(n - 1), 0 => -2 * ones(n), 1 => ones(n - 1))
    # This input puts the first Arnoldi step just inside the adaptation acceptance
    # boundary, so it exercises the safety factor on the estimated norm scale.
    B = randn(Random.Xoshiro(14), n, K + 1)
    Phi_half = phi(t / 2 * A, K)
    Phi = phi(t * A, K)
    uhalf_exact = sum((t / 2)^i * Phi_half[i + 1] * B[:, i + 1] for i in 0:K)
    u_exact = sum(t^i * Phi[i + 1] * B[:, i + 1] for i in 0:K)
    U = phiv_timestep([t / 2, t], A, B; adaptive = true, tol = tol)
    @test norm(U[:, 1] - uhalf_exact) / norm(uhalf_exact) < tol
    @test norm(U[:, 2] - u_exact) / norm(u_exact) < tol
    # p = 0 special case (expv_timestep)
    u_exact = Phi[1] * B[:, 1]
    u = expv_timestep(t, A, B[:, 1]; adaptive = true, tol = tol, opnorm = opnorm)
    @test_nowarn expv_timestep(
        t, A, B[:, 1]; adaptive = true, tol = tol,
        opnorm = opnorm(A, Inf)
    )
    @test norm(u - u_exact) / norm(u_exact) < tol
end

@testset "matrix-free default tolerance (Arnoldi estimate)" begin
    # A matrix wrapper whose `opnorm` throws, but which supports the operations
    # the Krylov time stepper needs. The default tolerance scale is estimated
    # matrix-free from the Arnoldi Hessenberg, so it must run on this operator
    # without ever calling `opnorm`.
    struct NoOpnorm{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
        A::M
    end
    Base.size(A::NoOpnorm) = size(A.A)
    Base.getindex(A::NoOpnorm, I...) = getindex(A.A, I...)
    LinearAlgebra.mul!(y, A::NoOpnorm, x) = mul!(y, A.A, x)
    LinearAlgebra.ishermitian(A::NoOpnorm) = ishermitian(A.A)
    LinearAlgebra.opnorm(::NoOpnorm, ::Real) = error("opnorm intentionally unsupported")

    n = 50
    t = 3.0
    tol = 1.0e-7
    Ad = spdiagm(-1 => ones(n - 1), 0 => -2 * ones(n), 1 => ones(n - 1))
    b = randn(n)
    u_exact = exp(Matrix(t * Ad)) * b

    # Default (matrix-free) path is accurate on a matrix that supports opnorm...
    u_def = expv_timestep(t, Ad, b; adaptive = true, tol = tol)
    @test norm(u_def - u_exact) / norm(u_exact) < 1.0e-5

    # ...and runs on an operator with no `opnorm` method at all -- the point of
    # the default no longer evaluating `opnorm(A, Inf)`.
    A = NoOpnorm(Matrix(Ad))
    u_mf = expv_timestep(t, A, b; adaptive = true, tol = tol)
    @test norm(u_mf - u_exact) / norm(u_exact) < 1.0e-5

    # A scalar `opnorm` override still selects the explicit operator-norm path.
    u_ovr = expv_timestep(
        t, A, b; adaptive = true, tol = tol, opnorm = opnorm(Ad, Inf)
    )
    @test norm(u_ovr - u_exact) / norm(u_exact) < 1.0e-5
end

@testset "Krylov for Hermitian matrices" begin
    # Hermitian matrices have real spectra. Ensure that the subspace
    # matrix is representable as a real matrix.

    n = 100
    m = 15
    tol = 1.0e-14

    e = ones(n)
    p = -im * Tridiagonal(-e[2:end], 0e, e[2:end])

    KsA = KrylovSubspace{ComplexF64}(n, m)
    KsL = KrylovSubspace{ComplexF64, Float64}(n, m)

    v = rand(ComplexF64, n)

    arnoldi!(KsA, p, v)
    lanczos!(KsL, p, v)

    AH = view(KsA.H, 1:(KsA.m), 1:(KsA.m))
    LH = view(KsL.H, 1:(KsL.m), 1:(KsL.m))

    @test norm(AH - LH) / norm(AH) < tol
end

@testset "Alternative Lanczos expv Interface" begin
    n = 300
    m = 30

    A = Hermitian(rand(n, n))
    b = rand(ComplexF64, n)
    dt = 0.1

    atol = 1.0e-10
    rtol = 1.0e-10
    w = expv(-im, dt * A, b, m = m, tol = atol, rtol = rtol, mode = :error_estimate)

    function fullexp(A, v)
        w = similar(v)
        eA = exp(A)
        mul!(w, eA, v)
        w
    end

    w′ = fullexp(-im * dt * A, b)

    δw = norm(w - w′)
    @test δw < atol
    @test δw / abs(1.0e-16 + norm(w)) < rtol

    z = zeros(ComplexF64, n)
    wz = expv(-im, dt * A, z, m = m, tol = atol, rtol = rtol, mode = :error_estimate)
    @test norm(wz) == 0
end

module ExternalMatrixFreeOperator
    using LinearAlgebra
    export Operator

    struct Operator{T, M <: AbstractMatrix{T}}
        data::M
    end

    Base.eltype(::Operator{T}) where {T} = T
    Base.size(A::Operator, dim::Integer) = size(A.data, dim)
    LinearAlgebra.mul!(y::AbstractVector, A::Operator, x::AbstractVector) =
        mul!(y, A.data, x)
    LinearAlgebra.ishermitian(A::Operator) = ishermitian(A.data)
end

@testset "Matrix-free operator generic interface" begin
    Random.seed!(123)
    n = 20
    for ishermitian in (false, true)
        A = rand(ComplexF64, n, n)
        M = ishermitian ? A'A : A
        Op = ExternalMatrixFreeOperator.Operator(M)
        b = rand(ComplexF64, n)
        Ks = arnoldi(Op, b; ishermitian, tol = 1.0e-12)
        pv = phiv(0.01, Ks, 2)
        pv′ = hcat(map(A -> A * b, phi(0.01Op.data, 2))...)

        @test pv ≈ pv′ atol = 1.0e-12
        @test expv(0.01, Op, b; m = n, ishermitian) ≈ exp(0.01M) * b atol = 1.0e-12
    end
end

@testset "Issue 206 - Double64 with pade order 13" begin
    using ExponentialUtilities: pade_order_for_type, _needs_higher_order

    # Double64 has ~106 bits precision, which is > 64 so _needs_higher_order returns true,
    # but pade_order_for_type returns 13, which previously caused infinite recursion
    @test _needs_higher_order(Double64) == true
    @test pade_order_for_type(Double64) == 13

    # Test that exponential! doesn't cause infinite recursion with matrices
    # (The original issue used vectors which have additional opnorm issues)
    A = Double64[1.0 0.5; 0.5 1.0]
    A_float = [Float64(A[i, j]) for i in 1:2, j in 1:2]
    result_matrix = exponential!(copy(A), ExpMethodGeneric{Val{13}()}())
    expected = exp(A_float)
    result_float = [Float64(result_matrix[i, j]) for i in 1:2, j in 1:2]
    @test result_float ≈ expected rtol = 1.0e-14

    # Test with default ExpMethodGeneric() as well
    result_default = exponential!(copy(A), ExpMethodGeneric())
    result_default_float = [Float64(result_default[i, j]) for i in 1:2, j in 1:2]
    @test result_default_float ≈ expected rtol = 1.0e-14

    # Test scalar case (which doesn't need opnorm)
    scalar_result = exponential!(Double64(1.0), ExpMethodGeneric{Val{13}()}())
    @test Float64(scalar_result) ≈ exp(1.0) rtol = 1.0e-14
end

@testset "Krylov cache reuse is non-allocating and size-independent" begin
    # Regression guard for the workspace-reuse fixes (PhivCache/ExpvCache no
    # longer reallocate the phiv/exp scratch buffers on every call). AllocCheck's
    # static check_allocs cannot certify this: the workspace is allocated lazily
    # on the first cache miss and reused afterwards, which is a runtime/temporal
    # property invisible to static analysis (and LinearSolve/BLAS/the dynamic
    # dispatch over the heterogeneous expcache store are flagged regardless). The
    # meaningful invariant is instead measured at runtime: after warmup the
    # per-call allocation is a small constant that does NOT grow with the problem
    # size n. Before the fix it scaled as O(n^2) plus a LinearSolve init per call.
    m = 30

    # Deterministic, non-symmetric operator so arnoldi takes the general (Higham)
    # path rather than the Lanczos/eigen path that the workspace fix does not touch.
    mkA(n) = [
        i == j ? -2.0 : 0.1 / (1 + abs(i - j)) * (i < j ? 1.0 : 0.5)
            for i in 1:n, j in 1:n
    ]

    # `minimum` over repeated calls filters one-off JIT/first-execution
    # allocations; the buffers are all preallocated before the measured loop.
    function phiv_alloc(n; k = 3)
        A = mkA(n)
        b = [1.0 / i for i in 1:n]
        Ks = arnoldi(A, b; m = m)
        w = Matrix{Float64}(undef, n, k + 1)
        cache = ExponentialUtilities.PhivCache(b, m, k + 1)
        for _ in 1:3
            phiv!(w, 0.1, Ks, k; cache = cache)
        end
        return minimum(@allocated(phiv!(w, 0.1, Ks, k; cache = cache)) for _ in 1:5)
    end

    function expv_alloc(n)
        A = mkA(n)
        b = [1.0 / i for i in 1:n]
        Ks = arnoldi(A, b; m = m)
        w = zeros(n)
        cache = ExponentialUtilities.ExpvCache{Float64}(m)
        for _ in 1:3
            expv!(w, 0.1, Ks; cache = cache)
        end
        return minimum(@allocated(expv!(w, 0.1, Ks; cache = cache)) for _ in 1:5)
    end

    function phiv_timestep_alloc(n; p = 1)
        A = mkA(n)
        B = [1.0 / (i + j) for i in 1:n, j in 1:(p + 1)]
        u = zeros(n)
        caches = ExponentialUtilities._phiv_timestep_caches(u, m, p)
        kws = (; m = m, caches = caches, tau = 0.0, tol = 1.0e-7, adaptive = false)
        for _ in 1:3
            phiv_timestep!(u, 0.05, A, B; kws...)
        end
        return minimum(@allocated(phiv_timestep!(u, 0.05, A, B; kws...)) for _ in 1:5)
    end

    # Size-independence is the load-bearing assertion: equal allocation at a small
    # and a large n proves the scratch buffers are reused, not reallocated. The
    # absolute ceiling catches any return to the O(n^2)+ per-call regression.
    for (name, f, ceiling) in (
            ("phiv!", phiv_alloc, 4096),
            ("expv!", expv_alloc, 4096),
            ("phiv_timestep!", phiv_timestep_alloc, 8192),
        )
        f(32)                         # compile the whole path before measuring
        small = f(64)
        large = f(1024)
        @test small == large          # independent of n -> buffers reused
        @test large <= ceiling        # small constant, not O(n^2)
    end
end
