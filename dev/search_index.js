var documenterSearchIndex = {"docs":
[{"location":"expv/#Expv:-Matrix-Exponentials-Times-Vectors","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"","category":"section"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"The main functionality of ExponentialUtilities is the computation of matrix-phi-vector products. The phi functions are defined as","category":"page"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"ϕ_0(z) = exp(z)\nϕ_(k+1)(z) = (ϕ_k(z) - 1) / z","category":"page"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"In exponential algorithms, products in the form of ϕ_m(tA)b are frequently encountered. Instead of computing the matrix function first and then computing the matrix-vector product, the common alternative is to construct a Krylov subspace K_m(A,b) and then approximate the matrix-phi-vector product.","category":"page"},{"location":"expv/#Support-for-matrix-free-operators","page":"Expv: Matrix Exponentials Times Vectors","title":"Support for matrix-free operators","text":"","category":"section"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"You can use any object as the \"matrix\" A as long as it implements the following linear operator interface:","category":"page"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"Base.eltype(A)\nBase.size(A, dim)\nLinearAlgebra.mul!(y, A, x) (for computing y = A * x in place).\nLinearAlgebra.opnorm(A, p=Inf). If this is not implemented or the default implementation can be slow, you can manually pass in the operator norm (a rough estimate is fine) using the keyword argument opnorm.\nLinearAlgebra.ishermitian(A). If this is not implemented or the default implementation can be slow, you can manually pass in the value using the keyword argument ishermitian.","category":"page"},{"location":"expv/#Core-API","page":"Expv: Matrix Exponentials Times Vectors","title":"Core API","text":"","category":"section"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"expv\nphiv\nexpv!\nphiv!\nexpv_timestep\nphiv_timestep\nexpv_timestep!\nphiv_timestep!","category":"page"},{"location":"expv/#ExponentialUtilities.expv","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.expv","text":"expv(t,A,b; kwargs) -> exp(tA)b\n\nCompute the matrix-exponential-vector product using Krylov.\n\nA Krylov subspace is constructed using arnoldi and exp! is called on the Hessenberg matrix. Consult arnoldi for the values of the keyword arguments. An alternative algorithm, where an error estimate generated on-the-fly is used to terminate the Krylov iteration, can be employed by setting the kwarg mode=:error_estimate.\n\nexpv(t,Ks; cache) -> exp(tA)b\n\nCompute the expv product using a pre-constructed Krylov subspace.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.phiv","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.phiv","text":"phiv(t,A,b,k;correct,kwargs) -> [phi_0(tA)b phi_1(tA)b ... phi_k(tA)b][, errest]\n\nCompute the matrix-phi-vector products using Krylov. k >= 1.\n\nThe phi functions are defined as\n\nvarphi_0(z) = exp(z)quad varphi_k+1(z) = fracvarphi_k(z) - 1z\n\nA Krylov subspace is constructed using arnoldi and phiv_dense is called on the Hessenberg matrix. If correct=true, then phi0 through phik-1 are updated using the last Arnoldi vector v_m+1 [1]. If errest=true then an additional error estimate for the second-to-last phi is also returned. For the additional keyword arguments, consult arnoldi.\n\nphiv(t,Ks,k;correct,kwargs) -> [phi0(tA)b phi1(tA)b ... phi_k(tA)b][, errest]\n\nCompute the matrix-phi-vector products using a pre-constructed Krylov subspace.\n\n[1]: Niesen, J., & Wright, W. (2009). A Krylov subspace algorithm for evaluating the φ-functions in exponential integrators. arXiv preprint arXiv:0907.4631. Formula (10).\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.expv!","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.expv!","text":"expv!(w,t,Ks[;cache]) -> w\n\nNon-allocating version of expv that uses precomputed Krylov subspace Ks.\n\n\n\n\n\nexpv!(w, t, A, b, Ks, cache)\n\nAlternative interface for calculating the action of exp(t*A) on the vector b, storing the result in w. The Krylov iteration is terminated when an error estimate for the matrix exponential in the generated subspace is below the requested tolerance. Ks is a KrylovSubspace and typeof(cache)<:HermitianSubspaceCache, the exact type decides which algorithm is used to compute the subspace exponential.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.phiv!","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.phiv!","text":"phiv!(w,t,Ks,k[;cache,correct,errest]) -> w[,errest]\n\nNon-allocating version of 'phiv' that uses precomputed Krylov subspace Ks.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.expv_timestep","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.expv_timestep","text":"expv_timestep(ts,A,b[;adaptive,tol,kwargs...]) -> U\n\nEvaluates the matrix exponentiation-vector product using time stepping\n\nu = exp(tA)b\n\nts is an array of time snapshots for u, with U[:,j] ≈ u(ts[j]). ts can also be just one value, in which case only the end result is returned and U is a vector.\n\nThe time stepping formula of Niesen & Wright is used [1]. If the time step tau is not specified, it is chosen according to (17) of Niesen & Wright. If adaptive==true, the time step and Krylov subspace size adaptation scheme of Niesen & Wright is used, the relative tolerance of which can be set using the keyword parameter tol. The delta and gamma parameters of the adaptation scheme can also be adjusted.\n\nSet verbose=true to print out the internal steps (for debugging). For the other keyword arguments, consult arnoldi and phiv, which are used internally.\n\nNote that this function is just a special case of phiv_timestep with a more intuitive interface (vector b instead of a n-by-1 matrix B).\n\n[1]: Niesen, J., & Wright, W. (2009). A Krylov subspace algorithm for evaluating the φ-functions in exponential integrators. arXiv preprint arXiv:0907.4631.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.phiv_timestep","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.phiv_timestep","text":"phiv_timestep(ts,A,B[;adaptive,tol,kwargs...]) -> U\n\nEvaluates the linear combination of phi-vector products using time stepping\n\nu = varphi_0(tA)b_0 + tvarphi_1(tA)b_1 + cdots + t^pvarphi_p(tA)b_p\n\nts is an array of time snapshots for u, with U[:,j] ≈ u(ts[j]). ts can also be just one value, in which case only the end result is returned and U is a vector.\n\nThe time stepping formula of Niesen & Wright is used [1]. If the time step tau is not specified, it is chosen according to (17) of Niesen & Wright. If adaptive==true, the time step and Krylov subspace size adaptation scheme of Niesen & Wright is used, the relative tolerance of which can be set using the keyword parameter tol. The delta and gamma parameters of the adaptation scheme can also be adjusted.\n\nWhen encountering a happy breakdown in the Krylov subspace construction, the time step is set to the remainder of the time interval since time stepping is no longer necessary.\n\nSet verbose=true to print out the internal steps (for debugging). For the other keyword arguments, consult arnoldi and phiv, which are used internally.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.expv_timestep!","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.expv_timestep!","text":"expv_timestep!(u,t,A,b[;kwargs]) -> u\n\nNon-allocating version of expv_timestep.\n\n\n\n\n\n","category":"function"},{"location":"expv/#ExponentialUtilities.phiv_timestep!","page":"Expv: Matrix Exponentials Times Vectors","title":"ExponentialUtilities.phiv_timestep!","text":"phiv_timestep!(U,ts,A,B[;kwargs]) -> U\n\nNon-allocating version of phiv_timestep.\n\n\n\n\n\n","category":"function"},{"location":"expv/#Caches","page":"Expv: Matrix Exponentials Times Vectors","title":"Caches","text":"","category":"section"},{"location":"expv/","page":"Expv: Matrix Exponentials Times Vectors","title":"Expv: Matrix Exponentials Times Vectors","text":"ExponentialUtilities.ExpvCache\nExponentialUtilities.PhivCache","category":"page"},{"location":"arnoldi/#Arnoldi-Iteration","page":"Arnoldi Iteration","title":"Arnoldi Iteration","text":"","category":"section"},{"location":"arnoldi/","page":"Arnoldi Iteration","title":"Arnoldi Iteration","text":"arnoldi\narnoldi!\nlanczos!","category":"page"},{"location":"arnoldi/#ExponentialUtilities.arnoldi","page":"Arnoldi Iteration","title":"ExponentialUtilities.arnoldi","text":"arnoldi(A,b[;m,tol,opnorm,iop]) -> Ks\n\nPerforms m arnoldi iterations to obtain the Krylov subspace K_m(A,b).\n\nThe n x (m + 1) basis vectors getV(Ks) and the (m + 1) x m upper Hessenberg matrix getH(Ks) are related by the recurrence formula\n\nv_1=b,\\quad Av_j = \\sum_{i=1}^{j+1}h_{ij}v_i\\quad(j = 1,2,\\ldots,m)\n\niop determines the length of the incomplete orthogonalization procedure [1]. The default value of 0 indicates full Arnoldi. For symmetric/Hermitian A, iop will be ignored and the Lanczos algorithm will be used instead.\n\nRefer to KrylovSubspace for more information regarding the output.\n\nHappy-breakdown occurs whenever norm(v_j) < tol * opnorm, in this case, the dimension of Ks is smaller than m.\n\n[1]: Koskela, A. (2015). Approximating the matrix exponential of an advection-diffusion operator using the incomplete orthogonalization method. In Numerical Mathematics and Advanced Applications-ENUMATH 2013 (pp. 345-353). Springer, Cham.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#ExponentialUtilities.arnoldi!","page":"Arnoldi Iteration","title":"ExponentialUtilities.arnoldi!","text":"arnoldi!(Ks,A,b[;tol,m,opnorm,iop,init]) -> Ks\n\nNon-allocating version of arnoldi.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#ExponentialUtilities.lanczos!","page":"Arnoldi Iteration","title":"ExponentialUtilities.lanczos!","text":"lanczos!(Ks,A,b[;tol,m,opnorm]) -> Ks\n\nA variation of arnoldi! that uses the Lanczos algorithm for Hermitian matrices.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#API","page":"Arnoldi Iteration","title":"API","text":"","category":"section"},{"location":"arnoldi/","page":"Arnoldi Iteration","title":"Arnoldi Iteration","text":"KrylovSubspace\nExponentialUtilities.firststep!\nExponentialUtilities.arnoldi_step!\nExponentialUtilities.lanczos_step!\nExponentialUtilities.coeff","category":"page"},{"location":"arnoldi/#ExponentialUtilities.KrylovSubspace","page":"Arnoldi Iteration","title":"ExponentialUtilities.KrylovSubspace","text":"KrylovSubspace{T}(n,[maxiter=30]) -> Ks\n\nConstructs an uninitialized Krylov subspace, which can be filled by arnoldi!.\n\nThe dimension of the subspace, Ks.m, can be dynamically altered but should be smaller than maxiter, the maximum allowed arnoldi iterations.\n\ngetV(Ks) -> V\ngetH(Ks) -> H\n\nAccess methods for the (extended) orthonormal basis V and the (extended) Gram-Schmidt coefficients H. Both methods return a view into the storage arrays and has the correct dimensions as indicated by Ks.m.\n\nresize!(Ks, maxiter) -> Ks\n\nResize Ks to a different maxiter, destroying its contents.\n\nThis is an expensive operation and should be used scarcely.\n\n\n\n\n\n","category":"type"},{"location":"arnoldi/#ExponentialUtilities.firststep!","page":"Arnoldi Iteration","title":"ExponentialUtilities.firststep!","text":"firststep!(Ks, V, H, b) -> nothing\n\nCompute the first step of Arnoldi or Lanczos iteration.\n\n\n\n\n\nfirststep!(Ks, V, H, b, b_aug, t, mu, l) -> nothing\n\nCompute the first step of Arnoldi or Lanczos iteration of augmented system.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#ExponentialUtilities.arnoldi_step!","page":"Arnoldi Iteration","title":"ExponentialUtilities.arnoldi_step!","text":"arnoldi_step!(j, iop, n, A, V, H)\n\nTake the j 'th step of the Lanczos iteration.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#ExponentialUtilities.lanczos_step!","page":"Arnoldi Iteration","title":"ExponentialUtilities.lanczos_step!","text":"lanczos_step!(j, m, n, A, V, H)\n\nTake the j'th step of the Lanczos iteration.\n\n\n\n\n\n","category":"function"},{"location":"arnoldi/#ExponentialUtilities.coeff","page":"Arnoldi Iteration","title":"ExponentialUtilities.coeff","text":"coeff(::Type,α)\n\nHelper functions that returns the real part if that is what is required (for Hermitian matrices), otherwise returns the value as-is.\n\n\n\n\n\n","category":"function"},{"location":"#ExponentialUtilities.jl:-High-Performance-Matrix-Exponentiation-and-Products","page":"Home","title":"ExponentialUtilities.jl: High-Performance Matrix Exponentiation and Products","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ExponentialUtilities is a package of utility functions for matrix functions of exponential type, including functionality for the matrix exponential and phi-functions. The tools are used by the exponential integrators in OrdinaryDiffEq.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install ExponentialUtilities.jl, use the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"ExponentialUtilities\")","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using ExponentialUtilities\n\nA = rand(2, 2)\nexponential!(A)\n\nv = rand(2);\nt = rand();\nexpv(t, A, v)","category":"page"},{"location":"#Matrix-phi-vector-product","page":"Home","title":"Matrix-phi-vector product","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The main functionality of ExponentialUtilities is the computation of matrix-phi-vector products. The phi functions are defined as","category":"page"},{"location":"","page":"Home","title":"Home","text":"ϕ_0(z) = exp(z)\nϕ_(k+1)(z) = (ϕ_k(z) - 1) / z","category":"page"},{"location":"","page":"Home","title":"Home","text":"In exponential algorithms, products in the form of ϕ_m(tA)b are frequently encountered. Instead of computing the matrix function first and then computing the matrix-vector product, the common alternative is to construct a Krylov subspace K_m(A,b) and then approximate the matrix-phi-vector product.","category":"page"},{"location":"#expv-and-phiv","page":"Home","title":"expv and phiv","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"expv(t,A,b;kwargs) -> exp(tA)b\nphiv(t,A,b,k;kwargs) -> [ϕ_0(tA)b ϕ_1(tA)b ... ϕ_k(tA)b][, errest]","category":"page"},{"location":"","page":"Home","title":"Home","text":"For phiv, all ϕ_m(tA)b products up to order k are returned as a matrix. This is because it's more economical to produce all the results at once than individually. A second output is returned if errest=true in kwargs. The error estimate is given for the second-to-last product, using the last product as an estimator. If correct=true, then ϕ_0 through ϕ_(k-1) are updated using the last Arnoldi vector. The correction algorithm is described in [1].","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can adjust how the Krylov subspace is constructed by setting various keyword arguments. See the Arnoldi iteration section for more details.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to SciML.\nSee the SciML Style Guide for common coding practices and other style decisions.\nThere are a few community forums:\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Slack\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Zulip\nOn the Julia Discourse forums\nSee also SciML Community page","category":"page"},{"location":"#Reproducibility","page":"Home","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status(; mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nusing Markdown\nversion = TOML.parse(read(\"../../Project.toml\", String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\", String))[\"name\"]\nlink_manifest = \"https://github.com/SciML/\" * name * \".jl/tree/gh-pages/v\" * version *\n                \"/assets/Manifest.toml\"\nlink_project = \"https://github.com/SciML/\" * name * \".jl/tree/gh-pages/v\" * version *\n               \"/assets/Project.toml\"\nMarkdown.parse(\"\"\"You can also download the\n[manifest]($link_manifest)\nfile and the\n[project]($link_project)\nfile.\n\"\"\")","category":"page"},{"location":"matrix_exponentials/#Matrix-Exponentials","page":"Matrix Exponentials","title":"Matrix Exponentials","text":"","category":"section"},{"location":"matrix_exponentials/","page":"Matrix Exponentials","title":"Matrix Exponentials","text":"exponential!\nphi","category":"page"},{"location":"matrix_exponentials/#ExponentialUtilities.exponential!","page":"Matrix Exponentials","title":"ExponentialUtilities.exponential!","text":"E=exponential!(A,[method [cache]])\n\nComputes the matrix exponential with the method specified in method. The contents of A are modified, allowing for fewer allocations. The method parameter specifies the implementation and implementation parameters, e.g. ExpMethodNative, ExpMethodDiagonalization, ExpMethodGeneric, ExpMethodHigham2005. Memory needed can be preallocated and provided in the parameter cache such that the memory can be recycled when calling exponential! several times. The preallocation is done with the command alloc_mem: cache=alloc_mem(A,method). A may not be sparse matrix type, since exp(A) is likely to be dense.\n\nExample\n\njulia> A = randn(50, 50);\n\njulia> B = A * 2;\n\njulia> method = ExpMethodHigham2005();\n\njulia> cache = ExponentialUtilities.alloc_mem(A, method); # Main allocation done here\n\njulia> E1 = exponential!(A, method, cache) # Very little allocation here\n\njulia> E2 = exponential!(B, method, cache) # Very little allocation here\n\n\n\n\n\n\n","category":"function"},{"location":"matrix_exponentials/#ExponentialUtilities.phi","page":"Matrix Exponentials","title":"ExponentialUtilities.phi","text":"phi(z,k[;cache]) -> [phi_0(z),phi_1(z),...,phi_k(z)]\n\nCompute the scalar phi functions for all orders up to k.\n\nThe phi functions are defined as\n\nvarphi_0(z) = exp(z)quad varphi_k+1(z) = fracvarphi_k(z) - 1z\n\nInstead of using the recurrence relation, which is numerically unstable, a formula given by Sidje is used (Sidje, R. B. (1998). Expokit: a software package for computing matrix exponentials. ACM Transactions on Mathematical Software (TOMS), 24(1), 130-156. Theorem 1).\n\n\n\n\n\nphi(A,k[;cache]) -> [phi_0(A),phi_1(A),...,phi_k(A)]\n\nCompute the matrix phi functions for all orders up to k. k >= 1.\n\nThe phi functions are defined as\n\nvarphi_0(z) = exp(z)quad varphi_k+1(z) = fracvarphi_k(z) - 1z\n\nCalls phiv_dense on each of the basis vectors to obtain the answer. If A is Diagonal, instead calls the scalar phi on each diagonal element and the return values are also Diagonals\n\n\n\n\n\n","category":"function"},{"location":"matrix_exponentials/#Methods","page":"Matrix Exponentials","title":"Methods","text":"","category":"section"},{"location":"matrix_exponentials/","page":"Matrix Exponentials","title":"Matrix Exponentials","text":"ExpMethodHigham2005\nExpMethodHigham2005Base\nExpMethodGeneric\nExpMethodNative\nExpMethodDiagonalization","category":"page"},{"location":"matrix_exponentials/#ExponentialUtilities.ExpMethodHigham2005","page":"Matrix Exponentials","title":"ExponentialUtilities.ExpMethodHigham2005","text":"ExpMethodHigham2005(A::AbstractMatrix);\nExpMethodHigham2005(b::Bool=true);\n\nComputes the matrix exponential using the algorithm Higham, N. J. (2005). \"The scaling and squaring method for the matrix exponential revisited.\" SIAM J. Matrix Anal. Appl.Vol. 26, No. 4, pp. 1179–1193\" based on generated code. If a matrix is specified, balancing is determined automatically.\n\n\n\n\n\n","category":"type"},{"location":"matrix_exponentials/#ExponentialUtilities.ExpMethodHigham2005Base","page":"Matrix Exponentials","title":"ExponentialUtilities.ExpMethodHigham2005Base","text":"ExpMethodHigham2005Base()\n\nThe same as ExpMethodHigham2005 but follows Base.exp closer.\n\n\n\n\n\n","category":"type"},{"location":"matrix_exponentials/#ExponentialUtilities.ExpMethodGeneric","page":"Matrix Exponentials","title":"ExponentialUtilities.ExpMethodGeneric","text":"struct ExpMethodGeneric{T}\nExpMethodGeneric()=ExpMethodGeneric{Val{13}}();\n\nGeneric exponential implementation of the method ExpMethodHigham2005, for any exp argument x  for which the functions LinearAlgebra.opnorm, +, *, ^, and / (including addition with UniformScaling objects) are defined. The type T is used to adjust the number of terms used in the Pade approximants at compile time.\n\nSee \"The Scaling and Squaring Method for the Matrix Exponential Revisited\" by Higham, Nicholas J. in 2005 for algorithm details.\n\n\n\n\n\n","category":"type"},{"location":"matrix_exponentials/#ExponentialUtilities.ExpMethodNative","page":"Matrix Exponentials","title":"ExponentialUtilities.ExpMethodNative","text":"ExpMethodNative()\n\nMatrix exponential method corresponding to calling Base.exp.\n\n\n\n\n\n","category":"type"},{"location":"matrix_exponentials/#ExponentialUtilities.ExpMethodDiagonalization","page":"Matrix Exponentials","title":"ExponentialUtilities.ExpMethodDiagonalization","text":"ExpMethodDiagonalization(enforce_real=true)\n\nMatrix exponential method corresponding to the diagonalization with eigen possibly by removing imaginary part introduced by the numerical approximation.\n\n\n\n\n\n","category":"type"},{"location":"matrix_exponentials/#Utilities","page":"Matrix Exponentials","title":"Utilities","text":"","category":"section"},{"location":"matrix_exponentials/","page":"Matrix Exponentials","title":"Matrix Exponentials","text":"ExponentialUtilities.alloc_mem","category":"page"},{"location":"matrix_exponentials/#ExponentialUtilities.alloc_mem","page":"Matrix Exponentials","title":"ExponentialUtilities.alloc_mem","text":"cache=alloc_mem(A,method)\n\nPre-allocates memory associated with matrix exponential function method and matrix A. To be used in combination with exponential!.\n\n\n\n\n\n","category":"function"}]
}
