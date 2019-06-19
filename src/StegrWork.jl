module Stegr
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt
import LinearAlgebra.LAPACK: stegr!
const liblapack = Base.liblapack_name

mutable struct StegrWork{T<:Real}
    jobz::Char
    range::Char
    dv::Vector{T}
    ev::Vector{T}
    vl::Real
    vu::Real
    il::BlasInt
    iu::BlasInt
    abstol::Vector{T}
    m::Vector{BlasInt}
    w::Vector{T}
    Z::Matrix{T}
    isuppz::Vector{BlasInt}
    work::Vector{T}
    lwork::BlasInt
    iwork::Vector{BlasInt}
    liwork::BlasInt
    info::Vector{BlasInt}
end

for (stegr,elty) in ((:dstegr_,:Float64),
                     (:sstegr_,:Float32))
    @eval begin
        function stegr!(n::BlasInt, sw::StegrWork{$elty})
            ldz = stride(sw.Z, 2)
            ccall((@blasfunc($stegr), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{$elty}, Ref{$elty}, Ref{BlasInt},
                   Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                  sw.jobz, sw.range, n,
                  sw.dv, sw.ev,
                  sw.vl, sw.vu, sw.il, sw.iu,
                  sw.abstol, sw.m,
                  sw.w, sw.Z, ldz,
                  sw.isuppz, sw.work, sw.lwork, sw.iwork, sw.liwork,
                  sw.info)
        end
    end
end

"""
    stegr!(α, β, sw)

Diagonalize the real-symmetric tridiagonal matrix with `α` on the
diagonal and `β` on the super-/subdiagonal, using the workspaces
allocated in `sw`.
"""
function stegr!(α::AbstractVector{T}, β::AbstractVector{T}, sw::StegrWork{T}) where T
    # @assert length(sw.dv) >= length(α)
    # @assert length(sw.ev) >= length(β)
    copyto!(sw.dv, α)
    copyto!(sw.ev, β)
    stegr!(BlasInt(length(α)), sw)
end

"""
    StegrWork(T, n)

Allocate work arrays for diagonalization of real-symmetric tridiagonal
matrices of sizes up to `n`×`n`.
"""
function StegrWork(::Type{T}, n::Integer,
                   jobz::Char = 'V', range::Char = 'A') where T
    n = convert(BlasInt, n)
    dv = Array{T}(undef, n)
    ev = Array{T}(undef, n)
    abstol = Array{T}(undef, 1)
    m = Vector{BlasInt}(undef, 1)
    w = Array{T}(undef, n)
    ldz = jobz == 'N' ? 1 : n
    Z = Array{T}(undef, ldz, n)
    isuppz = Array{BlasInt}(undef, 2n)
    work = Array{T}(undef, 1)
    lwork = -one(BlasInt)
    iwork = Array{BlasInt}(undef, 1)
    liwork = -one(BlasInt)
    info = Array{BlasInt}(undef, 1)
    sw = StegrWork(jobz, range,
                   dv, ev,
                   0.0, 0.0,
                   BlasInt(0), BlasInt(0),
                   abstol, m,
                   w, Z,
                   isuppz,
                   work, lwork,
                   iwork, liwork,
                   info)
    # Call stegr! once to query for necessary workspace sizes.
    stegr!(n, sw)
    sw.lwork = BlasInt(sw.work[1])
    sw.work = Array{T}(undef, sw.lwork)
    sw.liwork = sw.iwork[1]
    sw.iwork = Array{BlasInt}(undef, sw.liwork)
    sw
end

export StegrWork
end
