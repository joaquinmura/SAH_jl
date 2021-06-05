# Optimization methods: Selection of gradient descent methodsº
# deeply inspired (or copied) from 'Algorithms for Optimization' by Kochenderfer & Wheeler.
# 
# Dependencies (although removable): sah_dependencies
#
# Joaquin Mura, May 2021.

using LinearAlgebra # for 'I' in BFGS !

abstract type 
    DescentMethod 
end


# === Univariate search methods ===
function line_search(f, x, d, params)
    println("here in line-search!")
    objective = (α,p) -> f(x + α*d,p)
    #a, b = bracket_minimum(objective,s=10,p=params) #! not working!
    #α = minimize(objective, a, b)
    # this is not a precise search:
    α_s = LinRange(0.0,20.0,100)
    v = [objective(α_s[i],params) for i in 1:length(α_s)]
    α = α_s[argmin(v)]
    
    return x + α*d
end

function bracket_minimum(f, x=0; s=1e-2, k=2.0,p)
    println("here in bracket_minimum!")
    a, ya = x, f(x,p) 
    b, yb = a + s, f(a + s,p)
    println("before: a=$a, b=$b -- ya=$ya, yb=$yb")
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end
    println("later : a=$a, b=$b -- ya=$ya, yb=$yb")
    while true
        print("*")
        c, yc = b + s, f(b + s,p)
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end


# === Gradient Descent method ===
mutable struct GradientDescent <: DescentMethod
    α::Real
    α_min::Real
end

init!(M::GradientDescent) = M
init!(M::GradientDescent,x) = M


function direction!(M::GradientDescent,x,f,∇f,params::SAHvars;verbose=false)

    dJ = get_∇J(x,params)

    if verbose
        println(" Gradient Descent: |dJ|_2 = ",norm(dJ))
    end

    # optimization direction (need control over α)
    ∂J = dJ/(norm(dJ) + params.ϵ_dJ)
    
    return (-M.α*∂J)
end


# === Momentum method ===
mutable struct Momentum <: DescentMethod
    α::Real   # learning rate
    β::Real   # momentum decay
    v::Vector # momentum
    α_min::Real
end

function init!(M::Momentum,x)
    M.v = zeros(length(x))
    return M
end

function direction!(M::Momentum, x,f,∇f,params::SAHvars;verbose=false)

    ∇J = ∇f(x,params)

    if verbose
        println(" Momentum: |dJ|_2 = ",norm(∇J))
    end

    # optimization direction (need control over α)
    ∂J = ∇J/(norm(∇J) + params.ϵ_dJ)

    M.v[:] = M.β*M.v - M.α*∂J

    return M.v
end


# === Nesterov's Momentum ===
mutable struct NesterovMomentum <: DescentMethod
    α::Real # learning rate
    β::Real # momentum decay
    v::Vector # momentum
    α_min::Real
end
function init!(M::NesterovMomentum, x)
    M.v = zeros(length(x))
    return M
end

function direction!(M::NesterovMomentum, x,f,∇f,params::SAHvars;verbose=false)


    ∇J = ∇f(x + M.β*M.v,params) # original method doesn't have α here!

    if verbose
        println(" Nesterov Momentum: |dJ|_2 = ",norm(∇J))
    end

    # optimization direction (need control over α)
    ∂J = ∇J/(norm(∇J) + params.ϵ_dJ)

    M.v[:] = M.β*M.v - M.α*∂J

    return M.v #/(norm(M.v) + params.ϵ_dJ)

end

# === ADAM ===
#=
mutable struct Adam <: DescentMethod
α # learning rate
γv # decay
γs # decay
ϵ # small value
k # step counter
v # 1st moment estimate
s # 2nd moment estimate
end
function init!(M::Adam, f, ∇f, x)
M.k = 0
M.v = zeros(length(x))
M.s = zeros(length(x))
return M
end
function step!(M::Adam, f, ∇f, x)
α, γv, γs, ϵ, k = M.α, M.γv, M.γs, M.ϵ, M.k
s, v, g = M.s, M.v, ∇f(x)
v[:] = γv*v + (1-γv)*g
s[:] = γs*s + (1-γs)*g.*g
M.k = k += 1
v_hat = v ./ (1 - γv^k)
s_hat = s ./ (1 - γs^k)
return x - α*v_hat ./ (sqrt.(s_hat) .+ ϵ)
end
=#

# === Hyper Gradient ===
#=
mutable struct HyperGradientDescent <: DescentMethod
α0 # initial learning rate
μ # learning rate of the learning rate
α # current learning rate
g_prev # previous gradient
end
function init!(M::HyperGradientDescent, f, ∇f, x)
M.α = M.α0
M.g_prev = zeros(length(x))
return M
end
function step!(M::HyperGradientDescent, f, ∇f, x)
α, μ, g, g_prev = M.α, M.μ, ∇f(x), M.g_prev
α = α + μ*(g⋅g_prev)
M.g_prev, M.α = g, α
return x - α*g
end
=#


# === Hyper Gradient Nesterov ===
#=
mutable struct HyperNesterovMomentum <: DescentMethod
α0 # initial learning rate
μ # learning rate of the learning rate
β # momentum decay
v # momentum
α # current learning rate
g_prev # previous gradient
end
function init!(M::HyperNesterovMomentum, f, ∇f, x)
M.α = M.α0
M.v = zeros(length(x))
M.g_prev = zeros(length(x))
return M
end
function step!(M::HyperNesterovMomentum, f, ∇f, x)
α, β, μ = M.α, M.β, M.μ
v, g, g_prev = M.v, ∇f(x), M.g_prev
α = α - μ*(g⋅(-g_prev - β*v))
v[:] = β*v + g
M.g_prev, M.α = g, α
return x - α*(g + β*v)
end
=#

# ==== BFGS ====
#TODO: Not working ... normalization of gradients seems to work in some cases but gets very unstable,
#TODO: Bad results.

mutable struct BFGS <: DescentMethod
    Q
    α::Real
    α_min::Real
end

function init!(M::BFGS, x)
    m = length(x)
    M.Q = Matrix(1.0I, m, m)
    return M
end
function init!(M::BFGS, f, ∇f, x)
    m = length(x)
    M.Q = Matrix(1.0I, m, m)
    return M
end

function direction!(M::BFGS, x,f,∇f,params::SAHvars;verbose=false)
    g = ∇f(x,params)

    g = g/(norm(g) + params.ϵ_dJ) # gradient normalization
println(" BFGS: |g|_2 = ",norm(g))

    println("|Q|=",norm(M.Q))
    x1 = line_search(f, x, -M.Q*g, params) # x´ = x + α*d with optimal α #! error in bracket_minimum

    #=
    ∇J = M.Q*g
    if verbose
        println(" 00 BFGS: |dJ|_2 = ",norm(∇J))
    end 
    x1 = x - M.α*∇J =#

    g1 = ∇f(x1,params)
    δ = x1 - x
    println("norm delta=",norm(δ))
    γ = g1/(norm(g1) + params.ϵ_dJ) - g/(norm(g) + params.ϵ_dJ) #! gamma == 0 !!  needs a different α from line_search!!!!
    println("norm gamma=",norm(γ))
    M.Q[:] = M.Q - (δ*γ'*M.Q + M.Q*γ*δ')/(δ'*γ) + (1 + (γ'*M.Q*γ)/(δ'*γ))[1]*(δ*δ')/(δ'*γ)
println("|Q|=",norm(M.Q))

    ∇J = M.Q*g1

    if verbose
        println(" BFGS: |dJ|_2 = ",norm(∇J))
    end
    
    # optimization direction (need control over α)
    ∂J = ∇J/(norm(∇J) + params.ϵ_dJ)

    #return δ
    return -M.α*∂J
end





# ==== L-BFGS ====
#=
mutable struct LimitedMemoryBFGS <: DescentMethod
m
δs
γs
qs
end
function init!(M::LimitedMemoryBFGS, f, ∇f, x)
M.δs = []
M.γs = []
M.qs = []
return M
end
function step!(M::LimitedMemoryBFGS, f, ∇f, x)
δs, γs, qs, g = M.δs, M.γs, M.qs, ∇f(x)
m = length(δs)
if m > 0
q = g
for i in m : -1 : 1
qs[i] = copy(q)
q -= (δs[i]⋅q)/(γs[i]⋅δs[i])*γs[i]
end
z = (γs[m] .* δs[m] .* q) / (γs[m]⋅γs[m])
for i in 1 : m
z += δs[i]*(δs[i]⋅qs[i] - γs[i]⋅z)/(γs[i]⋅δs[i])
end
x′ = line_search(f, x, -z)
else
x′ = line_search(f, x, -g)
end
g′ = ∇f(x′)
push!(δs, x′ - x); push!(γs, g′ - g)
push!(qs, zeros(length(x)))
while length(δs) > M.m
popfirst!(δs); popfirst!(γs); popfirst!(qs)
end
return x′
end
=#
