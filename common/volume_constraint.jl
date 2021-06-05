# Control of Volume measure
#
# Dependencies: Gridap.jl
# 
# Joaquin Mura. May, 2021
#

using Gridap

"""
Volume Adjust for SAH using dichotomy
"""
function volume_adjust(v::Vector,δv::Vector,target::Real,dΩ::Measure;min_v::Real=0.0,max_v::Real=1.0,verbose::Bool=false)
    
    update(v_,δ,λ) = max.(min_v,min.(max_v,v_ + δ .- λ))

    #TODO 1: Too slow near the optimum: maybe δv should not be normalized? Λ goes to -4000 easily...nonsense exceot for too much moving away the optimum 
    #TODO 2: Add penalty iteration trigger.

    Λ = 0.01 # ab-initio
    v_new = update(v,δv,Λ)

    
    while norm(v_new) == 0.0
        Λ *= -0.8*Λ
        v_new = update(v,δv,Λ)
    end

    vol = sum(∫(v_new)dΩ)
    vtg = sum(∫(target)dΩ)
    
    δΛ = 1e-3
    Λ_min = 10000.0
    Λ_max = -10000.0
    iter_ = 0
    iter_max = 200
    rel_tol = 2e-3

    # Set lagrange multiplier bounds
    if vol<vtg
        Λ_min = Λ
        vol_new = vol
        iter_=0
        while (vol_new < vtg) & (iter_ < 5*iter_max)
            Λ -= δΛ
            v_new = update(v_new,δv,Λ)
            # do penalty if apply: theta = ( 1 - cos(pi*theta) )/2 once
            vol_new = sum(∫(v_new)dΩ)
            iter_ += 1
        end
        Λ_max = Λ
    elseif vol>vtg

        Λ_max = Λ
        vol_new = vol
        iter_=0
        while (vol_new > vtg) & (iter_ < 5*iter_max)
            Λ += δΛ
            v_new = update(v_new,δv,Λ)
            # do penalty if apply
            vol_new = sum(∫(v_new)dΩ)
            iter_ += 1
        end
        Λ_min = Λ
    else
        Λ_max = Λ
        Λ_min = Λ
    end
    vol = sum(∫(v_new)dΩ)

    # Fit Λ
    iter_=0
    while (abs(vol-vtg)>rel_tol*vtg) & (iter_ < iter_max)
        Λ = (Λ_min + Λ_max)/2
        v_new = update(v,δv,Λ) # moves from user's input
        # Do penalty ...

        global vol_new = sum(∫(v_new)dΩ)
        iter_ += 1
        if vol_new < vtg
            Λ_min = Λ
        elseif vol_new > vtg
            Λ_max = Λ
        end
    end

    # relative difference with input (as vectors)
    err_rel = norm(v - v_new)/norm(v)

    if verbose
        printfmt(" > vol.fit: num.iter={:d}/{:d} | err.rel={:.5f} | Λ={:.5f} | Λ(min,max)=({:.5f},{:.5f})\n",
            iter_,iter_max,err_rel,Λ,Λ_min,Λ_max)
    end

    return v_new,vol_new,err_rel # the end
end