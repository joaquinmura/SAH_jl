######################################################################
# Small-Amplitude Homogenization in Julia
# Linear Diffusion case
#
# Based on "_Optimal Design in Small Amplitude Homogenization_"
# G. Allaire and S. Gutiérrez, _ESAIM: Mathematical Modelling_ 
# _and Numerical Analysis_, pp543-574(41:3), 2007.
#
# Joaquin Mura 
# joaquin.mura_at_usm.cl
# May, 2021
######################################################################

using Base: Real

#TODO module SAH

#TODO:-1. clean code
#TODO:-0.5 get Nesterov momentum update  ------------------------  DONE √
#TODO: 0. create repo in Gitlab or Github -----------------------  DONE √
#TODO: 1. improve documentation
#TODO: 2. add output with objective function (saved step by step)  DONE √
#TODO: 3. calculate adjoints ------------------------------------  DONE √
#TODO: 4. add penalty
#TODO: 5. save microstructure if any ----------------------------  DONE √

using Gridap, Formatting

#TODO: module?
include("../common/sah_definitions.jl")
include("../common/volume_constraint.jl")
include("../common/optimization_methods.jl")

# Note: 
# We have A₁ = (1+η)A₀ for Aₙ(x)=(1-χₙ(x))A₀+χₙ(x)A₁=(1+ηχₙ(x))A₀
# with  η ∈ (-1,1), which implies
# If η > 0:  material 1 is the best conductor
#      θ = 0  -->  A = A₀
#      θ = 1  -->  A = A₁
#    and |A₁| > |A₀|. When η=0.9 ==> A₁ = 1.9 A₀
#    
# If η < 0:  material 1 is the worst conductor
#      θ = 0  -->  A = A₀
#      θ = 1  -->  A = A₁
#    but |A₁| < |A₀| ... indeed: A₀=1/(1+η)A₁ (η=-0.9 ==> A₀ = 10A₁ or A₁ = 0.1 A₀)

"""
TODO: Add usage and example

"""
function solve_sah_diffusion(geomodel,labels::Tuple,η::Real,A₀,θ,volume::Real,
    f::Function,g::Function,ud::Function,Jf::Vector,Jg::Vector;
    save_VTK::Bool=false,is_autoadjoint::Bool=false,max_iter::Int=5,degree::Int=1,
    output_path::String="",output_prefix::String="SAHdiff",verbose::Bool=false,use_microstructure::Bool=true)
    #

    println(" --- SAH Diffusion problem ---")
    #* == Initialization
    dirichlet_tags,neuman_tags = labels
    α     = 50
    α_min = 1e-5
    J     = 0.0   # definition to make it global
    Jold  = 1e+30
    ϵ_dJ  = 1e-15 # for gradient normalization: avoid division by zero
    fobj  = []
    accepted_iterations = []
    #mod_after_iter = 10 # after this, save vtk files each mod_iter iterations
    #mod_iter       = 10
    tol_error_rel = 1e-6  # limit tolerance for consecutive error in θ
    acc_error_rel = 2e-3  # when to increase α more agressively if step is accepted

    
    pre_filename = output_prefix;
    if !isempty(output_path)
        mkpath(output_path) # allows overwrite
        global pre_filename = output_path*Base.Filesystem.pathsep()*output_prefix;
    end

    if ismissing(dirichlet_tags) || isempty(dirichlet_tags)
        error("parameter dirichlet_tags must be declared as labels tuple.")
    end

    #* ==== MESH ===
    println(" * Mesh setting")
    # Triangulation from geometrical model
    Ω = Triangulation(geomodel)
    dΩ = Measure(Ω,degree)

    Γ = BoundaryTriangulation(geomodel)
    g0 = 0.0 # used it to erase the action of 'g'
    if !isempty(neuman_tags)
        global Γ, g0
        Γ = BoundaryTriangulation(geomodel,tags=neuman_tags)
        g0 = 1.0 
    end
    dΓ= Measure(Γ,degree)

    # Finite Element Spaces
    reffe = ReferenceFE(lagrangian,Float64,degree)
    V   = TestFESpace(geomodel,reffe;conformity=:H1, dirichlet_tags=dirichlet_tags)
    Ud  = TrialFESpace(V,ud)     # solution space with non homogeneous Dirichlet BC's
    Ud0 = TrialFESpace(V,x->0.0) # solution space with homogeneous Dirichlet BC's

    # use center coordinates to Evaluate and convert to array (compatible with θ)
    xy = get_cell_points(Ω) # all vertices in cell
    xc = [sum(xy.cell_phys_point[i])/length(xy.cell_phys_point[i]) for i in 1:length(xy.cell_phys_point)] # center at cell

    

    #* === Optimization ===

    ## If select gradient_descent method: #? maybe this should be placed somewhere else
    OptimMethod = GradientDescent(α,α_min) # 43 iter # very good indeed
    #OptimMethod = Momentum(α,0.1,[0],α_min) # 38 iter # very good one with β=0.1
    #OptimMethod = NesterovMomentum(α,0.1,[0],α_min) # β=α/2  # 38 with beta=0.1 16 iter ... but solution not good enough, like missing iterations, why?
    #OptimMethod = BFGS(0,α,α_min) #! not working with volume !!?? <-- possibly step size too small

    # Method initialization
    init!(OptimMethod,θ)


    # First Iteration
    println(" * Initial step")
    # Check volume fraction
    θ,area,error_rel = volume_adjust(θ,zeros(size(θ)),volume,dΩ)

    # General Bilinear from
    a(u,v) = ∫( A₀⋅∇(u) ⋅ ∇(v) )dΩ
    l0(v)  = ∫(v*f)dΩ + ∫( v * g0*g )dΓ

    op = AffineFEOperator(a,l0,Ud,V)
    K  = get_matrix(op)
    F0 = get_vector(op)

    # Get the solution: Direct problem 0
    U0h = K\F0; 
    u₀ = FEFunction(Ud,U0h)

    # Get the solution: Direct problem 1
    l1(v) = ∫( θ⋅(A₀⋅∇(u₀)⋅∇(v)))dΩ
    F1 = assemble_vector(l1,V)
    U1h = K\F1
    u₁ = FEFunction(Ud,U1h)

    if is_autoadjoint
        println("  Note: Using 'is_autoadjoint = true'")
        global p₀,p₁
        p₀ = u₀
        p₁ = u₁
        lp1(v) = ∫(0.0*v)dΩ  #! THIS ...
    else
        println("   solving adjoint problems.")
        global p₀,p₁
        # -div(A₀∇p₀) = j₁'(u₀) in Ω
        #          p₀ = 0       on Γ_D
        #     A₀∇p₀⋅n = j₂'(u₀) on Γ_N
        
        lp0(v)  = ∫(Jf[2](u₀,f)*v)dΩ + ∫( Jg[2](u₀,g) * v )dΓ 
        F0p = assemble_vector(lp0,V)
        P0h = K\F0p
        p₀  = FEFunction(Ud0,P0h)


        # -div(A₀∇p₁) = j₁"(u₀)u₁ + div(θA₀∇p₀)  in Ω
        #          p₁ = 0                        on Γ_D
        #     A₀∇p₁⋅n = j₂"(u₀)u₁ - θA₀∇p₀⋅n     on Γ_N

        #! AND THIS, PRODUCES A WARNING: WARNING: Method definition 
        lp1(v)  = ∫(Jf[3](u₀,f)*u₁*v)dΩ + ∫( Jg[3](u₀,g)*u₁ * v )dΓ - ∫( θ⋅(A₀⋅∇(p₀)⋅∇(v)))dΩ
        F1p = assemble_vector(lp1,V)
        P1h = K\F1p
        p₁  = FEFunction(Ud0,P1h)
    end

    # Calculate optimal microstructure
    A∇u₀_ = evaluate(A₀⋅∇(u₀),xc); 
    A∇p₀_ = evaluate(A₀⋅∇(p₀),xc); 

    if use_microstructure
        println(" * Calculating microstructural term")
        # M(ξ) = ξ⊗ξ / A₀ξ⋅ξ
        # Φ_opt = min_ξ [ M(ξ)A₀∇u₀⋅A₀∇p₀ ]
        dim = num_dims(Ω)
        A0 = reshape([A₀.data...],(dim,dim)) # A₀ is TensorValue !, A0 is Matrix
        global ϕ_opt = fill(1e30,num_cells(Ω));
        global β_opt = zeros(num_cells(Ω))
        if dim==2
            A∇u₀=[VectorValue(sum(A∇u₀_[i][1]),sum(A∇u₀_[i][2]))/length(A∇u₀_[i]) for i in 1:num_cells(Ω)];
            #A∇p₀=[sum(A∇p₀_[i])/length(A∇p₀_[i]) for i in 1:num_cells(Ω)];
            A∇p₀=[VectorValue(sum(A∇p₀_[i][1]),sum(A∇p₀_[i][2]))/length(A∇p₀_[i]) for i in 1:num_cells(Ω)];

            for e in 1:num_cells(Ω)
                for β in range(0,stop=π,length=1000)

                    ξ = [cos(β),sin(β)]
                    ϕ = (A∇u₀[e]⋅ξ )*(A∇p₀[e]⋅ξ) / ((A0*ξ)⋅ξ)

                    if ϕ < ϕ_opt[e] 
                        ϕ_opt[e] = ϕ
                        β_opt[e] = β
                    end
                end
            end
        else
            # TODO 3D
        end

        printfmt("    min ϕ: {:4.2e},\t  max ϕ: {:4.2e}\n",min(ϕ_opt...),max(ϕ_opt...))
        printfmt("    min β: {:4.3f},\t  max β: {:4.3f}\n",min(β_opt...),max(β_opt...))
    else
        # No microstructural term: Applies for Compliance
        println(" - Warning: No microstructure here")
        global ϕ_opt = zeros(num_cells(Ω))
        global β_opt = zeros(num_cells(Ω))
    end

    # Calculate independent terms from θ
    #* Recalculate J = J(u₀,u₁,p₀,p₁,θ) = J₀ + J₁(θ,u₁)
    J₀ =  sum(∫(Jf[1](u₀,f))dΩ)
    

    # Gradient: ∂J/∂θ(s) = ∫ dJsa(u₀,u₁,p₀,p₁)⋅s dΩ
    #dJsa(u₀,u₁,p₀,p₁) = -η*(A₀⋅∇(u₀)⋅∇(p₀)) - η^2 * ( (A₀⋅∇(u₁)⋅∇(p₀)) - (A₀⋅∇(u₀)⋅∇(p₁)) )  # + Microstructure
    #dJsa = -η*(A₀⋅∇(u₀)⋅∇(p₀)) - η^2 * ( (A₀⋅∇(u₁)⋅∇(p₀)) - (A₀⋅∇(u₀)⋅∇(p₁)) )  # + Microstructure
    #aux = evaluate(dJsa,xy)
    #dJ = [sum(aux[i])/length(aux[i]) for i in 1:length(θ)];
    dJsa1 = A₀⋅∇(u₀)⋅∇(p₀) # once for all the iterations
    dJsa2 = A₀⋅∇(u₁)⋅∇(p₀)
    dJsa3 = A₀⋅∇(u₀)⋅∇(p₁) 
    dJs1 = evaluate(dJsa1,xy) 
    dJs2 = evaluate(dJsa2,xy)
    dJs3 = evaluate(dJsa3,xy)
    dJ1 = [sum(dJs1[i])/length(dJs1[i]) for i in 1:length(θ)];
    dJ2 = [sum(dJs2[i])/length(dJs2[i]) for i in 1:length(θ)];
    dJ3 = [sum(dJs3[i])/length(dJs3[i]) for i in 1:length(θ)];
    dJ = -η*dJ1 - η^2*(dJ2 + dJ3) 
    dJm = zeros(size(dJ))
    if use_microstructure
        dJm = (1.0.-2.0.*θ).*ϕ_opt # Array! Microstructure
        dJ += η^2*dJm
    end

    # Parameter refill
    #TODO: Improve or reorder !!!
    Parameters = SAHvars(u₀,p₀,u₁,p₁,K,l1,lp1,J₀,η*dJ1,A₀,η,
        ϕ_opt,ϵ_dJ,Jf,Jg,dΩ,
        is_autoadjoint,use_microstructure,V,Ud,Ud0,xy,f)

    # First gradient (normalized)
    ∂J = dJ/(norm(dJ) + ϵ_dJ) # quasi-normalized!
    δx = -OptimMethod.α*∂J # descent direction (first step)

    if save_VTK
        # save first iteration
        filename  = pre_filename*"_0000"
        filemicro = pre_filename*"_micro"
        if verbose
            print(" solution and microstructure saved in $(filename).vtu")
            if use_microstructure
                print(" and $(filemicro).vtu, respectively")
            end
            print(" ... ")
        end
        
        if !is_autoadjoint
            writevtk(Ω,filename,cellfields=["theta"=>θ,"U0"=>u₀,"U1"=>u₁,"Micro"=>ϕ_opt,"angle"=>β_opt,"sensitivity"=>-dJ])
        else
            writevtk(Ω,filename,cellfields=["theta"=>θ,"U0"=>u₀,"P0"=>p₀,"U1"=>u₁,"P1"=>p₁,"Micro"=>ϕ_opt,"angle"=>β_opt,"sensitivity"=>-dJ])
        end
        if use_microstructure
                writevtk(Ω,filemicro,cellfields=["phi_micro"=>ϕ_opt,"angle_micro"=>β_opt])
        end

        if verbose
            print("done.\n")
        end
    end



    ##* === Main Loop
    for iter in 1:max_iter
        println("\n===== iter $(iter)/$max_iter  =====")
        
        #* update θ: constraint
        θ_old = θ
        θ,area,error_rel_vol = volume_adjust(θ,δx,volume,dΩ,verbose=verbose)
        if verbose
            printfmt(" volume = {:2.4f}",area)
        end
        error_rel = norm(θ - θ_old)/norm(θ_old)

        # ----- here given θ, u₀,p₀,u₁,p₁,ϕ_opt,η,volume
        J  = get_J!(θ,Parameters)
        
        #* Check descent
        printfmt(" || Obj.Func: old={:.6e}, new={:.6e}\n",Jold,J)
        relax_factor = (1.0 + 0.05/iter^2)

        if J < Jold * relax_factor
            Jold  = J
            OptimMethod.α *= 1.05 # acceleration factor

            if error_rel < acc_error_rel
                # agressive acceleration when iterations does not improve solution
                OptimMethod.α *= 1.2 
            end

            # save data
            append!(fobj,J)
            append!(accepted_iterations,iter)

            if save_VTK
                snumber = fmt(FormatSpec("0>4s"),iter) # 4 digits with 0 at left
                filename  = pre_filename*"_"*snumber
                if verbose
                    print(" solution saved in $(filename).vtu ...")
                end
                #TODO Add p0,p1 if corresponds
                # note: no need of u₀
                writevtk(Ω,filename,cellfields=["theta"=>θ,"U1"=>Parameters.u1,"sensitivity"=>δx])
                if verbose
                    print("done.\n")
                end
            end

            # choose algorithm according 'OptimMethod'
            δx = direction!(OptimMethod,θ,get_J!,get_∇J,Parameters,verbose=true)

            #=
            # re-calculate ∂J/∂θ
            dJ = get_∇J(θ,Parameters)
        
            if verbose
                println(" |dJ|_2 = ",norm(dJ))
            end

            # optimization direction (need control over α)
            ∂J = dJ/(norm(dJ) + Parameters.ϵ_dJ)
            gradJ = α*∂J
            #TODO modify gradJ to improve optimization ... BFGS? nonlinear CG?
            =#

            if error_rel < tol_error_rel
                #? STOP if convergence
                println("# STOP: Convergence achived after $(iter) iterations: rel.error=",error_rel)
                break;
            end

        else
            # Keep same direction ∂J, but reduce the step until tolerance
            OptimMethod.α *= 0.75 # deacceleration factor


            println(" iteration $(iter) not accepted: step size reduced to ",OptimMethod.α)

            if (OptimMethod.α < OptimMethod.α_min) || (error_rel < tol_error_rel)
                #! STOP
                println("# STOP: No convergence found after $(iter) iterations: α=",OptimMethod.α,
                    "<α_min=",OptimMethod.α_min," rel.error=",error_rel)
                break;
            end
        end

    end # loop
    
    return θ,fobj,accepted_iterations
end





function get_J!(θ,Params::SAHvars)

    #? really needs to copy variables?
    u₀,p₀,u₁,A₀ = Params.u0,Params.p0,Params.u1,Params.A0
    K,l1,Jf,J₀,η,dΩ = Params.K,Params.l1,Params.Jf,Params.J0,Params.η,Params.dΩ
    V,Ud,Ud0 = Params.V,Params.Vd,Params.Vd0
    is_autoadjoint = Params.is_autoadjoint
    f = Params.f

    # Recalculate new u1 and p1
    F1 = assemble_vector(l1,V)
    U1h = K\F1
    u₁ = FEFunction(Ud,U1h)
    p₁ = u₁
    if !is_autoadjoint
        F1p = assemble_vector(Params.lp1,V)
        P1h = K\F1p
        p₁  = FEFunction(Ud0,P1h)
    end
    # update u1,p1
    Params.u1 = u₁
    Params.p1 = p₁

    #* Recalculate J = J(u₀,u₁,p₀,p₁,θ) = J₀ + J₁
    #  Jf = [j₁,∂j₁,∂²j₁]
    #? update_state! ???
    J₁ = η * sum(∫(  θ⋅((A₀⋅∇(u₀))⋅∇(p₀)) )dΩ) 
    J₁ += (0.5*η^2) * sum(∫(Jf[3](u₀,f)⋅(u₁⋅u₁))dΩ)
    J₁ -= (η^2) * sum(∫(  θ⋅((A₀⋅∇(u₁))⋅∇(p₀)) )dΩ)
    if Params.use_microstructure
        J₁ += η^2*sum(∫(θ.*(1.0.-θ).*Params.micro)dΩ ) #? Microstructure
    end

    return J₀+J₁
end


function get_∇J(θ,Params::SAHvars)
    u₀,p₀,u₁,p₁,A₀ = Params.u0,Params.p0,Params.u1,Params.p1,Params.A0
    η = Params.η
    xy,ηdJ1 = Params.xyz,Params.ηdJ1


    
    # re-calculate ∂J/∂θ
    dJsa2 = A₀⋅∇(u₁)⋅∇(p₀)
    dJsa3 = A₀⋅∇(u₀)⋅∇(p₁) 
    dJs2 = evaluate(dJsa2,xy)
    dJs3 = evaluate(dJsa3,xy)
    dJ2 = [sum(dJs2[i])/length(dJs2[i]) for i in 1:length(θ)];
    dJ3 = [sum(dJs3[i])/length(dJs3[i]) for i in 1:length(θ)];
    dJ = -ηdJ1 - η^2*(dJ2 + dJ3)

    if Params.use_microstructure
        dJ += η^2*(1.0.-2.0.*θ).*Params.micro # Array! Microstructure
    end

    return dJ
end
