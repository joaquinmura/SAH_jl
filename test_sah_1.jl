# TEST SAH: 1
# 
# Replicates example 1 : Diffusion compliance with η=-0.5 and volume_fraction=0.5
# from "Optimal design in small amplitude homogenization", G.Allaire & S.Gutiérrez, ESAIM 2007


using Gridap

include("sah/sah_diffusion.jl")

# Generate the domain
n=100 #60 .. #! 100 ok but 110 stop at iter=3!!!! why?
domain = (0,1,0,1)
partition = (n,n)
geomodel = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(geomodel)
add_tag_from_tags!(labels,"dir1",[3,4,6]) # top
add_tag_from_tags!(labels,"dir0",[1,2,5,7,8]) # sides and bottom

# Triangulation from geometrical model
Ω = Triangulation(geomodel)
degree = 2
dΩ = Measure(Ω,degree)

# Physical parameters
K₀ = 1000*TensorValue(1.0,0.0,0.0,1.0)

vol = 0.5  # volume fraction of material 1 (better conductor if η>0, worst otherwise)

eta = -0.5 #0.1 # contrast factor (between -1 and 1)

theta = vol*ones(num_cells(Ω)) # rand(num_cells(Ω))


area = sum(∫(theta)dΩ)
println("Area(θ)=",area,"  area_equi=",sum(theta)/n^2) # coincide if dx=dy

# test 1
f(x) = 1.0 
g(x) = 0.0 # not used in this example

# Evaluate the objective function
j1(u,f)= f⋅u  # compliance *maximization* with '-'
dj1(u,f)= f
ddj1(u,f)=0.0
j2(u,f)=0.0  # not used here
dj2(u,f)=0.0
ddj2(u,f)=0.0
autoadj = true
micro_ = false

Jf = [j1,dj1,ddj1]
Jg = [j2,dj2,ddj2]

# Run!
theta_opt, J, iter = solve_sah_diffusion(geomodel,(["dir1","dir0"],""),eta,K₀,theta,vol,f,g,x->0.0,
    Jf,Jg;degree=degree,save_VTK=true,is_autoadjoint=autoadj,max_iter=100,
    output_path="out",use_microstructure=micro_,verbose=true);

# Show a plot with convergence history
display(plot(iter,J,w=3))
