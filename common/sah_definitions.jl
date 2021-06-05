# Definitions file
# Small-Amplitude Homogenization
#
# Joaquin Mura, May 2021.


mutable struct SAHvars
    u0::FEFunction
    p0::FEFunction
    u1::FEFunction
    p1::FEFunction
    K       # stiffness matrix 
    l1      # rhs functional for u1
    lp1     # rhs functional for p1
    J0::Real # 
    ηdJ1    # gradient (Array) with u0,p0 contribution * η
    A0      # Conductivity matrix (constant)
    η::Real # contrast
    micro   # term with microstructural contribution * η²
    ϵ_dJ::Real # lower limit factor for gradient normalization
    Jf::Vector # Jf = [j₁,∂j₁,∂²j₁] : volume obj functions
    Jg::Vector # Jg = [j₂,∂j₂,∂²j₂] : surface obj functions
    dΩ::Measure
    #dΓ::Measure
    is_autoadjoint::Bool
    use_microstructure::Bool
    V::FESpace  # Test FE space
    Vd::FESpace # Trial FE space (with Dirichlet BCs)
    Vd0::FESpace # Trial FE space with homogeneous Dirichlet BC
    xyz         # set of evaluation points
    f  # rhs function in PDE (for J'' evaluation)
end