using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using GridapSolvers, GridapSolvers.BlockSolvers
using FiniteDiff

n = 25             # mesh size
max_steps = 10     # Time-steps for evolution equation
vf = 0.15          # Volume fraction
α_coeff = 2        # Regularisation coefficient extension-regularisation
iter_mod = 1       # Write output every iter_mod iterations
order = 1

# Model and some refinement
xmax,ymax=(2.0,1.0)
prop_Γ_N = 0.05
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,xmax,0,ymax),(2n,n)))
ref_model = refine(base_model, refinement_method = "barycentric")
ref_model = refine(ref_model)
ref_model = refine(ref_model)
model = get_model(ref_model)
h = minimum(get_element_diameters(model))
f_Γ_D(x) = (x[1] ≈ 0.0)
f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
writevtk(model,path*"model")

## Level-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_regs = [TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"]) for _ in 1:2]
V_reg = MultiFieldFESpace(V_regs;style=BlockMultiFieldStyle())
U_reg = MultiFieldFESpace(TrialFESpace.(V_reg.spaces);style=BlockMultiFieldStyle())
V_φ = MultiFieldFESpace([TestFESpace(model,reffe_scalar) for _ in 1:2])

## Level-set function
f((x,y),a,b) = -cos(4π*(x+b*x))cos(4π*y) - a
# φh = interpolate([x->f(x,0.5,-0.1),x->f(x,0.5,0.0)],V_φ)
φh = interpolate([x->f(x,0.3,0),x->f(x,0.6,0.0)],V_φ)

# Check LS
GridapTopOpt.correct_ls!(φh)

## Triangulations and measures
Ω_bg = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ_bg = Measure(Ω_bg,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ_bg)

reinits = [StabilisedReinitialiser(V_φ[i],dΩ_bg,h;stabilisation_method=ArtificialViscosity(0.75)) for i in 1:2]
evos = [CutFEMEvolver(V_φ[i],dΩ_bg,h;max_steps,γg=0.1) for i in 1:2]
ls_evo = MultiLevelSetEvolution(evos,reinits,V_φ;reuse_cache=true);
reinit!(ls_evo,φh)

function compute_geo(φh1,φh2)
  geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
  geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
  setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
  setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
  intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
  outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
  geo = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
  return cut(PolytopalLevelSetCutter(),model,geo)
end

Ωs = EmbeddedCollection(model,φh;compute_cut=false) do φh
  φh1, φh2 = φh
  cutgeo = compute_geo(φh1,φh2)
  # Physical triangulations
  Ω2 = DifferentiableTriangulation(cutgeo,"Ω2")
  Ω3 = DifferentiableTriangulation(cutgeo,"Ω3")
  Γ23 = DifferentiableEmbeddedBoundary(cutgeo,"Ω2","Ω3")
  # Ghost triangulations
  Γg2 = GhostSkeleton(cutgeo,"Ω2")
  Γg3 = GhostSkeleton(cutgeo,"Ω3")
  # Active triangulations
  Ω2act = Triangulation(cutgeo,ACTIVE,"Ω2")
  Ω3act = Triangulation(cutgeo,ACTIVE,"Ω3")
  # Isolated volumes
  φ1 = get_free_dof_values(φh1)
  φ2 = get_free_dof_values(φh2)
  φ_Ω2orΩ3 = min.(max.(-φ1,φ2),max.(φ1,φ2));
  φ_Ω2orΩ3_cv = get_cell_dof_values(FEFunction(V_φ[1],φ_Ω2orΩ3))
  χ,_ = get_isolated_volumes_mask_polytopal(model,φ_Ω2orΩ3_cv,["Gamma_D"])
  (;
    :Ω2  => Ω2, :Ω2act => Ω2act, :dΩ2 => Measure(Ω2,2*order),
    :Ω3  => Ω3, :Ω3act => Ω3act, :dΩ3 => Measure(Ω3,2*order),
    :Γ23 => Γ23, :dΓ23 => Measure(Γ23,2*order), :n_Γ23 => get_normal_vector(Γ23),
    :Γg2 => Γg2, :dΓg2 => Measure(Γg2,2*order), :n_Γg2 => get_normal_vector(Γg2),
    :Γg3 => Γg3, :dΓg3 => Measure(Γg3,2*order), :n_Γg3 => get_normal_vector(Γg3),
    :χ => χ
  )
end

## Weak form
# Material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
E1 = 1.0; E2 = 0.5;
λs1, μs1 = lame_parameters(E1,0.3)
λs2, μs2 = lame_parameters(E2,0.3)
σ1(ε) = λs1*tr(ε)*one(ε) + 2*μs1*ε
σ2(ε) = λs2*tr(ε)*one(ε) + 2*μs2*ε
# Stabilization
α_Gd = 1e-7
γ_Gd1 = α_Gd*(λs1 + μs1)*h^3
γ_Gd2 = α_Gd*(λs2 + μs2)*h^3
# Jumps
λ = 10^2/h*max(λs1 + μs1, λs2 + μs2)
w1 = E2/(E1 + E2)
w2 = E1/(E1 + E2)
jump_u(u1,u2) = u1 - u2
mean_t(u1,u2) = w1*(σ1 ∘ ε(u1)) + w2*(σ2 ∘ ε(u2))

function a((u1,u2),(v1,v2),(φh1,φh2))
  # Compute normal
  n_Γ23 = get_normal_vector(Ωs.Γ23)
  n_Γg2 = Ωs.n_Γg2; n_Γg3 = Ωs.n_Γg3
  return ∫( ε(v1) ⊙ (σ1∘ε(u1)) )Ωs.dΩ2 + ∫( ε(v2) ⊙ (σ2∘ε(u2)) )Ωs.dΩ3 +
    ∫( λ*jump_u(v1,v2)⋅jump_u(u1,u2)
      - n_Γ23⋅mean_t(u1,u2)⋅jump_u(v1,v2)
      - n_Γ23⋅mean_t(v1,v2)⋅jump_u(u1,u2) )Ωs.dΓ23 +
    ∫( γ_Gd1*jump(n_Γg2⋅∇(v1))⋅jump(n_Γg2⋅∇(u1)) )Ωs.dΓg2 +
    ∫( γ_Gd2*jump(n_Γg3⋅∇(v2))⋅jump(n_Γg3⋅∇(u2)) )Ωs.dΓg3 +
    ∫(Ωs.χ*v1⋅u1)Ωs.dΩ2 + ∫(Ωs.χ*v2⋅u2)Ωs.dΩ3
end

g = VectorValue(0.0,-1.0);
l((v1,v2),(φh1,φh2)) = ∫(v1⋅g+v2⋅g)dΓ_N

## Optimisation functionals
J((u1,u2),φ) = ∫(ε(u1) ⊙ (σ1∘ε(u1)))Ωs.dΩ2 + ∫(ε(u2) ⊙ (σ2∘ε(u2)))Ωs.dΩ3
Vol_Ω2(u,φ) = ∫(1)Ωs.dΩ2
Vol_Ω3(u,φ) = ∫(1)Ωs.dΩ3

## FE operators
reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
state_collection = EmbeddedCollection(model,φh;compute_cut=false) do _φh
  update_collection!(Ωs,_φh)
  V1 = TestFESpace(Ωs.Ω2act,reffe_d;dirichlet_tags=["Gamma_D"])
  U1 = TrialFESpace(V1,zero(VectorValue{2,Float64}))
  V2 = TestFESpace(Ωs.Ω3act,reffe_d;dirichlet_tags=["Gamma_D"])
  U2 = TrialFESpace(V2,zero(VectorValue{2,Float64}))
  V = MultiFieldFESpace([V1,V2])
  U = MultiFieldFESpace([U1,U2])
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  (;
    :state_map => state_map,
    :J => StateParamMap(J,state_map),
    :C => map(Ci -> StateParamMap(Ci,state_map),[Vol_Ω2,Vol_Ω3])
  )
end

function φ_to_jc(φ)
  GridapTopOpt.ignore_derivatives() do
    update_collection!(state_collection,FEFunction(V_φ,φ))
  end
  u = state_collection.state_map(φ)
  j = state_collection.J(u,φ)
  c1 = state_collection.C[1](u,φ)/vol_D - vf
  c2 = state_collection.C[2](u,φ)/vol_D - vf
  return [j,c1,c2]
end

function dCi!(Ci,dC,φ)
  φh = FEFunction(V_φ,φ)
  _dC(q) = gradient(φ -> Ci(nothing,φ),φh)
  Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
end

pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,2,
    analytic_dC=[(dC,φ)->dCi!(Vol_Ω2,dC,φ), (dC,φ)->dCi!(Vol_Ω3,dC,φ)])

## Hilbertian extension-regularisation problems
α = α_coeff*h
a_hilb((p1,p2),(q1,q2)) = ∫(α^2*∇(p1)⋅∇(q1) + p1*q1 + α^2*∇(p2)⋅∇(q2) + p2*q2)dΩ_bg;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=BlockDiagonalSolver([LUSolver(),LUSolver()]))

## Optimiser
path = "results/cantilever_E1=1.0_E2=0.5/"
mkpath(path)
optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;verbose=true,constraint_names=[:Vol_Ω2,:Vol_Ω3],ls_γ_max=0.1)
for (it,_,φh) in optimiser
  if iszero(it % iter_mod)
    uh = get_state(state_collection.state_map)
    writevtk(Ω_bg,path*"Omega_$it",cellfields=[
      "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),"uh"=>uh[1],
      "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"uh"=>uh[2],"χ"=>Ωs.χ
    ])
    writevtk(Ωs.Ω2,path*"Omega2_$it",cellfields=["uh"=>uh[1]])
    writevtk(Ωs.Ω3,path*"Omega3_$it",cellfields=["uh"=>uh[2]])
    writevtk(Ωs.Γ23,path*"Gamma23_$it",cellfields=["jump"=>jump_u(uh[1],uh[2])])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(state_collection.state_map)
writevtk(Ω_bg,path*"Omega_$it",cellfields=[
  "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),"uh"=>uh[1],
  "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"uh"=>uh[2],"χ"=>Ωs.χ
])