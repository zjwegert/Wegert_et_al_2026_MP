using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapDistributed, GridapPETSc, PartitionedArrays
using PartitionedArrays
using GridapTopOpt

write_dir = ARGS[1]
nprocs = parse(Int,ARGS[2])
linesearch = ARGS[3]
γ = parse(Float64,ARGS[4])
alpha = parse(Float64,ARGS[5])
E1 = parse(Float64,ARGS[6])
E2 = parse(Float64,ARGS[7])
vf = parse(Float64,ARGS[8])
γg = parse(Float64,ARGS[9])

function main(model,geo_params,ranks,ls,γ,alpha,E1,E2,vf,γg,write_dir)
  # Output path
  path="$(write_dir)/MPI_FSI_LineSearch=$(ls)_γ=$(γ)_α=$(alpha)_E1=$(E1)_E2=$(E2)_vf=$(vf)_γg=$(γg)/"
  files_path = path*"data/"
  model_path = path*"model/"
  if i_am_main(ranks)
    mkpath(files_path); mkpath(model_path);
  end

  # Triangulation
  writevtk(model,model_path*"model")
  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Params
  max_steps = 1/hmin/10
  iter_mod = 5
  D = 2

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_regs = [TestFESpace(model,reffe_scalar) for _ in 1:2]
  V_reg = MultiFieldFESpace(V_regs)
  U_regs = [TrialFESpace(V_regs[i]) for i in 1:2]
  U_reg = MultiFieldFESpace(U_regs)
  V_φ = MultiFieldFESpace([TestFESpace(model,reffe_scalar) for _ in 1:2])
  assem_V_φ = SparseMatrixAssembler(V_φ,V_φ)

  _e = 1/3*hmin
  L,H,x0,l,w,a,b = geo_params
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+7_e),a*(1+7_e))
  fsolid(x) = min(f0(x,l+3/2*_e,b+3/2*_e),f0(x,w+3/2*_e,a+3/2*_e))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  lsf1(x) = max(fin(x),fholes(x,17,0.3))
  lsf2(x) = max(fin(x),fholes(x,17,0.5))
  φh = interpolate([lsf1,lsf2],V_φ)

  V_φ_wall = TestFESpace(model,reffe_scalar)
  φh_wall = interpolate(fsolid,V_φ_wall)
  GridapTopOpt.correct_ls!(φh_wall)
  geo_wall = DiscreteGeometryFromFEFunction(φh_wall,model,name="φ_wall")

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  ## Triangulations and measures
  order = 1
  degree = 2*order;
  dΩ_act = Measure(Ω_act,2order)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  dΓf_D = Measure(Γf_D,degree)
  vol_D = sum(∫(1)dΩ_act)

  evolve_ls = LUSolver()
  reinit_nls = NewtonSolver(LUSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  reinits = [StabilisedReinitialiser(V_φ[i],dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(0.75),nls=reinit_nls) for i in 1:2]
  evos = [CutFEMEvolver(V_φ[i],dΩ_act,hₕ;max_steps,γg,ode_ls=evolve_ls) for i in 1:2]
  ls_evo = MultiLevelSetEvolution(evos,reinits,V_φ;reuse_cache=true);
  reinit!(ls_evo,φh)

  function compute_geo(φh1,φh2)
    geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
    geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
    setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
    outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
    Ωf = setdiff(union(setdiff_geo1_geo2,outside_geo1_geo2),geo_wall,name="Ωf")
    Ωs1 = setdiff(setdiff(geo2,geo1),geo_wall,name="Ωs1")
    Ωs2 = union(intersect(geo1,geo2),geo_wall,name="Ωs2")
    geo = Ωf ∪ Ωs1 ∪ Ωs2
    return cut(model,geo), cut_facets(model,geo)
  end

  Ω = EmbeddedCollection(model,φh;compute_cut=false) do φh
    φh1, φh2 = φh
    cutgeo, cutgeo_facets = compute_geo(φh1,φh2)
    # Physical triangulations
    Ωs1 = DifferentiableTriangulation(cutgeo,"Ωs1")
    Ωs2 = DifferentiableTriangulation(cutgeo,"Ωs2")
    Ωf = DifferentiableTriangulation(cutgeo,"Ωf")
    Γ_s1s2 = DifferentiableEmbeddedBoundary(cutgeo,"Ωs1","Ωs2")
    Γ_s1f = DifferentiableEmbeddedBoundary(cutgeo,"Ωs1","Ωf")
    Γ_s2f = DifferentiableEmbeddedBoundary(cutgeo,"Ωs2","Ωf")
    # Ghost triangulations
    Γg_s1 = GhostSkeleton(cutgeo,"Ωs1")
    Γg_s2 = GhostSkeleton(cutgeo,"Ωs2")
    Γg_f = GhostSkeleton(cutgeo,"Ωf")
    # Active triangulations
    Ωs1_act = Triangulation(cutgeo,ACTIVE,"Ωs1")
    Ωs2_act = Triangulation(cutgeo,ACTIVE,"Ωs2")
    Ωf_act = Triangulation(cutgeo,ACTIVE,"Ωf")
    # Cut facets
    Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE,"Ωf")
    # Isolated volumes
    φ1 = get_free_dof_values(φh1)
    φ2 = get_free_dof_values(φh2)
    φ_wall = get_free_dof_values(φh_wall)
    φ_Ωs1orΩs2 = min.(max.(-φ1,φ2),max.(φ1,φ2),φ_wall);
    φ_Ωs1orΩs2_cv = map(get_cell_dof_values,local_views(FEFunction(V_φ[1],φ_Ωs1orΩs2)))
    ψ_s,_ = get_isolated_volumes_mask_polytopal(model,φ_Ωs1orΩs2_cv,["Gamma_s_D","Gamma_Bottom"])
    φ_Ωf = max.(min.(max.(-φ1,-φ2),max.(φ1,-φ2)),-φ_wall);
    φ_Ωf_cv = map(get_cell_dof_values,local_views(FEFunction(V_φ[1],φ_Ωf)))
    ψ_f,_ = get_isolated_volumes_mask_polytopal(model,φ_Ωf_cv,["Gamma_f_D",])
    (;
      :Ωf  => Ωf, :Ωf_act => Ωf_act, :dΩf => Measure(Ωf,degree),
      :Ωs1  => Ωs1, :Ωs1_act => Ωs1_act, :dΩs1 => Measure(Ωs1,degree),
      :Ωs2  => Ωs2, :Ωs2_act => Ωs2_act, :dΩs2 => Measure(Ωs2,degree),
      :Γ_s1s2 => Γ_s1s2, :dΓ_s1s2 => Measure(Γ_s1s2,degree), :n_Γ_s1s2 => get_normal_vector(Γ_s1s2),
      :Γ_s1f => Γ_s1f, :dΓ_s1f => Measure(Γ_s1f,degree), :n_Γ_s1f => get_normal_vector(Γ_s1f),
      :Γ_s2f => Γ_s2f, :dΓ_s2f => Measure(Γ_s2f,degree), :n_Γ_s2f => get_normal_vector(Γ_s2f),
      :Γi => Γi, :dΓi => Measure(Γi,degree), :n_Γi => get_normal_vector(Γi),
      :Γg_f => Γg_f, :dΓg_f => Measure(Γg_f,degree), :n_Γg_f => get_normal_vector(Γg_f),
      :Γg_s1 => Γg_s1, :dΓg_s1 => Measure(Γg_s1,degree), :n_Γg_s1 => get_normal_vector(Γg_s1),
      :Γg_s2 => Γg_s2, :dΓg_s2 => Measure(Γg_s2,degree), :n_Γg_s2 => get_normal_vector(Γg_s2),
      :ψ_s => ψ_s, :ψ_f => ψ_f
    )
  end

  ### Weak form
  ## Fluid
  uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)
  # Properties
  μ = 1

  # Stabilization parameters
  α_Nu = 100
  α_u  = 0.1
  α_p  = 0.25

  γ_Nu(h) = α_Nu*μ/h
  γ_u(h) = α_u*μ*h
  γ_p(h) = α_p*h/μ
  k_p    = 1.0 # (Villanueva and Maute, 2017)
  γ_Nu_h = γ_Nu ∘ hₕ
  γ_u_h = mean(γ_u ∘ hₕ)
  γ_p_h = mean(γ_p ∘ hₕ)

  # Terms
  _I = one(SymTensorValue{D,Float64})
  σf(u,p) = 2μ*ε(u) - p*_I
  a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
  b_Ω(div_v,p) = -p*(div_v)
  ab_Γ(u,∇u,v,∇v,p,q,n) = n ⋅ ( - μ*(∇u ⋅ v + ∇v ⋅ u) + v*p + u*q) + γ_Nu_h*(u⋅v)
  ju(∇u,∇v) = γ_u_h*(jump(Ω.n_Γg_f ⋅ ∇u) ⋅ jump(Ω.n_Γg_f ⋅ ∇v))
  jp(p,q) = γ_p_h*(jump(p) * jump(q))
  v_ψ(p,q) = k_p * Ω.ψ_f*p*q

  function a_fluid((u,p),(v,q),φ)
    ∇u = ∇(u); ∇v = ∇(v);
    div_u = ∇⋅u; div_v = ∇⋅v
    n_Γ_s1f = -get_normal_vector(Ω.Γ_s1f)
    n_Γ_s2f = -get_normal_vector(Ω.Γ_s2f)
    return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(ab_Γ(u,∇u,v,∇v,p,q,n_Γ_s1f))Ω.dΓ_s1f +
      ∫(ab_Γ(u,∇u,v,∇v,p,q,n_Γ_s2f))Ω.dΓ_s2f +
      ∫(ju(∇u,∇v))Ω.dΓg_f - ∫(jp(p,q))Ω.dΓi
  end

  l_fluid((v,q),φ) =  ∫(0q)Ω.dΩf

  ## Structure
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs1, μs1 = lame_parameters(E1,0.3)
  λs2, μs2 = lame_parameters(E2,0.3)
  σ1(ε) = λs1*tr(ε)*one(ε) + 2*μs1*ε
  σ2(ε) = λs2*tr(ε)*one(ε) + 2*μs2*ε
  # Stabilization
  α_Gd = 1e-3
  k_d = 1.0
  γ_Gd1(h) = α_Gd*(λs1 + μs1)*h^3
  γ_Gd2(h) = α_Gd*(λs2 + μs2)*h^3
  γ_Gd1_h = mean(γ_Gd1 ∘ hₕ)
  γ_Gd2_h = mean(γ_Gd2 ∘ hₕ)
  # Jumps
  _λ(h) = 10^2/h*max(λs1 + μs1, λs2 + μs2)
  λ = _λ ∘ hₕ
  w1 = E2/(E1 + E2)
  w2 = E1/(E1 + E2)
  jump_u(u1,u2) = u1 - u2
  mean_t(u1,u2) = w1*(σ1 ∘ ε(u1)) + w2*(σ2 ∘ ε(u2))
  # Terms
  a_s1_Ω(d,s) = ε(s) ⊙ (σ1 ∘ ε(d))
  a_s2_Ω(d,s) = ε(s) ⊙ (σ2 ∘ ε(d))
  j_s_k1(d,s) = γ_Gd1_h*(jump(Ω.n_Γg_s1 ⋅ ∇(s)) ⋅ jump(Ω.n_Γg_s1 ⋅ ∇(d)))
  j_s_k2(d,s) = γ_Gd2_h*(jump(Ω.n_Γg_s2 ⋅ ∇(s)) ⋅ jump(Ω.n_Γg_s2 ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω.ψ_s)*(d⋅s) # Isolated volume term
  j_s_Γ((d1,d2),(s1,s2),n_s1s2) = λ*jump_u(s1,s2)⋅jump_u(d1,d2) - n_s1s2⋅mean_t(d1,d2)⋅jump_u(s1,s2) - n_s1s2⋅mean_t(s1,s2)⋅jump_u(d1,d2)

  function a_solid((d1,d2),(s1,s2),(u,p,φ))
    n_Γ_s1s2 = get_normal_vector(Ω.Γ_s1s2)
    return ∫(a_s1_Ω(d1,s1))Ω.dΩs1 + ∫(a_s2_Ω(d2,s2))Ω.dΩs2 +
      ∫(j_s_Γ((d1,d2),(s1,s2),n_Γ_s1s2))Ω.dΓ_s1s2 +
      ∫(j_s_k1(d1,s1))Ω.dΓg_s1 + ∫(j_s_k2(d2,s2))Ω.dΓg_s2 +
      ∫(v_s_ψ(d1,s1))Ω.dΩs1 + ∫(v_s_ψ(d2,s2))Ω.dΩs2
  end
  function l_solid((s1,s2),(u,p,φ))
    n_Γ_s1f = -get_normal_vector(Ω.Γ_s1f)
    n_Γ_s2f = -get_normal_vector(Ω.Γ_s2f)
    return ∫(-(1-Ω.ψ_s)*(n_Γ_s1f ⋅ σf(u,p)) ⋅ s1)Ω.dΓ_s1f + ∫(-(1-Ω.ψ_s)*(n_Γ_s2f ⋅ σf(u,p)) ⋅ s2)Ω.dΓ_s2f
  end

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_act)
  iso_vol_frac(φ) = ∫(Ω.ψ_s/hmin^2)Ω.dΩs1 + ∫(Ω.ψ_s/hmin^2)Ω.dΩs2
  J_comp((u,p,d1,d2),φ) = ∫(ε(d1) ⊙ (σ1 ∘ ε(d1)))Ω.dΩs1 + ∫(ε(d2) ⊙ (σ2 ∘ ε(d2)))Ω.dΩs2 + iso_vol_frac(φ)
  Vol_Ωs1(_,φ) = ∫(1)Ω.dΩs1
  Vol_Ωs2(_,φ) = ∫(1)Ω.dΩs2

  ## Staggered operators and spaces

  # Setup spaces
  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  ls_elast = LUSolver()
  ls_fluid = LUSolver()

  state_collection = EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω,_φh)
    # Test spaces
    V = TestFESpace(Ω.Ωf_act,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_f_D","Gamma_Top","Gamma_Bottom","Gamma_s_D"])
    Q = TestFESpace(Ω.Ωf_act,reffe_p,conformity=:L2)
    T1 = TestFESpace(Ω.Ωs1_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D","Gamma_Bottom"])
    T2 = TestFESpace(Ω.Ωs2_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D","Gamma_Bottom"])
    T = MultiFieldFESpace([T1,T2])
    # Trial spaces
    U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
    P = TrialFESpace(Q)
    R1 = TrialFESpace(T1)
    R2 = TrialFESpace(T2)
    R = MultiFieldFESpace([R1,R2])
    # Multifield spaces
    UP = MultiFieldFESpace([U,P])
    VQ = MultiFieldFESpace([V,Q])
    # Aux spaces
    V_upφ = MultiFieldFESpace([U,P,V_φ...])
    V_upd1d2 = MultiFieldFESpace([U,P,R1,R2])
    assem_U = SparseMatrixAssembler(V_upd1d2,V_upd1d2)
    # State maps
    φ_to_up = AffineFEStateMap(a_fluid,l_fluid,UP,VQ,V_φ;ls=ls_fluid,adjoint_ls=ls_fluid)
    upφ_to_d1d2 = AffineFEStateMap(a_solid,l_solid,R,T,V_upφ;ls=ls_elast,adjoint_ls=ls_elast,∂ϕ_ad_type=:split)
    (;
      :φ_to_up => φ_to_up, :upφ_to_d1d2 => upφ_to_d1d2,
      :J => GridapTopOpt.StateParamMap(J_comp,V_upd1d2,V_φ,assem_U,assem_V_φ),
      :C => map(Ci -> GridapTopOpt.StateParamMap(Ci,V_upd1d2,V_φ,assem_U,assem_V_φ),[Vol_Ωs1,Vol_Ωs2]),
      :UP => UP, :R => R, :V_upφ => V_upφ, :V_upd1d2 => V_upd1d2
    )
  end

  function φ_to_jc(φ)
    GridapTopOpt.ignore_derivatives() do
      update_collection!(state_collection,FEFunction(V_φ,φ))
    end
    up = state_collection.φ_to_up(φ)
    u = restrict(state_collection.UP,up,1)
    p = restrict(state_collection.UP,up,2)
    φ1,φ2 = map(i->restrict(V_φ,φ,i),1:2)
    upφ1φ2 = combine_fields(state_collection.V_upφ,u,p,φ1,φ2)
    d1d2 = state_collection.upφ_to_d1d2(upφ1φ2)
    d1 = restrict(state_collection.R,d1d2,1)
    d2 = restrict(state_collection.R,d1d2,2)
    upd1d2 = combine_fields(state_collection.V_upd1d2,u,p,d1,d2)
    j = state_collection.J(upd1d2,φ)
    c1 = state_collection.C[1](upd1d2,φ)/vol_D - vf
    c2 = state_collection.C[2](upd1d2,φ)/vol_D - vf
    return [j,c1,c2]
  end

  function dCi!(Ci,dC,φ)
    φh = FEFunction(V_φ,φ)
    _dC(q) = gradient(φ -> Ci(nothing,φ),φh)
    Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
  end

  pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,2,
    analytic_dC=[(dC,φ)->dCi!(Vol_Ωs1,dC,φ), (dC,φ)->dCi!(Vol_Ωs2,dC,φ)])

  ## Hilbertian extension-regularisation problems
  hilb_ls = LUSolver()
  _α(hₕ) = (alpha*hₕ)^2
  a_hilb((p1,p2),(q1,q2)) = ∫((_α ∘ hₕ)*∇(p1)⋅∇(q1) + p1*q1 + (_α ∘ hₕ)*∇(p2)⋅∇(q2) + p2*q2)dΩ_act;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=hilb_ls)

  ## Optimiser
  has_oscillations(m,i) = GridapTopOpt.default_has_oscillations(m,i;itlength=100,itstart=200)
  if ls == "on-late"
    optimiser=HilbertianProjection(pcfs,ls_evo,vel_ext,φh;verbose=i_am_main(ranks),constraint_names=[:Vol_Ωs1,:Vol_Ωs2],γ,
      ls_γ_max=γ,ls_enabled=true,ls_ξ_reduce_abs_tol=0.001,ls_max_iters=3,ls_δ_inc=1.2,ls_enable_it=200)
  elseif ls == "on"
    optimiser=HilbertianProjection(pcfs,ls_evo,vel_ext,φh;verbose=i_am_main(ranks),constraint_names=[:Vol_Ωs1,:Vol_Ωs2],γ,
      ls_γ_max=γ,ls_enabled=true,ls_ξ_reduce_abs_tol=0.001,ls_max_iters=3,ls_δ_inc=1.2)
  else
    optimiser=HilbertianProjection(pcfs,ls_evo,vel_ext,φh;verbose=i_am_main(ranks),constraint_names=[:Vol_Ωs1,:Vol_Ωs2],γ,
      ls_enabled=false,has_oscillations)
  end
  for (it,_,φh) in optimiser
    if iszero(it % iter_mod)
      uh,ph = get_state(state_collection.φ_to_up)
      dh1,dh2 = get_state(state_collection.upφ_to_d1d2)
      writevtk(Ω_act,files_path*"Omega_bg_$it",
        cellfields=[
          "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),
          "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),
          "uh"=>uh,"ph"=>ph,"dh1"=>dh1,"dh2"=>dh2,
          "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f
        ])
      writevtk(Ω.Ωs1,files_path*"Omega_s1_$it",
        cellfields=["dh"=>dh1])
      writevtk(Ω.Ωs2,files_path*"Omega_s2_$it",
        cellfields=["dh"=>dh2])
      writevtk(Ω.Ωf,files_path*"Omega_f_$it",
        cellfields=["uh"=>uh,"ph"=>ph])
    end
    write_history(path*"/history.txt",optimiser.history; ranks)

    isolated_vol = sum(iso_vol_frac(φh))
    println(" --- Isolated volume: ",isolated_vol)
  end
  it = get_history(optimiser).niter;
  uh,ph = get_state(state_collection.φ_to_up)
  dh1,dh2 = get_state(state_collection.upφ_to_d1d2)
  writevtk(Ω_act,files_path*"Omega_bg_$it",
    cellfields=[
      "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),
      "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),
      "uh"=>uh,"ph"=>ph,"dh1"=>dh1,"dh2"=>dh2,
      "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f
    ])
  writevtk(Ω.Ωs1,files_path*"Omega_s1_$it",
    cellfields=["dh"=>dh1])
  writevtk(Ω.Ωs2,files_path*"Omega_s2_$it",
    cellfields=["dh"=>dh2])
  writevtk(Ω.Ωf,files_path*"Omega_f_$it",
    cellfields=["uh"=>uh,"ph"=>ph])
  nothing
end

## Run
function build_model(mesh_path,ranks=nothing;L=1.0,H=0.5,x0=0.5,l=0.4,w=0.015,a=0.3,b=0.015)
  geo_params = (;L,H,x0,l,w,a,b)
  if isnothing(ranks)
    model = GmshDiscreteModel(mesh_path)
  else
    model = GmshDiscreteModel(ranks,mesh_path)
  end
  model = UnstructuredDiscreteModel(model)
  return model, geo_params
end

with_mpi() do distribute
  ranks = distribute(LinearIndices((nprocs,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type superlu_dist"
  model, geo_params = build_model((@__DIR__)*"/Meshes/FSI.msh",ranks)
  GridapPETSc.with(;args=split(petsc_options)) do
    main(model, geo_params, ranks, linesearch, γ, alpha, E1, E2, vf, γg, write_dir)
  end
end