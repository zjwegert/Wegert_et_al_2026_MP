using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapDistributed, GridapPETSc, PartitionedArrays

dir_path = ARGS[1]

function main(ranks,mesh_partition,path)
  # Output path
  files_path = path*"data/"
  if i_am_main(ranks)
    mkpath(files_path);
  end

  n = 10         # mesh size
  max_steps = 10 # Time-steps for evolution equation
  vf = 0.02      # Volume fraction
  α_coeff = 2    # Regularisation coefficient extension-regularisation
  iter_mod = 5   # Write output every iter_mod iterations
  γ = 0.2        # Step size in HJ transport equation

  # Model and some refinement
  base_model = CartesianDiscreteModel(ranks,mesh_partition,(0,0.5,0,0.5,0,0.5),(n,n,n))
  ref_model = refine(UnstructuredDiscreteModel(base_model), refinement_method = "barycentric")
  ref_model = refine(ref_model)
  model = get_model(ref_model)
  h = minimum(get_element_diameters(model))
  f_Γ_D(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 + (x[3]-0.5)^2 <= 0.05^2
  f_Γ_N(x) = ((x[1] ≈ 0 || x[1] ≈ 1) && (0.45 <= x[2] <= 0.55 + eps()) && (0.45 <= x[3] <= 0.55 + eps())) ||
    ((x[2] ≈ 0 || x[2] ≈ 1) && (0.45 - eps() <= x[1] <= 0.55) && (0.45 <= x[3] <= 0.55 + eps())) ||
    ((x[3] ≈ 0 || x[3] ≈ 1) && (0.45 - eps() <= x[1] <= 0.55) && (0.45 - eps() <= x[2] <= 0.55))
  update_labels!(1,model,f_Γ_D,"Omega_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")
  writevtk(model,path*"/model")

  ## Level-set function space and derivative regularisation space
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_regs = [TestFESpace(model,reffe_scalar) for _ in 1:2]
  U_regs = [TestFESpace(model,reffe_scalar) for _ in 1:2]
  V_reg = MultiFieldFESpace(V_regs)
  U_reg = MultiFieldFESpace(U_regs)
  V_φ = MultiFieldFESpace([TestFESpace(model,reffe_scalar) for _ in 1:2])

  ## Level-set function
  f1 = (x,y,z,a,b) -> -cos(a*π*(x-1/(2a)))*cos(2π*(y-1/(2a)))*cos(2π*(z-1/(2a)))-b
  f2 = (x,y,z,a,b) -> -cos(a*π*(x-3/(2a)))*cos(2π*(y-1/(2a)))*cos(2π*(z-1/(2a)))-b
  f = ((x,y,z),a,b) -> max(f1(x,y,z,a,b),f2(x,y,z,a,b))
  φh = interpolate([x->f(x,2,0.2),x->f(x,2,0.4)],V_φ)

  # Level-set function for fixed Dirichlet region
  V_φ_diri = TestFESpace(model,reffe_scalar)
  f_diri((x,y,z),) = (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 - (0.05+h/2)^2
  φh_diri = interpolate(f_diri,V_φ_diri)
  GridapTopOpt.correct_ls!(φh_diri)
  geo_diri = DiscreteGeometryFromFEFunction(φh_diri,model,name="φ_diri")

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  ## Triangulations and measures
  Ω_bg = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ_bg = Measure(Ω_bg,2)
  dΓ_N = Measure(Γ_N,2)
  vol_D = sum(∫(1)dΩ_bg)

  evolve_ls = MUMPS_Solver()
  reinit_nls = NewtonSolver(MUMPS_Solver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  reinits = [StabilisedReinitialiser(V_φ[i],dΩ_bg,h;stabilisation_method=ArtificialViscosity(1.0),nls=reinit_nls) for i in 1:2]
  evos = [CutFEMEvolver(V_φ[i],dΩ_bg,h;max_steps,γg=0.1,ode_ls=evolve_ls) for i in 1:2]
  ls_evo = MultiLevelSetEvolution(evos,reinits,V_φ;reuse_cache=true);

  function compute_geo(φh1,φh2)
    geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
    geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
    Ω1 = setdiff(setdiff(geo1,geo2),geo_diri,name="Ω1")
    Ω2 = setdiff(setdiff(geo2,geo1),geo_diri,name="Ω2")
    Ω3 = union(intersect(geo1,geo2),geo_diri,name="Ω3")
    return cut(PolytopalLevelSetCutter(),model,Ω1 ∪ Ω2 ∪ Ω3)
  end

  Ωs = EmbeddedCollection(model,φh;compute_cut=false) do φh
    φh1, φh2 = φh
    cutgeo = compute_geo(φh1,φh2)
    # Physical triangulations
    Ω1 = DifferentiableTriangulation(cutgeo,"Ω1")
    Ω2 = DifferentiableTriangulation(cutgeo,"Ω2")
    Ω3 = DifferentiableTriangulation(cutgeo,"Ω3")
    Γ13 = DifferentiableEmbeddedBoundary(cutgeo,"Ω1","Ω3")
    Γ23 = DifferentiableEmbeddedBoundary(cutgeo,"Ω2","Ω3")
    # Ghost triangulations
    Γg1 = GhostSkeleton(cutgeo,"Ω1")
    Γg2 = GhostSkeleton(cutgeo,"Ω2")
    Γg3 = GhostSkeleton(cutgeo,"Ω3")
    # Active triangulations
    Ω1act = Triangulation(cutgeo,ACTIVE,"Ω1")
    Ω2act = Triangulation(cutgeo,ACTIVE,"Ω2")
    Ω3act = Triangulation(cutgeo,ACTIVE,"Ω3")
    # Isolated volumes
    φ1 = get_free_dof_values(φh1)
    φ2 = get_free_dof_values(φh2)
    φ_diri = get_free_dof_values(φh_diri)
    φ_Ω2orΩ3 = min.(max.(φ1,-φ2),max.(-φ1,φ2),max.(φ1,φ2),φ_diri);
    φ_Ω2orΩ3_cv = map(get_cell_dof_values,local_views(FEFunction(V_φ[1],φ_Ω2orΩ3)))
    χ,_ = get_isolated_volumes_mask_polytopal(model,φ_Ω2orΩ3_cv,["Omega_D"])
    (;
      :Ω1  => Ω1, :Ω1act => Ω1act, :dΩ1 => Measure(Ω1,2),
      :Ω2  => Ω2, :Ω2act => Ω2act, :dΩ2 => Measure(Ω2,2),
      :Ω3  => Ω3, :Ω3act => Ω3act, :dΩ3 => Measure(Ω3,2),
      :Γ23 => Γ23, :dΓ23 => Measure(Γ23,2), :n_Γ23 => get_normal_vector(Γ23),
      :Γ13 => Γ13, :dΓ13 => Measure(Γ13,2), :n_Γ13 => get_normal_vector(Γ13),
      :Γg1 => Γg1, :dΓg1 => Measure(Γg1,2), :n_Γg1 => get_normal_vector(Γg1),
      :Γg2 => Γg2, :dΓg2 => Measure(Γg2,2), :n_Γg2 => get_normal_vector(Γg2),
      :Γg3 => Γg3, :dΓg3 => Measure(Γg3,2), :n_Γg3 => get_normal_vector(Γg3),
      :χ => χ
    )
  end

  ## Weak form
  α1 = SymTensorValue(5.0,0.0,0.0,1.0,0.0,1.0)
  α2 = SymTensorValue(1.0,0.0,0.0,5.0,0.0,1.0)
  α3 = SymTensorValue(1.0,0.0,0.0,1.0,0.0,5.0)
  α_max = max(maximum(α1),maximum(α2),maximum(α3))
  γg = 0.1*α_max
  λ = 10^2*α_max/h

  jump_u(u1,u2) = u1 - u2

  _w13_1(n) = n⋅α3⋅n/(n⋅α1⋅n + n⋅α3⋅n)
  _w13_2(n) = n⋅α1⋅n/(n⋅α1⋅n + n⋅α3⋅n)
  mean13_q(u1,u3,n) = (_w13_1 ∘ n)*(α1⋅∇(u1)) + (_w13_2 ∘ n)*(α3⋅∇(u3))

  _w23_1(n) = n⋅α3⋅n/(n⋅α2⋅n + n⋅α3⋅n)
  _w23_2(n) = n⋅α2⋅n/(n⋅α2⋅n + n⋅α3⋅n)
  mean23_q(u2,u3,n) = (_w23_1 ∘ n)*(α2⋅∇(u2)) + (_w23_2 ∘ n)*(α3⋅∇(u3))

  function a((u1,u2,u3),(v1,v2,v3),(φh1,φh2))
    # Compute normal
    # n_Γ12 = get_normal_vector(Ωs.Γ12)
    n_Γ13 = get_normal_vector(Ωs.Γ13)
    n_Γ23 = get_normal_vector(Ωs.Γ23)
    n_Γg1 = Ωs.n_Γg1; n_Γg2 = Ωs.n_Γg2; n_Γg3 = Ωs.n_Γg3
    return ∫( ∇(v1)⋅(α1⋅∇(u1)) + Ωs.χ*(v1*u1))Ωs.dΩ1 + ∫( ∇(v2)⋅(α2⋅∇(u2)) + Ωs.χ*(v2*u2))Ωs.dΩ2 + ∫( ∇(v3)⋅(α3⋅∇(u3)) + Ωs.χ*(v3*u3))Ωs.dΩ3 +
      ∫( (γg*h)*jump(n_Γg1⋅∇(v1))*jump(n_Γg1⋅∇(u1)) )Ωs.dΓg1 +
      ∫( (γg*h)*jump(n_Γg2⋅∇(v2))*jump(n_Γg2⋅∇(u2)) )Ωs.dΓg2 +
      ∫( (γg*h)*jump(n_Γg3⋅∇(v3))*jump(n_Γg3⋅∇(u3)) )Ωs.dΓg3 +
      ∫( λ*jump_u(v2,v3)*jump_u(u2,u3)
        - n_Γ23⋅mean23_q(u2,u3,n_Γ23)*jump_u(v2,v3)
        - n_Γ23⋅mean23_q(v2,v3,n_Γ23)*jump_u(u2,u3) )Ωs.dΓ23 +
      ∫( λ*jump_u(v1,v3)*jump_u(u1,u3)
        - n_Γ13⋅mean13_q(u1,u3,n_Γ13)*jump_u(v1,v3)
        - n_Γ13⋅mean13_q(v1,v3,n_Γ13)*jump_u(u1,u3) )Ωs.dΓ13
  end

  l((v1,v2,v3),(φh1,φh2)) = ∫(v1+v2+v3)dΓ_N

  ## Optimisation functionals
  J((u1,u2,u3),φ) = ∫(∇(u1)⋅(α1⋅∇(u1)))Ωs.dΩ1  +
    ∫(∇(u2)⋅(α2⋅∇(u2)))Ωs.dΩ2 +
    ∫(∇(u3)⋅(α3⋅∇(u3)))Ωs.dΩ3
  Vol_Ω1(u,φ) = ∫(1)Ωs.dΩ1
  Vol_Ω2(u,φ) = ∫(1)Ωs.dΩ2
  Vol_Ω3(u,φ) = ∫(1)Ωs.dΩ3

  ## FE operators
  ls = MUMPS_Solver()

  state_collection = EmbeddedCollection(model,φh;compute_cut=false) do _φh
    update_collection!(Ωs,_φh)
    V1 = TestFESpace(Ωs.Ω1act,reffe_scalar;dirichlet_tags=["Omega_D"])
    U1 = TrialFESpace(V1,0.0)
    V2 = TestFESpace(Ωs.Ω2act,reffe_scalar;dirichlet_tags=["Omega_D"])
    U2 = TrialFESpace(V2,0.0)
    V3 = TestFESpace(Ωs.Ω3act,reffe_scalar;dirichlet_tags=["Omega_D"])
    U3 = TrialFESpace(V3,0.0)
    V = MultiFieldFESpace([V1,V2,V3])
    U = MultiFieldFESpace([U1,U2,U3])
    state_map = AffineFEStateMap(a,l,U,V,V_φ;ls,adjoint_ls=ls)
    (;
      :state_map => state_map,
      :J => StateParamMap(J,state_map),
      :C => map(Ci -> StateParamMap(Ci,state_map),[Vol_Ω1,Vol_Ω2,Vol_Ω3])
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
    c3 = state_collection.C[3](u,φ)/vol_D - vf
    return [j,c1,c2,c3]
  end

  function dCi!(Ci,dC,φ)
    φh = FEFunction(V_φ,φ)
    _dC(q) = gradient(φ -> Ci(nothing,φ),φh)
    Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
  end

  pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,3,
    analytic_dC=[(dC,φ)->dCi!(Vol_Ω1,dC,φ), (dC,φ)->dCi!(Vol_Ω2,dC,φ), (dC,φ)->dCi!(Vol_Ω3,dC,φ)])

  ## Hilbertian extension-regularisation problems
  hilb_ls = CGAMGSolver()
  α = α_coeff*h
  a_hilb((p1,p2),(q1,q2)) = ∫(α^2*∇(p1)⋅∇(q1) + p1*q1 + α^2*∇(p2)⋅∇(q2) + p2*q2)dΩ_bg;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=hilb_ls)

  ## Optimiser
  optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;
    verbose=i_am_main(ranks),constraint_names=[:Vol_Ω1,:Vol_Ω2,:Vol_Ω3],γ=γ,ls_γ_max=γ,ls_ξ_reduce_abs_tol=1e-3)
  for (it,_,φh) in optimiser
    GridapPETSc.destroy(state_collection.state_map.cache.fwd_cache[1])
    GridapPETSc.destroy(state_collection.state_map.cache.adj_cache[1])
    if iszero(it % iter_mod)
      uh = get_state(state_collection.state_map)
      writevtk(Ω_bg,files_path*"Omega_$it",cellfields=[
        "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),
        "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"χ"=>Ωs.χ
      ])
      writevtk(Ωs.Ω1,files_path*"Omega1_$it",cellfields=["uh"=>uh[1]])
      writevtk(Ωs.Ω2,files_path*"Omega2_$it",cellfields=["uh"=>uh[2]])
      writevtk(Ωs.Ω3,files_path*"Omega3_$it",cellfields=["uh"=>uh[3]])
      writevtk(Ωs.Γ23,files_path*"Gamma23_$it",cellfields=["jump"=>jump_u(uh[2],uh[3])])
      writevtk(Ωs.Γ13,files_path*"Gamma13_$it",cellfields=["jump"=>jump_u(uh[1],uh[3])])
    end
    write_history(path*"/history.txt",optimiser.history;ranks)
  end
  it = get_history(optimiser).niter; uh = get_state(state_collection.state_map)
  writevtk(Ω_bg,files_path*"Omega_$it",cellfields=[
    "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),
    "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"χ"=>Ωs.χ
  ])
  writevtk(Ωs.Ω1,files_path*"Omega1_$it",cellfields=["uh"=>uh[1]])
  writevtk(Ωs.Ω2,files_path*"Omega2_$it",cellfields=["uh"=>uh[2]])
  writevtk(Ωs.Ω3,files_path*"Omega3_$it",cellfields=["uh"=>uh[3]])
  writevtk(Ωs.Γ23,files_path*"Gamma23_$it",cellfields=["jump"=>jump_u(uh[2],uh[3])])
  writevtk(Ωs.Γ13,files_path*"Gamma13_$it",cellfields=["jump"=>jump_u(uh[1],uh[3])])
end

## CG-AMG solver
CGAMGSolver(;kwargs...) = PETScLinearSolver(gamg_ksp_setup(;kwargs...))

function gamg_ksp_setup(;rtol=10^-8,maxits=100)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    rtol = PetscScalar(rtol)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = PetscInt(maxits)

    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  end

  return ksp_setup
end

MUMPS_Solver() = PETScLinearSolver(mumps_setup)

function mumps_setup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 0)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  6, 7)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  8, 77)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  24, 1)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 14, 200)
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
end

## Run
with_mpi() do distribute
  mesh_partition = (2,2,4)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks, mesh_partition, dir_path)
  end
end