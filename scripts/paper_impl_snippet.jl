using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt
# Background mesh
base_model = CartesianDiscreteModel((0,1,0,1,0,1),(15,15,15))
ref_model = refine(UnstructuredDiscreteModel(base_model), refinement_method = "barycentric")
model = get_model(ref_model)
# Level-set functions and geometries Dφᵢ
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vφᵢ = MultiFieldFESpace([TestFESpace(model,reffe) for i in 1:3])
φ₁ = x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.25
φ₂ = x->sqrt((x[1]-0.75)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.1
φ₃ = x->sqrt((0.25-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))^2 + (x[3]-0.5)^2) - 0.025
φhᵢ = interpolate([φ₁,φ₂,φ₃],Vφᵢ)
Dφ₁ = DiscreteGeometryFromFEFunction(φhᵢ[1],model)
Dφ₂ = DiscreteGeometryFromFEFunction(φhᵢ[2],model)
Dφ₃ = DiscreteGeometryFromFEFunction(φhᵢ[3],model)
# Triangulation of Ω and Γ
Ω_geo = Dφ₁ ∩ !Dφ₂ ∩ !Dφ₃
cutgeo = cut(PolytopalLevelSetCutter(),model,Ω_geo)
Ω = DifferentiableTriangulation(cutgeo,Ω_geo)
Γ = DifferentiableEmbeddedBoundary(cutgeo,Ω_geo,Dφ₃)
dΩ = Measure(Ω,2*order)
dΓ = Measure(Γ,2*order)
# Functional and gradient
J(φhᵢ) = ∫(1)dΩ + ∫(1)dΓ
dJ = gradient(J,φhᵢ)
vec_dJ = assemble_vector(dJ,Vφᵢ)

## SOME VALIDATION (Further validation can be found in GridapTopOpt/test/PolytopalCuttersTests.jl)
using FiniteDiff
_i = 0
function fdm_compute(φ)
  global _i += 1
  @show _i
  φhᵢ = FEFunction(Vφᵢ,φ)
  Dφ₁ = DiscreteGeometryFromFEFunction(φhᵢ[1],model)
  Dφ₂ = DiscreteGeometryFromFEFunction(φhᵢ[2],model)
  Dφ₃ = DiscreteGeometryFromFEFunction(φhᵢ[3],model)
  Ω_geo = Dφ₁ ∩ !Dφ₂ ∩ !Dφ₃
  cutgeo = cut(PolytopalLevelSetCutter(),model,Ω_geo)
  Ω = Triangulation(cutgeo,Ω_geo)
  Γ = EmbeddedBoundary(cutgeo,Ω_geo,Dφ₃)
  dΩ = Measure(Ω,2*order)
  dΓ = Measure(Γ,2*order)
  return sum(∫(1)dΩ + ∫(1)dΓ)
end
dJ_FD = FiniteDiff.finite_difference_gradient(fdm_compute,get_free_dof_values(φhᵢ))
maximum(abs,dJ_FD - vec_dJ)/maximum(abs,dJ_FD) < 1e-7 # 8.140788030303583e-8 < 1e-7