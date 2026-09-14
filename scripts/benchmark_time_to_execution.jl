using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt
# Background mesh
base_model = CartesianDiscreteModel((0,1,0,1,0,1),(41,41,41))
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
Dφ₁ = DiscreteGeometryFromFEFunction(φhᵢ[1],model,name="Dφ₁")
Dφ₂ = DiscreteGeometryFromFEFunction(φhᵢ[2],model,name="Dφ₂")
Dφ₃ = DiscreteGeometryFromFEFunction(φhᵢ[3],model,name="Dφ₃")
# Triangulation of Ω and Γ
Ω1 = intersect(Dφ₁ ∩ !Dφ₂,!Dφ₃,name="Ω1")
Ω2 = intersect(Dφ₁ ∩ Dφ₂,!Dφ₃,name="Ω2")
Ω3 = intersect(Dφ₁ ∩ !Dφ₂,Dφ₃,name="Ω3")
Ω4 = intersect(Dφ₁ ∩ Dφ₂,Dφ₃,name="Ω4")
Ω5 = intersect(!Dφ₁ ∩ !Dφ₂,!Dφ₃,name="Ω5")
Ω6 = intersect(!Dφ₁ ∩ Dφ₂,!Dφ₃,name="Ω6")
Ω7 = intersect(!Dφ₁ ∩ !Dφ₂,Dφ₃,name="Ω7")
Ω8 = intersect(!Dφ₁ ∩ Dφ₂,Dφ₃,name="Ω8")
_all = Ω1 ∪ Ω2 ∪ Ω3 ∪ Ω4 ∪ Ω5 ∪ Ω6 ∪ Ω7 ∪ Ω8
cutgeo = cut(PolytopalLevelSetCutter(),model,_all)
Ω = DifferentiableTriangulation(cutgeo,"Ω1")
dΩ = Measure(Ω,2*order)
# Functional and gradient
g((x,y,z)) = cos(x)*cos(y)*cos(z)
J(φhᵢ) = ∫(g)dΩ

using BenchmarkTools
function ad_run()
  gradient(J,φhᵢ);
  assemble_vector(dJ,Vφᵢ)
end
@benchmark ad_run()

n1 = ∇(φhᵢ[1])/(norm ∘ (∇(φhᵢ[1])))
n2 = ∇(φhᵢ[2])/(norm ∘ (∇(φhᵢ[2])))
n3 = ∇(φhᵢ[3])/(norm ∘ (∇(φhᵢ[3])))
Γ1 = EmbeddedBoundary(cutgeo,"Ω1","Ω5")
Γ2 = EmbeddedBoundary(cutgeo,"Ω1","Ω2")
Γ3 = EmbeddedBoundary(cutgeo,"Ω1","Ω3")
dΓ1 = Measure(Γ1,order*2)
dΓ2 = Measure(Γ2,order*2)
dΓ3 = Measure(Γ3,order*2)
dJ_analytic((w1,w2,w3)) = -1*∫(g*w1/abs(n1⋅∇(φhᵢ[1])))dΓ1 + ∫(g*w2/abs(n2⋅∇(φhᵢ[2])))dΓ2 + ∫(g*w3/abs(n3⋅∇(φhᵢ[3])))dΓ3

function analytic_run()
  assemble_vector(dJ_analytic,Vφᵢ);
end
@benchmark analytic_run()