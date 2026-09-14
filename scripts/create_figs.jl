using Gridap, Gridap.Geometry, Gridap.Algebra, Gridap.FESpaces, Gridap.Fields, Gridap.Adaptivity,
  Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs, Gridap.TensorValues, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt: PolytopalCellCutterMap, pref_to_gen_polytope_graph

# Fig2
begin
path = "results/Fig2/"
mkpath(path)
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(13,13)))
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)
f1 = x->sqrt((x[1]-0.67)^2+(x[2]-0.5)^2)-0.27
f2 = x->sqrt((x[1]-0.33)^2+(x[2]-0.5)^2)-0.27
reffe = ReferenceFE(lagrangian,Float64,1);
V_φ = MultiFieldFESpace([FESpace(model,reffe),FESpace(model,reffe)]);
φh = interpolate([f1,f2],V_φ)
Dφ1 = DiscreteGeometry(φh[1],model)
Dφ2 = DiscreteGeometry(φh[2],model)
Ω1 = intersect(Dφ1,!Dφ2)
Ω2 = intersect(Dφ1,Dφ2)
Ω3 = intersect(!Dφ1,Dφ2)
Ω4 = intersect(!Dφ1,!Dφ2)
cutgeo = cut(model,Ω1∪Ω2∪Ω3∪Ω4)
Dφ1_phys = Triangulation(cutgeo,Dφ1)
Dφ2_phys = Triangulation(cutgeo,Dφ2)
Ω1_phys = Triangulation(cutgeo,Ω1)
Ω1_act = Triangulation(cutgeo,ACTIVE,Ω1)
Ω2_phys = Triangulation(cutgeo,Ω2)
Ω2_act = Triangulation(cutgeo,ACTIVE,Ω2)
Ω3_phys = Triangulation(cutgeo,Ω3)
Ω3_act = Triangulation(cutgeo,ACTIVE,Ω3)
Ω4_phys = Triangulation(cutgeo,Ω4)
Ω4_act = Triangulation(cutgeo,ACTIVE,Ω4)
Γ_12 = EmbeddedBoundary(cutgeo,Ω1,Ω2)
Γ_14 = EmbeddedBoundary(cutgeo,Ω1,Ω4)
Γ_23 = EmbeddedBoundary(cutgeo,Ω2,Ω3)
Γ_34 = EmbeddedBoundary(cutgeo,Ω3,Ω4)

writevtk(model,path*"model")
writevtk(Dφ1_phys,path*"Dφ1_phys")
writevtk(Dφ2_phys,path*"Dφ2_phys")
writevtk(Ω1_phys,path*"Ω1_phys")
writevtk(Ω1_act,path*"Ω1_act")
writevtk(Ω2_phys,path*"Ω2_phys")
writevtk(Ω2_act,path*"Ω2_act")
writevtk(Ω3_phys,path*"Ω3_phys")
writevtk(Ω3_act,path*"Ω3_act")
writevtk(Ω4_phys,path*"Ω4_phys")
writevtk(Ω4_act,path*"Ω4_act")
writevtk(Γ_12,path*"Γ_12")
writevtk(Γ_14,path*"Γ_14")
writevtk(Γ_23,path*"Γ_23")
writevtk(Γ_34,path*"Γ_34")
end

# Rec cutter fig
path = "results/figs_rec_cutter/"
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(1,1)))
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)
# writevtk(model,path*"model")
N = 4

bg_vertices = get_cell_ref_coordinates(model);
bg_ctypes = get_cell_type(model);
bgcell_to_polys = expand_cell_data(get_polytopes(model),bg_ctypes);
bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);
bgcell_ref_to_phys_map = get_cell_map(model);

ls_2_path = path*"ls=2/"
mkpath(ls_2_path)
reffe = ReferenceFE(lagrangian,Float64,1);
V_φ = MultiFieldFESpace([FESpace(model,reffe),FESpace(model,reffe)]);
φh = zero(V_φ);
φh.free_values[1:5] .= [1,-1, 1, -1, -1]
φh.free_values[6:10] .= [-0.25, -1, -2, 1, 1]
cell_map_ls = get_data.(φh);

cutter = PolytopalCellCutterMap(2,2,1)
cache = return_cache(cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],getindex.(cell_map_ls,N)...);
evaluate!(cache,cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],getindex.(cell_map_ls,N)...);

φh[1](bgcell_ref_to_phys_map[N](Point(0,0)))
φh[1](bgcell_ref_to_phys_map[N](Point(1,0)))
φh[1](bgcell_ref_to_phys_map[N](Point(0,1)))

φh[2](bgcell_ref_to_phys_map[N](Point(0,0)))
φh[2](bgcell_ref_to_phys_map[N](Point(1,0)))
φh[2](bgcell_ref_to_phys_map[N](Point(0,1)))

φh[2](bgcell_ref_to_phys_map[N](Point(0,0.5)))
φh[2](bgcell_ref_to_phys_map[N](Point(0.5,0.5)))

# Parent
c = cache[3][1].data.counters[1]
p_graph = cache[3][1].data.graph[1:c]
p_vertices = cache[3][1].data.vertices[1:c]
# Left
c = cache[3][1].left.data.counters[1]
L_graph = cache[3][1].left.data.graph[1:c]
L_vertices = cache[3][1].left.data.vertices[1:c]
# Right
c = cache[3][1].right.data.counters[1]
R_graph = cache[3][1].right.data.graph[1:c]
R_vertices = cache[3][1].right.data.vertices[1:c]
# Left Left
c = cache[3][1].left.left.data.counters[1]
LL_graph = cache[3][1].left.left.data.graph[1:c]
LL_vertices = cache[3][1].left.left.data.vertices[1:c]
# Left Right
c = cache[3][1].left.right.data.counters[1]
LR_graph = cache[3][1].left.right.data.graph[1:c]
LR_vertices = cache[3][1].left.right.data.vertices[1:c]
# Right Left
c = cache[3][1].right.left.data.counters[1]
RL_graph = cache[3][1].right.left.data.graph[1:c]
RL_vertices = cache[3][1].right.left.data.vertices[1:c]
# Left Right
c = cache[3][1].right.right.data.counters[1]
RR_graph = cache[3][1].right.right.data.graph[1:c]
RR_vertices = cache[3][1].right.right.data.vertices[1:c]

writevtk(Polygon(p_vertices,p_graph),ls_2_path*"split_p_polygon")
writevtk(Polygon(L_vertices,L_graph),ls_2_path*"split_L_polygon")
writevtk(Polygon(R_vertices,R_graph),ls_2_path*"split_R_polygon")
writevtk(Polygon(LL_vertices,LL_graph),ls_2_path*"split_LL_polygon")
writevtk(Polygon(LR_vertices,LR_graph),ls_2_path*"split_LR_polygon")
writevtk(Polygon(RL_vertices,RL_graph),ls_2_path*"split_RL_polygon")
writevtk(Polygon(RR_vertices,RR_graph),ls_2_path*"split_RR_polygon")

for i in 1:cache[1][4][1]
  verts = cache[1][1][i]
  graph = cache[1][2][i]
  inout = cache[1][3][i][1]
  writevtk(Polygon(verts),ls_2_path*"split_simplexified_$(inout)_$i")
end

## 2D split and simplexify
path = "results/figs_split_and_simplexify/"
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(1,1)))
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)
N = 4

bg_vertices = get_cell_ref_coordinates(model);
bg_ctypes = get_cell_type(model);
bgcell_to_polys = expand_cell_data(get_polytopes(model),bg_ctypes);
bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);
bgcell_ref_to_phys_map = get_cell_map(model);

ls_1_path = path*"ls=1_2d/"
mkpath(ls_1_path)
reffe = ReferenceFE(lagrangian,Float64,1);
V_φ = FESpace(model,reffe);
φh = zero(V_φ);
φh.free_values .= [1,-1, 1, -1, -1]
cell_map_ls = get_data(φh);

cutter = PolytopalCellCutterMap(2,1,1)
cache = return_cache(cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],cell_map_ls[N]);
evaluate!(cache,cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],cell_map_ls[N]);

c = cache[3][1].data.counters[1]
p_graph = cache[3][1].data.graph[1:c]
p_vertices = cache[3][1].data.vertices[1:c]

c = cache[3][1].left.data.counters[1]
L_graph = cache[3][1].left.data.graph[1:c]
L_vertices = cache[3][1].left.data.vertices[1:c]

c = cache[3][1].right.data.counters[1]
R_graph = cache[3][1].right.data.graph[1:c]
R_vertices = cache[3][1].right.data.vertices[1:c]

writevtk(Polygon(p_vertices,p_graph),ls_1_path*"split_p_polygon")
writevtk(Polygon(L_vertices,L_graph),ls_1_path*"split_L_polygon")
writevtk(Polygon(R_vertices,R_graph),ls_1_path*"split_R_polygon")

for i in 2:cache[1][4][1]
  verts = cache[1][1][i]
  graph = cache[1][2][i]
  inout = cache[1][3][i][1]
  writevtk(Polygon(verts),ls_1_path*"split_R_simplexified_$(inout)_$i")
end

## 3D split and simplexify
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1,0,1),(1,1,1)))
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)
N = 4
# writevtk(model,path*"model")

bg_vertices = get_cell_ref_coordinates(model);
bg_ctypes = get_cell_type(model);
bgcell_to_polys = expand_cell_data(get_polytopes(model),bg_ctypes);
bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);
bgcell_ref_to_phys_map = get_cell_map(model);

ls_1_path = path*"ls=1_3d/"
mkpath(ls_1_path)
reffe = ReferenceFE(lagrangian,Float64,1);
V_φ = FESpace(model,reffe);
φh = zero(V_φ);
# φh.free_values .= 1
# φh.free_values[15] = -1
φh.free_values .= 1
φh.free_values[[1,5,11,2,6]] .= -1.5
cell_map_ls = get_data(φh);

cutter = PolytopalCellCutterMap(3,1,1)
cache = return_cache(cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],cell_map_ls[N]);
evaluate!(cache,cutter,bg_vertices[N],bgcell_to_graph[N],bgcell_ref_to_phys_map[N],cell_map_ls[N]);

c = cache[3][1].data.counters[1]
p_graph = cache[3][1].data.graph[1:c]
p_vertices = cache[3][1].data.vertices[1:c]

c = cache[3][1].left.data.counters[1]
L_graph = cache[3][1].left.data.graph[1:c]
L_vertices = cache[3][1].left.data.vertices[1:c]

c = cache[3][1].right.data.counters[1]
R_graph = cache[3][1].right.data.graph[1:c]
R_vertices = cache[3][1].right.data.vertices[1:c]

writevtk(Polyhedron(p_vertices,p_graph),ls_1_path*"split_p_polyhedra")
writevtk(Polyhedron(L_vertices,L_graph),ls_1_path*"split_L_polyhedra")
writevtk(Polyhedron(R_vertices,R_graph),ls_1_path*"split_R_polyhedra")

for i in 2:cache[1][4][1]
  verts = cache[1][1][i]
  graph = cache[1][2][i]
  inout = cache[1][3][i][1]
  writevtk(Polyhedron(TET,verts),ls_1_path*"split_R_simplexified_$(inout)_$i")
end

### 3d domain and facets
using GridapEmbedded,GridapEmbedded.LevelSetCutters
path = "results/figs_3d_dom_and_facets/"
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1,0,1),(41,41,41)))
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V_φs = MultiFieldFESpace([TestFESpace(model,reffe) for _ in 1:3])

f1 = x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.25
f2 = x->sqrt((x[1]-0.75)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.1
f3 = x->sqrt((0.25-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))^2 + (x[3]-0.5)^2) - 0.025

φsh = interpolate([f1,f2,f3],V_φs)
φh1, φh2, φh3 = φsh
geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
geo3 = DiscreteGeometryFromFEFunction(φh3,model,name="φ3")

geo_Ω = setdiff(setdiff(geo1,geo2),geo3)
cutgeo = cut(model,geo_Ω)

Ω_φ1 = Triangulation(cutgeo,"φ1")
Ω_φ2 = Triangulation(cutgeo,"φ2")
Ω_φ3 = Triangulation(cutgeo,"φ3")
Ω = Triangulation(cutgeo,geo_Ω)

mkpath(path)
writevtk(Ω_φ1,path*"/Ω_φ1")
writevtk(Ω_φ2,path*"/Ω_φ2")
writevtk(Ω_φ3,path*"/Ω_φ3")
writevtk(Ω,path*"/Ω")