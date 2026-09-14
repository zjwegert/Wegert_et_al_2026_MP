using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers, Gridap.FESpaces
using GridapDistributed
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt
using PartitionedArrays
using Random

const Px = parse(Int,ARGS[1])
const Py = parse(Int,ARGS[2])
const Pz = parse(Int,ARGS[3])
const n = parse(Int,ARGS[4])
const writedir = ARGS[5]
const run_type = ARGS[6]
const filename = "results_$(run_type)_P$(Px*Py*Pz)_n$(n)"
const nreps = 10

function generate_model(D,n,ranks,partition)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = CartesianDiscreteModel(ranks,partition,domain,cell_partition)
  ref_model = refine(UnstructuredDiscreteModel(base_model), refinement_method = "barycentric")
  return get_model(ref_model)
end

function run_benchmarks(ranks,partition,n)
  bgmodel = generate_model(3,n,ranks,partition)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = TestFESpace(bgmodel,reffe)
  V_φ1 = TestFESpace(bgmodel,reffe)
  V_φ2 = TestFESpace(bgmodel,reffe)
  V_φ3 = TestFESpace(bgmodel,reffe)
  V_φi = MultiFieldFESpace([V_φ1, V_φ2, V_φ3])
  φ1 = x -> cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3]) - 0.1sqrt(2)
  φ2 = x -> cos(4π*(x[1]-1/4))*cos(4π*x[2])*cos(4π*x[3]) - 0.1sqrt(2)
  φ3 = x -> cos(6π*(x[1]-1/6))*cos(6π*x[2])*cos(6π*x[3]) - 0.1sqrt(2)
  φhi = interpolate([φ1,φ2,φ3],V_φi);
  φh1 = φhi[1]
  φh2 = φhi[2]
  φh3 = φhi[3]

  function compute_geo(φh1,φh2,φh3)
      geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
      geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
      geo3 = DiscreteGeometryFromFEFunction(φh3,bgmodel,name="φ3")
      Ω4 = intersect(geo1 ∩ geo2,geo3,name="Ω4")
      return cut(PolytopalLevelSetCutter(),bgmodel,Ω4)
  end

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)
  cutgeo = compute_geo(φh1,φh2,φh3)
  Ω4 = DifferentiableTriangulation(cutgeo,"Ω4")
  dΩ = Measure(Ω4,2order)
  F1(φ) = ∫(fh)dΩ

  ∂Ω4 = DifferentiableEmbeddedBoundary(cutgeo,"Ω4");
  d∂Ω4 = Measure(∂Ω4,2order)
  F2(φ) = ∫(fh)d∂Ω4

  function benchmark_grad(F, φh, ad_type, ranks; nreps = 10)
    function f(F,φh)
      gradient(F,φh;ad_type)
    end
    return GridapTopOpt.benchmark(f, (F,φh), ranks; nreps)
  end

  bmark_f1_mono = benchmark_grad(F1,φhi,:monolithic,ranks; nreps)
  bmark_f2_mono = benchmark_grad(F2,φhi,:monolithic,ranks; nreps)
  bmark_f1_split = benchmark_grad(F1,φhi,:split,ranks; nreps)
  bmark_f2_split = benchmark_grad(F2,φhi,:split,ranks; nreps)

  if i_am_main(ranks)
    mkpath(writedir)
    open(writedir*"/"*filename*".txt","w") do f
      bcontent = "f1_mono,f2_mono,f1_split,f2_split\n"
      for i = 1:nreps
        bcontent *= "$(bmark_f1_mono[i]),$(bmark_f2_mono[i]),$(bmark_f1_split[i]),$(bmark_f2_split[i])\n"
      end
      write(f,bcontent)
    end
  end
  return
end

with_mpi() do distribute
  partition = (Px,Py,Pz)
  ranks = distribute(LinearIndices((prod(partition),)))
  run_benchmarks(ranks,partition,n)
end