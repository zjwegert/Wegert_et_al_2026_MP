using Pkg
Pkg.activate(".")
using Mustache

template = read("./job_template.sh",String)

# Strong scaling benchmark
ssb_parts = [(1,1,1),(1,1,2),(1,2,2),(2,2,2),(2,2,4),(2,4,4),(3,4,4)]
ssb_n = 62
ssb_time = [2,2,1,1,1,1,1]
ssb_mem = max.(64,4prod.(ssb_parts))

for i ∈ Base.OneTo(length(ssb_parts))
  ncpus = prod(ssb_parts[i])
  mem = ssb_mem[i]
  wallhr = ssb_time[i]
  Px,Py,Pz = ssb_parts[i]
  name = "strong_P$(ncpus)_n$(ssb_n)"
  println("$name -> $(ssb_n^3*24/ncpus) cells per CPU")
  settings = (;name,ncpus,mem,wallhr,Px,Py,Pz,n=ssb_n,run_type="strong",wallmin="00")
  content = Mustache.render(template, settings)
  open((@__DIR__)*"/$name.pbs","w") do f
    write(f,content)
  end
end

# Weak scaling benchmark
cells_per_proc = 120000
wsb_parts = [(3,4,4), (6,8,8), (12,16,16), (24,24,24)]
wsb_n = round.(Int,map(x->(prod(x)*cells_per_proc/24)^(1/3), wsb_parts))
wsb_time = [30,30,30,30]
wsb_mem = 4prod.(wsb_parts)

for i ∈ Base.OneTo(length(wsb_parts))
  ncpus = prod(wsb_parts[i])
  mem = wsb_mem[i]
  wallmin = wsb_time[i]
  Px,Py,Pz = wsb_parts[i]
  n = wsb_n[i]
  name = "weak_P$(ncpus)_n$(n)"
  println("$name -> $(n^3*24/ncpus) cells per CPU")
  settings = (;name,ncpus,mem,wallhr="00",Px,Py,Pz,n,run_type="weak",wallmin)
  content = Mustache.render(template, settings)
  open((@__DIR__)*"/$name.pbs","w") do f
    write(f,content)
  end
end