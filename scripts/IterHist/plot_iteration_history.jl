using DelimitedFiles, CairoMakie, ColorSchemes, Colors, LaTeXStrings

history_file = "../../results/ANISO/history.txt"
data,head = readdlm(history_file,',',header=true)
J = data[:,1]
Vol_Ω1 = data[:,2] .+ 0.05
Vol_Ω3 = data[:,3] .+ 0.05# Renamed in paper to Vol_Ω2

# colors=colorschemes[:seaborn_pastel][[3,4,1]]

colors = [RGB(0,0,0),
  Colors.RGB(198/255,204/255,255/255),
  Colors.RGB(255/255,181/255,181/255)
];
fig = with_theme(theme_latexfonts(), palette=(color=colors,markercolor=color,patchcolor=color)) do
    fig = Figure(fontsize = 20,size=(700,350))#, markersize = 20)
    ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel="Objective")
    lines!(ax,1:length(J),J,label=L"Objective",linestyle=:solid, linewidth = 3)
    ax = Axis(fig[1, 2], xlabel = "Iteration", ylabel="Volume fraction")
    lines!(ax,[1,length(Vol_Ω1)], [0.05,0.05], color=:black, linestyle=:dot, linewidth = 3)
    lines!(ax,1:length(Vol_Ω1),Vol_Ω1,label=L"D_{\phi_1}\cap D_{\phi_2}^\complement",linestyle=:solid, linewidth = 3, color=colors[2])
    lines!(ax,1:length(Vol_Ω3),Vol_Ω3,label=L"D_{\phi_1}\cap D_{\phi_2}",linestyle=:dash, linewidth = 3, color=colors[3])
    Legend(fig[0,2],ax,orientation=:horizontal, )#patchsize = (60, 10))
    fig
end


save("./ANISO_iter_hist.png",fig;pt_per_unit=300)