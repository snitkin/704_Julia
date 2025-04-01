using Debugger
using SparseArrays
using Plots
using DelimitedFiles
using DataFrames
using LinearAlgebra
using JLD
using Optim
using NLsolve
using Statistics
using Printf
using LaTeXStrings
using Plots.PlotMeasures
using Roots
using Interpolations

include("./functions_sub.jl")
include("./Sequence_Space_Solver.jl")
include("./DMP_Model.jl")
include("./compute_steady_state.jl")

colplot = palette(:Oranges_4);
colplot_green = palette(:Greens_4);
colplot_blue = palette(:Blues_3);


function set_parameters(;
        gamma = 0.5,
        s = 0.02,
        beta = (0.96).^(1/12),
        b = 0.95,
        iota = 1.6,
        w_fix = 0.95
    )

    P = Dict{String,Any}()
    m = 1
    P["m"] = m;
    P["iota"] = iota;

    uss = 0.05;
    theta_vals = range(0.25, .5, length=100);
    f_vals = [ffun(P, theta) for theta in theta_vals];
    f_inv = LinearInterpolation(f_vals, theta_vals);
    #lambdau = (s*(1-uss))/uss;
    thetass = f_inv(0.38);
    
 
    zss =1;
    wssfun(c) = (1-gamma)*b + gamma*(zss + thetass*c)

    solvefun(c) = c - beta./(1-beta.*(1-s)).*qfun(P,thetass).*(zss - wssfun(c));
    c = fzero(solvefun, 1.0)  # Provide an initial guess instead of an interval
    P["c"] = c;
    P["xi"] = 0.5;
    P["w_fix"] = w_fix;
    P["gamma"] = gamma;
    P["s"] = s;
    P["b"] = b;
    P["beta"] = beta;
    P["rho"] = 0.97327566
    P["uss"] = uss
    P["thetass"] = thetass;
    vss = uss*thetass;
    P["vss"] = vss
    P["omega"] = qfun(P,thetass)*vss;
    P["z shock"] = 1;
    return P
end


P = set_parameters()



T = 500;

pi_prob = 1.97/2;
z_h = exp(0.01);
z_l = exp(-0.01);

Shock_Var_in = "z shock"
Shock_Path = ones(T) * z_h
for t = 1:T
    if Shock_Path[t] == z_h
        Shock_Path[t] = rand() < pi_prob ? z_h : z_l
    elseif Shock_Path[t] == z_l
        Shock_Path[t] = rand() < pi_prob ? z_l : z_h
    end
end


ss0 = 0.1*ones(length(references))
ss0 = steady_state(ss0,references,P,varargin_eq,"z shock",ones(T))
ss1 = copy(ss0)
xfull = system_solve(ss0,ss1,references,P,varargin_eq,T,
Shock_Var_in,Shock_Path);

d = construct_IRF(xfull,n)

print(std(log.(d["ud"])))
print(std(log.(d["theta"])))
print(std(log.(d["theta"].*d["ud"])))

"""
Tlim = 200;
lwset = 4
plt_u = plot(1:T,d["ud"], lw=lwset)
plot!(1:T,P["uss"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Unemployment rate, "*L"u")
fig_name_save = "/Users/snitkin/Documents/Macro/PSETS/PSET Masao 1/IRF_u.pdf"
savefig(fig_name_save)

vacancy_rate = d["theta"] .* d["ud"]
plt_vacancy_rate = plot(1:T, vacancy_rate, lw=lwset)
plot!(1:T, P["thetass"] * P["uss"] * ones(T), lw=lwset, color=:grey, linestyle=:dash)
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Vacancy rate, "*L"v")
fig_name_save = "/Users/snitkin/Documents/Macro/PSETS/PSET Masao 1/IRF_v.pdf"
savefig(fig_name_save)



plt_s = plot(1:T,Shock_Path, lw=lwset)
xlims!((0, Tlim))
plot!(1:T,P["s"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Separation rate, "*L"s")


plt_v = plot(1:T,d["vd"], lw=lwset)
plot!(1:T,P["vss"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Vacancy, "*L" v")


plt_theta = plot(1:T,d["theta"], lw=lwset)
plot!(1:T,P["thetass"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Market tightness, "*L"\theta")
ylims!((0,0.05))

plot(plt_s,plt_u,plt_v,plt_theta,size=(800,500))
plot!(bottom_margin=4mm)
fig_name_save = "./figure/IRF_s.pdf"
savefig(fig_name_save)

##########################################################################################
dt = 0.01;
Tex = Int(1/dt)
up_ck = P["uss"]*ones(T*Tex)
up_ck[1] = P["uss"] + Shock_Path[1]*(1-P["uss"])
vp_ck = P["vss"]*ones(T*Tex)
for t = 2:(T*Tex)
    theta_temp = vp_ck[t-1]/up_ck[t-1]
    up_ck[t] = up_ck[t-1] + dt*(P["s"]*(1-up_ck[t-1]) - ffun(P,theta_temp)*up_ck[t-1]);
    vp_ck[t] = vp_ck[t-1] + dt*(P["omega"] - qfun(P,theta_temp)*vp_ck[t-1]);
end
tgrid = dt*(1:(T*Tex))
plot(tgrid,vp_ck)
plot(tgrid,up_ck)

theta_ck = vp_ck./up_ck


Tlim = T;
lwset = 4
plt_u = plot(tgrid,up_ck, lw=lwset)
plot!(1:T,P["uss"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Unemployment rate, "*L"u")


plt_s = plot(1:T,Shock_Path, lw=lwset)
xlims!((0, Tlim))
plot!(1:T,P["s"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Separation rate, "*L"s")


plt_v = plot(tgrid,vp_ck, lw=lwset)
plot!(1:T,P["vss"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Vacancy, "*L" v")


plt_theta = plot(tgrid,theta_ck, lw=lwset)
plot!(1:T,P["thetass"]*ones(T), lw=lwset, color=:grey, linestyle=:dash )
xlims!((0, Tlim))
plot!(legend=:false)
plot!(xlabel="Months")
plot!(title="Market tightness, "*L"\theta")
ylims!((0,0.05))

plot(plt_s,plt_u,plt_v,plt_theta,size=(800,500))
plot!(bottom_margin=4mm)
fig_name_save = "./figure/IRF_s_ck.pdf"
savefig(fig_name_save)
"""