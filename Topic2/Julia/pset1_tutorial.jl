# PSET 1 Question 3 Code. Edited by Shraddha Mandi for Tutorial Purposes
using NLsolve
using QuantEcon
using Parameters 

@with_kw struct ModelParams
    beta::Float64 = 0.96^(1/12);
    s::Float64 = 0.02;
    b::Float64 = 0.4;
    gamma::Float64 = 0.5;
    barm::Float64 = 1;
    iota::Float64 = 1.6;
    z_ss::Float64 = 1;
    z_h = exp(0.01);
    z_l = exp(-0.01);
    pi_mc::Float64 = 0.985;
end

function solve_for_c(mp::ModelParams, u_value)
    # Unpack parameters
    @unpack barm, s, iota, gamma, b, beta, z_ss = mp
    # Define f and q as functions of theta
    f(theta) = barm /( (1.0 + (1/theta)^(iota))^(1/iota) )
    q(theta) = barm /( (1.0 + (theta)^(iota))^(1/iota) )
    # Define the equation to solve
    solve_theta(u,theta) = s*(1 - u) - u*f(theta);
    # Solve for theta using nlsolve where the intial guess is 0.1
    sol = nlsolve(x -> solve_theta.(u_value,x), [0.1])
    theta_ss = sol.zero[1]
    println("The steady state value of theta  is: ", theta_ss)

    wssfun(c) = (1-gamma)*b + gamma*(z_ss + theta_ss*c)
    solvefun(c) = c - beta./(1-beta.*(1-s)).*q(theta_ss).*(z_ss - wssfun(c));
    c_ss = nlsolve(x -> solvefun.(x), [1.0]).zero[1]  # Provide an initial guess 
    return c_ss
end

mp = ModelParams();
c = solve_for_c(mp, 0.05);

println("The steady state value of c is: ", c)


# Question 3
#####################################################################################
# Create a Markov Chain for the state variable z
T = 500;
markov_matrix = [mp.pi_mc 1-mp.pi_mc; 1-mp.pi_mc mp.pi_mc];
mc = MarkovChain(markov_matrix)
# Simulate a Markov chain of length 500 for the state variable z
z_values = [mp.z_l, mp.z_h]
z_simulation = simulate(mc, T; init=1)
z_vector = z_values[z_simulation]

