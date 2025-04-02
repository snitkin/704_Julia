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

function q(mp, theta)
    # Unpack parameters
    @unpack barm, iota = mp
    # Define q as a function of theta
    q = barm / ((1.0 + (theta)^(iota))^(1/iota))
    return q
end
function f(mp, theta)
    # Unpack parameters
    @unpack barm, iota = mp
    # Define f as a function of theta
    f = barm / ((1.0 + (1/theta)^(iota))^(1/iota))
    return f
end

function solve_for_c(mp::ModelParams, u_value)
    # Unpack parameters
    @unpack barm, s, iota, gamma, b, beta, z_ss = mp
    # Define the equation to solve
    solve_theta(u,theta) = s*(1 - u) - u*f(mp, theta);
    # Solve for theta using nlsolve where the intial guess is 0.1
    sol = nlsolve(x -> solve_theta.(u_value,x), [0.1])
    theta_ss = sol.zero[1]
    println("The steady state value of theta  is: ", theta_ss)

    wssfun(c) = (1-gamma)*b + gamma*(z_ss + theta_ss*c)
    solvefun(c) = c - beta./(1-beta.*(1-s)).*q(mp, theta_ss).*(z_ss - wssfun(c));
    c_ss = nlsolve(x -> solvefun.(x), [1.0]).zero[1]  # Provide an initial guess 
    return c_ss
end

mp = ModelParams();
c = solve_for_c(mp, 0.05);


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

function solve_system(mp::ModelParams, z_h, z_l, c)
    @unpack gamma, b, beta, s, pi_mc = mp

    function equations(vars)
        theta_h, theta_l, w_h, w_l, J_h, J_l = vars
        eq1 = w_h - ((1 - gamma) * b + gamma * (z_h + theta_h * c))
        eq2 = w_l - ((1 - gamma) * b + gamma * (z_l + theta_l * c))
        eq3 = J_h - (z_h - w_h + beta * (1 - s) * (pi_mc * J_h + (1 - pi_mc) * J_l))
        eq4 = J_l - (z_l - w_l + beta * (1 - s) * (pi_mc * J_l + (1 - pi_mc) * J_h))
        eq5 = 0 - (beta * q(mp, theta_h) * (pi_mc * J_h + (1 - pi_mc) * J_l) - c)
        eq6 = 0 - (beta * q(mp, theta_l) * (pi_mc * J_l + (1 - pi_mc) * J_h) - c)
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    end

    initial_guess = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1]  # Initial guesses for theta_h, theta_l, w_h, w_l, J_h, J_l
    sol = nlsolve(x -> equations(x), initial_guess)
    theta_h, theta_l, w_h, w_l, J_h, J_l = sol.zero
    println("Solved theta_h: ", theta_h, ", theta_l: ", theta_l)
    println("Solved w_h: ", w_h, ", w_l: ", w_l)
    println("Solved J_h: ", J_h, ", J_l: ", J_l)
    return theta_h, theta_l, w_h, w_l, J_h, J_l
end

# Example usage
theta_h, theta_l, w_h, w_l, J_h, J_l = solve_system(mp, mp.z_h, mp.z_l, c)

theta_values = [theta_l, theta_h]
theta_vector = theta_values[z_simulation]

function ud_def(mp::ModelParams, ud,theta ,ud_m1)
    @unpack s = mp
    eval =  ud - ud_m1 - ( s.*(1-ud_m1) - f(mp,theta).*ud_m1 )
    return eval;
end