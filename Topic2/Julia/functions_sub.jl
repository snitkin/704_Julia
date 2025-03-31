
function construct_IRF(xfull,n)
    d = Dict{String,Any}()
    for i = 1:n
        d[(allinputs[i])] = xfull[:,i]
    end
    return d
end

function qfun(P,theta)
    eval = P["m"].*(1/theta)./(1 + (1/theta).^P["iota"]).^(1-P["iota"]); 
end
function ffun(P,theta)
    eval = P["m"].*(theta./(1 + theta.^P["iota"]).^(1-P["iota"]))
end

function ffun_inverse(P, eval)
    theta = (eval ./ P["m"]) .* (1 + (eval ./ P["m"]).^(P["iota"] - 1))
    return theta
end