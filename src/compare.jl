@everywhere include("episodiccontrol.jl")
using RLEnvDiscrete, CSV, DataFrames, JLD2

function getbaselineperformances(env, γ, policy)
    mdpl = MDPLearner(env, γ)
    policy_iteration!(mdpl)
    xmax = RLSetup(mdpl, env, ConstantNumberSteps(10^7), 
                   policy = policy, callbacks = [MeanReward()])
    run!(xmax)
    xmin = RLSetup(mdpl, env, ConstantNumberSteps(10^7), 
                   policy = EpsilonGreedyPolicy(1.), callbacks = [MeanReward()])
    run!(xmin)
    getvalue(xmin.callbacks[1]), getvalue(xmax.callbacks[1])
end

function getsetups(env, T; γ = 1., ϵ = .1, αql = 1., αnstep = .03 * αql, 
                   αlambda = .01 * αql, explorationinit = 1.,
                   λ = 1., nsteps = 10, explorationinitps = explorationinit,
                   explorationinitql = explorationinit)
    na = typeof(env) == DiscreteMaze ? env.mdp.na : env.na
    ns = typeof(env) == DiscreteMaze ? env.mdp.ns : env.ns
    params = ((:na, na), (:ns, ns), (:γ, γ), (:initvalue, Inf64))
    policy = VeryOptimisticEpsilonGreedyPolicy(ϵ)
    RL(l, c = []) = RLSetup(l, env, ConstantNumberSteps(T), 
                            policy = policy,
                            callbacks = [EvaluationPerT(div(T, 100)); c])
    ql(i) = RL(QLearning(; params..., λ = 0, α = αql))
    qlexplore(i) = RL(QLearning(; params..., λ = 0, α = αql, 
                               initvalue = explorationinitql))
    qllambda(i) = RL(QLearning(; params..., λ = λ, α = αlambda))
    nstepql(i) = RL(Sarsa(; nsteps = nsteps, params..., α = αnstep))
    ec(i) = RL(EpisodicControl(; params...))
    ps(i) = RL(SmallBackups(; params[1:end-1]...))
    myps(i) = RL(MySmallBackups(; params...))
    mc(i) = RL(MonteCarlo(; params...))
    psexplore(i) = RL(SmallBackups(; params[1:end-1]..., 
                                  initvalue = explorationinitps))
    psreset(i) = RL(SmallBackups(; params[1:end-1]..., maxcount = 0), 
                   [ModelReset()])
    Dict("ql" => ql, "qlexplore" => qlexplore, "qllambda" => qllambda,
         "nstepql" => nstepql, "ec" => ec, "ps" => ps, "mc" => mc,
         "psexplore" => psexplore, "psreset" => psreset), policy
end

function scaleres!(res, env, γ, policy)
    rmin, rmax = getbaselineperformances(typeof(env) == DiscreteMaze ? env.mdp :
                                         env, γ, policy)
    res[:result] .-= rmin
    res[:result] .*= 1/(rmax - rmin)
    res
end

function eccompare(getenv, N, T; agents = "all", γ = 1., relperf = true, 
                   offset = 1., offsetonlynonzero = false, kargs...)
    env = getenv()
    if typeof(env) == DiscreteMaze
        r = env.mdp.reward
    else
        r = env.reward
    end
    if offsetonlynonzero
        r[find(env.reward)] .+= offset
    else
        r .+= offset
    end
    setupdict, policy = getsetups(env, T; γ = γ, kargs...)
    if agents == "all" 
        setupcreators = setupdict
    else
        setupcreators = Dict()
        for a in agents
            setupcreators[a] = setupdict[a]
        end
    end
    res = compare(setupcreators, N, verbose = false)
    relperf ? scaleres!(res, env, γ, policy) : res
end

# push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\input{/home/j/research/latex/templates/colordef.tex}")
# plotopts = [(:linestyles, ["mblue", "gray", "{mblue, dashed}", "morange", "myellow",
#                            "mgreen", "{mgreen, dashed}", "mviolet", "mred"]),
#             (:showbest, false), (:nmaxpergroup, 5)]

@time res1 = vcat([eccompare(DetTreeMDP, 8, 200*100, γ = 1., ϵ = .1, 
                             explorationinit = 5., αnstep = 8e-2, 
                             αlambda = 1., λ = .2, offsetonlynonzero = true) 
                   for _ in 1:100]...);
# plotcomparison(res1; plotopts...)

@time res2 = vcat([eccompare(DetTreeMDPwithinrew, 8, 200*100, γ = 1., ϵ = .1,
                             explorationinit = 5., αnstep = 8e-2, 
                             αlambda = 1, λ = .2) 
                   for _ in 1:100]...);
# plotcomparison(res2; plotopts...)

@time res3 = vcat([eccompare(StochTreeMDP, 50, 100*100, γ = 1., ϵ = .1, αql = .1,
                             explorationinit = 5., αnstep = .05,
                             offsetonlynonzero = true, 
                             αlambda = .1, λ = .2) 
                   for _ in 1:100]...);
# plotcomparison(res3; plotopts...)

@time res4 = vcat([eccompare(DiscreteMaze, 8, 100*10000, γ = .99, ϵ = .1,
                             αnstep = 1e-2, nsteps = 25,
                             αlambda = 5e-3, λ = .2, offset = 0,
                             explorationinitql = 1/200,
                             explorationinitps = 5.) 
                   for _ in 1:50]...);
# plotcomparison(res4; plotopts...) 

@save "eccompare.jld2" res1, res2, res3, res4

function dftocsv(filename, df; T = 1, samples = 5)
    dfo = DataFrame(); dfo[:x] = collect(1:length(df[:result][1]))*T
    for g in groupby(df, :name)
        dfo[Symbol(g[:name][1])] = mean(g[:result])
        for i in 1:samples
            dfo[Symbol(g[:name][i], "_$i")] = g[:result][i]
        end
    end
    CSV.write(filename, dfo)
end

Ts = [200, 200, 100, 10000]
for i in 1:4
    dftocsv("res$i.csv", eval(Symbol("res$i")), T = Ts[i])
end
