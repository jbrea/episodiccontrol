using Parameters, ReinforcementLearning, DataStructures.PriorityQueue

@with_kw struct EpisodicControl 
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = 1.
    initvalue::Float64 = Inf64
    Q::Array{Float64, 2} = zeros(na, ns) + initvalue
end
import ReinforcementLearning.defaultbuffer
function defaultbuffer(learner::EpisodicControl, env, preprocessor)
    defaultbuffer(MonteCarlo(), env, preprocessor)
end

import ReinforcementLearning.update!, ReinforcementLearning.selectaction
function update!(learner::EpisodicControl, buffer)
    actions = buffer.actions
    rewards = buffer.rewards
    states = buffer.states
    if learner.Q[actions[end-1], states[end-1]] == Inf64
        learner.Q[actions[end-1], states[end-1]] = -Inf64
    end
    if buffer.done[end]
        G = 0.
        for t in length(rewards):-1:1
            G = learner.γ * G + rewards[t]
            if G > learner.Q[actions[t], states[t]]
                learner.Q[actions[t], states[t]] = G
            end
        end
    end
end
selectaction(learner::EpisodicControl, policy, state) = selectaction(policy, learner.Q[:, state])

@with_kw mutable struct ModelReset 
    counter::Int64 = 0
end
import ReinforcementLearning.callback!
function callback!(c::ModelReset, rlsetup, state, action, reward, done)
    c.counter += 1
    if rlsetup.buffer.done[end]
        learner = rlsetup.learner
        learner.maxcount = c.counter
        ReinforcementLearning.processqueue!(learner)
        learner.Nsa .= 0
        learner.Ns1a0s0 = [Dict{Tuple{Int64, Int64}, Int64}() for _ in
                           1:length(learner.Ns1a0s0)]
        learner.queue = PriorityQueue(Base.Order.Reverse, zip(Int64[], Float64[]))
        c.counter = 0
        learner.maxcount = 0
    end
end
