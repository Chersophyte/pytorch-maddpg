from pettingzoo.sisl import foodcollector_v0, foodcollector_v1
from MADDPG import MADDPG
import numpy as np
import torch
from collections import deque
from param import Param
import os
import re
import gym

def make_advcomm_env(adv_agents, good_policy_dir, victim, ac_kwargs, ablate_kwargs, 
                     confidence_kwargs, **kwargs):
    env_fn = foodcollector_v1.parallel_advcomm_wrapper_fn(foodcollector_v1.env, 
        adv_agents, victim)
    env = env_fn(**kwargs)
    a = env.good_names[0]
    good_observation_space = env.good_observation_spaces[a]
    good_comm_space = env.good_communication_spaces[a]
    #print("shape of good comomunication space:{}".format(good_comm_space.shape))
    good_action_space = env.good_action_spaces[a]
    if ablate_kwargs is None:
        obs_dim = good_observation_space.shape[0] + good_comm_space.shape[0]
    else:
        k, n = ablate_kwargs['k'], ablate_kwargs['n']
        obs_dim =  good_observation_space.shape[0] + good_comm_space.shape[0]*k//(n-1)
    ac = MLPActorCritic(obs_dim, good_action_space, **ac_kwargs).to(Param.device)
    if good_policy_dir:
        state_dict, mean, std = torch.load(good_policy_dir, map_location=Param.device)
        ac.load_state_dict(state_dict)
        ac.moving_mean = mean
        ac.moving_std = std
    
    detector_ac = None
    
    default_policy = DefaultPolicy(ac, True if kwargs["victim_dist_action"] else False, 
                                   ablate_kwargs=ablate_kwargs, confidence_kwargs=confidence_kwargs,
                                   detector_ac=detector_ac)
    env.set_default_policy(default_policy)
    env.reset()
    return env, adv_agents, []

def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)
    return comm[choice]

def concatenate(obs, comm, ablate_kwargs=None):
    for agent in obs:
        if ablate_kwargs is None or agent in ablate_kwargs["adv_agents"]:
            obs[agent] = np.concatenate([obs[agent], comm[agent]])
        else:
            comm_agent = comm[agent].reshape(len(obs)-1, -1)
            comm_agent = shuffle(comm_agent, ablate_kwargs['k']).reshape(-1)
            obs[agent] = np.concatenate([obs[agent], comm_agent])
    return obs

def encode_obs(good_agent_name, obs_dim, obs, terminal=False):
    obs_array = np.zeros((n_agents, obs_dim))
    for i in range(len(good_agent_name)):
        agent = good_agent_name[i]
        if not terminal:
            obs_array[i] = obs[agent]
        else:
            obs_array[i] = terminal
    return obs_array
                          
def make_food_env(comm, **kwargs):
    if comm:
        env = foodcollector_v1.parallel_env(**kwargs)
    else:
        env = foodcollector_v0.parallel_env(**kwargs)
    env.reset()
    agent = env.agents[0]
    observation_space = env.observation_spaces[agent]
    action_space = env.action_spaces[agent]
    
    adv_agents = []
    good_agents = ["pursuer_{}".format(i) for i in range(kwargs["n_pursuers"])]
    return env, good_agents, adv_agents


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no-save', action="store_true")
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--log-freq', type=int, default=20)
    parser.add_argument('--trained-dir', type=str)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--truth', action="store_true")
    parser.add_argument('--smart', action="store_true")
    
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--critic-lr', type=float, default=3e-6)
    parser.add_argument('--noise-var', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.01)
        
    parser.add_argument('--count-agent', type=str, default='pursuer_0')
    parser.add_argument('--window-size', type=float, default=1.0)
    parser.add_argument('--n-pursuers', type=int, default=3)
    parser.add_argument('--n-evaders', type=int, default=1)
    parser.add_argument('--n-poison', type=int, default=1)
    parser.add_argument('--poison-scale', type=float, default=0.75)
    parser.add_argument('--poison-reward', type=float, default=-1.0)
    parser.add_argument('--n-sensors',  type=int, default=6)
    parser.add_argument('--max-cycle', type=int, default=200)
    parser.add_argument('--comm', action="store_true")
    parser.add_argument('--comm-freq',  type=int, default=1)
    parser.add_argument('--dist-action', action="store_true")
    parser.add_argument('--victim-dist-action', action="store_true")
    parser.add_argument('--sensor-range',  type=float, default=0.2)
    parser.add_argument('--evader-speed', type=float, default=0)
    parser.add_argument('--poison-speed', type=float, default=0)
    parser.add_argument('--speed-features', action="store_true")
    parser.add_argument('--recurrent', action="store_true")
    parser.add_argument('--food-revive', action="store_true", help="whether the food can be refreshed after being eaten")
    
    parser.add_argument('--convert-adv', type=str, nargs='+')
    parser.add_argument('--good-policy', type=str)
    parser.add_argument('--victim', type=str, default="pursuer_0")
    
    parser.add_argument('--ablate', action='store_true')
    parser.add_argument('--ablate-k', type=int, default=1)
    parser.add_argument('--ablate-median', type=int, default=1)
    
    parser.add_argument('--detector-policy-dir', type=str, default=None)
    args = parser.parse_args()
    
    ### Setup Cuda
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    ### Setup logger file
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(args.exp_name)), "wt")
    logger_file.write("Number of Pursuers: {}  Number of Food:{}  Number of Poison:{}\n".\
                      format(args.n_pursuers, args.n_evaders, args.n_poison))
    logger_file.write("Number of Sensors: {}  Sensor Range:{}    Speed Features:{}\n".\
                      format(args.n_sensors, args.sensor_range, args.speed_features))
    logger_file.write("Food Speed: {}  Poison Speed:{}\n".\
                      format(args.evader_speed, args.poison_speed))
    
    
    ### Setup Environment
    if args.convert_adv:
        env, good_agent_name, adv_agent_name = make_advcomm_env(adv_agents=args.convert_adv, good_policy_dir=args.good_policy, victim=args.victim,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=not args.victim_no_beta, 
                            recurrent=args.recurrent, ep_len=args.max_cycle),
                ablate_kwargs = ablate_kwargs, confidence_kwargs = confidence_kwargs,
                window_size=args.window_size, poison_scale=args.poison_scale,  food_revive=args.food_revive,
                max_cycles=args.max_cycle, n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.victim_dist_action, poison_reward=args.poison_reward,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, poison_speed=args.poison_speed,
                speed_features=args.speed_features, use_groudtruth=args.truth, smart_comm=args.smart, 
                comm_freq=args.comm_freq, victim_dist_action=args.victim_dist_action)
    else:
        env, good_agent_name, adv_agent_name = make_food_env(
                comm=args.comm, max_cycles=args.max_cycle,
                window_size=args.window_size, poison_scale=args.poison_scale, food_revive=args.food_revive,
                n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.dist_action, poison_reward=args.poison_reward,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, poison_speed=args.poison_speed,
                speed_features=args.speed_features, use_groudtruth=args.truth, 
                smart_comm=args.smart, comm_freq=args.comm_freq)
    
    ablate_kwargs=None
    
    observation_space = env.observation_spaces[good_agent_name[0]]
    action_space      = env.action_spaces[good_agent_name[0]]
    act_dim           = action_space.shape[0]
    if args.comm:
        comm_space    = env.communication_spaces[good_agent_name[0]]
        if ablate_kwargs is None or ablate_kwargs['adv_train']:
            obs_dim       = observation_space.shape[0]  + comm_space.shape[0]
        else:
            num_agents = len(good_agent_name) + len(adv_agent_name)
            print("comm shape:{}, num_agent:{}".format(comm_space.shape[0], num_agents))
            obs_dim       = observation_space.shape[0]  + \
                             comm_space.shape[0]*ablate_kwargs['k']//(num_agents-1)
    else:
        obs_dim = env.observation_spaces[good_agent_name[0]].shape 
    
    gym.logger.set_level(40)
    
    ### Set Environment Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    ### Set MADDPG Parameters
    kwargs = {"actor-lr":args.actor_lr,
              "critic-lr":args.critic_lr,
              "alpha":args.alpha,
              "noise-var":args.noise_var
             }
    
    
    capacity = int(5e+5)
    batch_size = 1000
    n_agents = len(good_agent_name)
    episodes_before_train = 60

    maddpg = MADDPG(n_agents, obs_dim, act_dim, batch_size, capacity,
                    episodes_before_train, kwargs=kwargs)

    avg_return = deque(maxlen=args.log_freq)
    for i_episode in range(args.episodes):
        good_total_rewards = np.zeros(len(good_agent_name))
        if args.comm:
            o, c = env.reset()
            o = concatenate(o,  c, ablate_kwargs)
        else:
            o = env.reset()
        
        obs = encode_obs(good_agent_name, obs_dim, o)
        for t in range(args.max_cycle): 
            good_actions = {}
            count  = 0
            action = maddpg.select_action(torch.from_numpy(obs).\
                        to(Param.device).type(Param.dtype)).data.cpu().numpy()
            for i in range(len(good_agent_name)):
                good_actions[good_agent_name[i]] = action[i,:]
                #print(action[i,:], flush=True)

            if args.comm:
                next_o, c, reward, done, infos = env.step(good_actions)
                next_o = concatenate(next_o, c, ablate_kwargs)
            else:
                next_o, reward, done, infos = env.step(good_actions)

            next_obs = encode_obs(good_agent_name, obs_dim, next_o, terminal = (t==args.max_cycle-1))
            reward_arr = torch.FloatTensor([reward[agent] for agent in good_agent_name])
            if args.render:
                env.render()

            ### Update the Memory
            #print("Observation:{}".format(obs[0]), flush=True)
            #print("Action:{}".format(action[0]), flush=True)
            #print("Next Observation:{}".format(next_obs[0]), flush=True)
            #print("Variance:{}".format(maddpg.var[0]), flush=True)
            
            maddpg.memory.push(torch.from_numpy(obs), torch.from_numpy(action), torch.from_numpy(next_obs), reward_arr)
            obs = next_obs

            ### Counting Reward
            for i in range(len(good_agent_name)):
                agent = good_agent_name[i]
                if good_agent_name[i] in env.agents:
                    good_total_rewards[i] += reward[agent]

            ### Adding towards Final Reward
            if (t == args.max_cycle - 1):
                i = int(re.match('pursuer_(\d+)', args.count_agent).group(1))
                ep_ret = good_total_rewards[i]
                avg_return.append(ep_ret)

            c_loss, a_loss = maddpg.update_policy()
        #print("c_loss:{}, a_loss:{}".format(c_loss, a_loss))
        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
        maddpg.episode_done += 1

        if (i_episode%args.log_freq == 0):
            print("----------------------Episode {}----------------------------".format(i_episode))
            logger_file.write("----------------------Episode {}----------------------------\n".format(i_episode))
            print("EpRet:{}".format(sum(avg_return)/len(avg_return)))
            logger_file.write("EpRet:{}\n".format(sum(avg_return)/len(avg_return)))
            maddpg.save(model_name='maddpg_{}'.format(args.exp_name))
            logger_file.flush()
    print("------------------------Finish Training-----------------------------")
    logger_file.close()
