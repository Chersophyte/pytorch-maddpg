def encode_obs(good_agent_name, obs_dim, obs, terminal=False):
    obs_array = np.zeros((n_agents, obs_dim))
    for i in range(len(good_agent_name)):
        agent = good_agent_name[i]
        if not terminal:
            obs_array[i] = obs[agent]
        else:
            obs_array[i] = terminal
    return obs_array
        

good_total_rewards = np.zeros(len(good_agent_name))
avg_return = deque(maxlen=args.log_freq)
for i_episode in range(args.episodes):
    o = env.reset()
    obs = encode_obs(good_agent_name, obs_dim, o)
    for t in range(args.max_cycle): 
        good_actions = {}
        if comm:
            o, c = env.reset()
            o = concatenate(o,  c, ablate_kwargs)
        else:
            o = env.reset()
        next_obs = np.zeros((n_agents, obs_dim))
        action = np.zeros((n_agents, act_dim))
        count  = 0
        for agent in good_agent_name:
            a = maddpg.select_action(torch.from_numpy(o[agent]).\
                            to(Param.device).type(Param.dtype)).data
            if not dist_action:
                a = a.cpu().numpy()
            else:
                a = a.item()
            good_actions[agent] = a
            action[count, :] = a
            count += 1
            
        if comm:
            next_o, c, reward, done, infos = env.step(good_actions)
            next_o = concatenate(next_o, c, ablate_kwargs)
        else:
            next_o, reward, done, infos = env.step(good_actions)
        
        next_obs = encode_obs(good_agent_name, obs_dim, next_o, (terminal = (t==args.max_cycle-1)))
                
        if args.render:
            env.render()
        
        ### Update the Memory
        maddpg.memory.push(obs, action, next_obs, reward)
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
        
        ### Update Policy
        if (t%20 == 0):
            c_loss, a_loss = maddpg.update_policy()
            
    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
    maddpg.episode_done += 1
    
    if (i_episode%args.log_freq == 0):
        print("----------------------Episode {}----------------------------".format(i_episode))
        logger_file.write("----------------------Epoch {}----------------------------\n".format(epoch))
        print("EpRet:{}".format(sum(avg_return)/len(avg_return)))
        logger_file.write("EpRet:{}\n".format(sum(avg_return)/len(avg_return)))
    