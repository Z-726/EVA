import os
import logging
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import utils
from tcp import trajectory_central_planning

logging.basicConfig(level=20)

def train(env, qnet, target_net, optimizer, replay, value_buffer, config, device, envpool):
    """
    Performs loop for a train step for EVA
    """
    n_actions = env.action_space.n
    eval_rewards = []
    total_rewards = []
    eval_global_steps = []
    total_global_steps = []
    global_step = 0
    logging.info("training started...")
    
    def full_filename(filename):
        return os.path.normpath('/'.join([config.save_dir, filename]))

    def save_results(episode):
        with open (full_filename('rewards.pkl'), 'wb') as rew_file:
            pickle.dump({'eval_rewards' : eval_rewards,
                         'eval_global_steps' : eval_global_steps,
                         'total_rewards' : total_rewards,
                         'total_global_steps' : total_global_steps},
                         rew_file
                    )
        # with open (full_filename('replay_buffer.pkl'), 'wb') as replay_file:
        #     pickle.dump(replay, replay_file)

        with open (full_filename('value_buffer_{}.pkl'.format(episode)), 'wb') as vb_file:
            pickle.dump(value_buffer, vb_file)
        torch.save(qnet.state_dict(), full_filename('qnet_state_dict_{}.pkl'.format(episode)))
        torch.save(qnet, full_filename('qnet.pkl'))      
    def choose_action_embedding(state, epsilon):
        """
        EVA policy is dictated by the action-value function 
        
        Q(s, a) = lambda * Q_param(s, a) + (1 - lambda) * QNP(s, a)
        
        we choose action based on the argmax of this Q value function.
        With probablity epsilon action is taken randomly
        
        Input: state 
        return: action, embedding for state
        """

        q_param, embedding = qnet(torch.FloatTensor(state).to(device).unsqueeze(0))
        q_param            = q_param.detach().cpu().squeeze()
        embedding          = embedding.detach().cpu().squeeze().numpy()
        
        # Chooses random action with probability epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            if len(value_buffer) == value_buffer.capacity:
                q_param = config.lambd * q_param + (1 - config.lambd) * value_buffer.nn_qnp_mean(
                          embedding, n_neighbors=config.n_neighbors_value_buffer)
            action = torch.argmax(q_param).item()
        return action, embedding

    def step(action):
        """returns next state and revard based on action"""
        next_state, reward, is_terminal,_ = env.step(action)
        if envpool:
            next_state = np.squeeze(next_state)
        if is_terminal:
            next_state = None
        return next_state, reward, is_terminal

    def train_step():
        """
        train step per batch, single step of the optimisation as in DQN model.
        Implementation is inspired by official pytorch turorial for DQN
        https://pytorch.org/tutorials/
        """
        # Take state, action, reward, next_state per batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch = zip(*replay.sample(config.batch_size))
        state_batch      = torch.from_numpy(np.stack(state_batch)).to(device)
        action_batch     = torch.LongTensor(action_batch).view(-1,1).to(device)
        
        # Compute mask of non-terminal states
        non_final_mask   = torch.BoolTensor([next_state_batch_i is not None 
                                             for next_state_batch_i in next_state_batch]).view(-1).to(device)

        # concatenate non terminal state batch elements
        next_state_batch = torch.from_numpy(np.stack([next_state_batch_i 
                                                      for next_state_batch_i in next_state_batch 
                                                      if next_state_batch_i is not None])).to(device)

        reward_batch     = torch.FloatTensor(reward_batch).view(-1,1).to(device)

        # Clean optimizer
        optimizer.zero_grad()
        
        # Compute Q values for all next states. 
        # Expected values for non-terminal next states are computed based on older target net. 
        # The use of mask helps to have expected state value or 0 for terminal state
        q_predicted              = qnet(state_batch.float())[0].gather(1, action_batch)
        q_target                 = torch.zeros_like(q_predicted)
        q_target[non_final_mask] = target_net(next_state_batch.float())[0].max(1, keepdim=True)[0].detach()
        
        # Compute expected Q values and optimize the model
        loss = F.mse_loss(q_predicted, q_target * config.gamma + reward_batch)
        loss.backward()
        optimizer.step()
    
    def evaluate(episode, n_episodes = 10):
        """
        Runs n_episodes episodes with epsilon=0 and returns mean reward.
        """
        if config.write_video:
            state_frames = []

        episode_rewards = []
        for _ in range(n_episodes):


            state = env.reset()
            if envpool:
                state = np.squeeze(state)
            is_terminal = False
            episode_reward = 0.
            for _ in range(config.t_max):
                action, _ = choose_action_embedding(state, epsilon=0)
                if envpool:
                    action_list=[]
                    action_list.append(action)
                    action = np.array(action_list)
                state, reward, is_terminal = step(action)
                if config.write_video and not is_terminal:
                    state_frames.append(state[0])
                episode_reward += reward
                if is_terminal:
                    episode_rewards.append(episode_reward)
                    break
            
        if config.write_video:
            from moviepy import editor
            state_frames = np.stack(state_frames, axis=0) * 255.
            state_frames = state_frames.astype(np.uint8)
            # convert image from grey space to rgb (img still remains grey)
            state_frames = np.repeat(state_frames[:, :, :, np.newaxis], 3, axis=3)
            state_frames = [frame for frame in state_frames]
            movie = editor.ImageSequenceClip(state_frames, fps=30)
            movie.write_videofile(full_filename('movie_{}.mp4'.format(episode)), 
                                  verbose=False, codec='mpeg4', logger=None)

        return np.mean(episode_rewards)

    os.makedirs(config.save_dir, exist_ok=True)
    episode = 0
    while global_step < config.n_episodes:
        state = env.reset()
        if envpool:
            state = np.squeeze(state)
        is_terminal = False
        episode_reward = 0
        
        # main loop for playing one episode
        for _ in range(config.t_max):
            action, embedding  = choose_action_embedding( state, 
                                                          utils.epsilon(global_step, config) )
            if envpool:
                action_list=[]
                action_list.append(action)
                action = np.array(action_list)
            next_state, reward, is_terminal = step(action)
            replay.push(state, action, reward, next_state, embedding)        
            state = next_state
            episode_reward += reward 
            
            if len(replay) >= config.batch_size:           
                train_step() 
            
            # we call trajectory central planning with TCP frequency and when replay is full
            if ((global_step % config.tcp_frequency) == 0) and  (len(replay) == replay.capacity) :           
                trajectory_central_planning(embedding, replay, value_buffer, qnet, config, device)
            # update of target net periodically
            global_step +=1           
            if (global_step % config.t_update) == 0:
                target_net.load_state_dict(qnet.state_dict())
            
            if is_terminal:
                total_rewards.append(episode_reward)
                total_global_steps.append(global_step)
                episode += 1
                break

            if global_step % config.eval_freq == 0:
                eval_rewards.append(evaluate(episode))
                logging.info("Episode: {}    mean_eval_reward = {}".format(episode, eval_rewards[-1]))
                eval_global_steps.append(global_step) 



    return eval_rewards