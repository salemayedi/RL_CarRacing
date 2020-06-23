from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json

import torch
from agent.bc_agent import BCAgent
from utils import *
from collections import deque


def check_freeze (act_hist, action):        
    first_action = act_hist[0]
    for a in act_hist:
        if a!=first_action: # check if all the actions are the same in the last 100 actions
            return False
        if int(a) == 3: # check if we had an acceleration in the last 100 actions
            return False
    # if we came here, it means that we are stuck somewhere
    return True

def check_stcuk (act_hist, action):
    if action == 1 or action == 2:
        return False
    
    if act_hist.count(3)>10: # check if we had an acceleration in the last 100 actions
        return False
    # if we came here, it means that we are stuck somewhere
    return True

def run_episode(env, agent, history_length, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 
    image_hist = []
    # action history, we are going to accelerate if we freeze
    max_len_action_history = 50
    action_history = deque(maxlen = max_len_action_history)


    state = rgb2gray(state).reshape(96, 96) / 255.0
    image_hist.extend([state] * (history_length))
    state = np.array(image_hist).reshape(96, 96, history_length)
    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...
        # import pdb; pdb.set_trace()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        action = agent.predict(state)
        action = torch.argmax(action, dim = 1).item()
        
        # append in action history 
        if step < max_len_action_history:
            action_history.append(action)
        else:
            if check_stcuk (action_history, action):
                #print('the network freezed')
                action = 3 # if we freeze, we accelerate
            action_history.append(action)
            action_history.popleft()
            
            
        
        # Continue
        action = id_to_action(action )
        next_state, r, done, info = env.step(action)   
        episode_reward += r
        
        # transform next state
        next_state = rgb2gray(next_state).reshape(96, 96) / 255.0
        image_hist.append(next_state)
        image_hist.pop(0)
#        import pdb; pdb.set_trace()
        next_state = np.array(image_hist).reshape(96, 96, history_length)

        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            print('############## Episode Reward: ', episode_reward)
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")
    history_length = 3
    agent = BCAgent (history_length)
    agent.load("/home/salem/Documents/freiburg/Lab/CarRacing/imitation_learning/models/agent_best.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_length, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
