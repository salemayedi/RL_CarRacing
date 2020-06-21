import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    # TODO: load DQN agent
    # ...
    state_dim = 4
    num_actions = 2
    Q = MLP (state_dim, num_actions)
    Q_target = MLP (state_dim, num_actions)
    PATH = '/home/salem/Documents/freiburg/Lab/CarRacing/reinforcement_learning/models_cartpole/best_eval_dqn_agent.pt'
    Q.load_state_dict(torch.load(PATH), strict=False)
    Q_target.load_state_dict(torch.load(PATH), strict=False)
    agent = DQNAgent(Q, Q_target, num_actions, double = True)
    n_test_episodes = 150

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent,eps =0.1, deterministic=True, do_training=False, rendering = True)
        print(stats.episode_reward)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

