import sys
sys.path.append("../") 

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP, MLP_Duel
from utils import EpisodeStats
from statistics import mean 
from collections import deque

def run_episode(env, agent, eps, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic, epsilon = eps)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)
            
        #print('action_id', action_id)
        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1
    return stats

def train_online(env, agent, num_episodes, epsilon_decay= False, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), "train",["episode_reward","len_episode","a_0", "a_1"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "train"), "eval",["performance_mean_episode_reward"])

    # training
    rewards = deque()
    rewards_eval = deque()
    early_stop = False
    eps_end = 0.06
    eps_decay = 0.995
    if epsilon_decay:
        eps = 1.0
    else:
        eps = 0.1

    for episode in range(num_episodes):
        print("episode: ",episode, ' with epsilon: ', eps)
        #epsilon decay
        if epsilon_decay:
            eps = max(eps_end, eps_decay*eps)

        stats = run_episode(env, agent, eps, deterministic=False, do_training=True, rendering = False)
        tensorboard.write_episode_data(episode, eval_dict={  "episode_reward" : stats.episode_reward,
                                                                "len_episode" : len(stats.actions_ids), 
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})
        #if (i <= 150):
        #    rewards.append(stats.episode_reward)
        #else:
        #    rewards.append(stats.episode_reward)
        #    rewards.popleft()
        #    if (len(rewards)>=150) and (mean(rewards) >= 250):
        #        early_stop = True     
        
        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        if episode % eval_cycle == 0:
            eval_episode_reward = []
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, eps,deterministic=True, do_training=True, rendering = False)
                eval_episode_reward.append(stats.episode_reward)
            eval_reward = mean(eval_episode_reward)
            if eval_reward > 900:
                agent.save(os.path.join(model_dir, "best_eval_dqn_agent.pt"))
            tensorboard_eval.write_episode_data(episode, eval_dict={  "performance_mean_episode_reward" : eval_reward })
            
            if (episode <= 100):
                rewards_eval.append(eval_reward)
            else:
                rewards_eval.append(eval_reward)
                rewards_eval.popleft()
                if (len(rewards_eval)>=10) and (mean(rewards_eval) >= 320):
                    early_stop = True  

        if early_stop == True:
            print ("We converged in episode: ", episode, " good job dude ;) ")
            agent.save(os.path.join(model_dir, "best_cartpole_dqn_agent.pt"))
            early_stop = False
            #break
        # store model.
        if episode % eval_cycle == 0 or episode >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
    tensorboard.close_session()
    tensorboard_eval.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 50         # evaluate every 10 episodes

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped
    #import pdb; pdb.set_trace()
    state_dim = 4
    num_actions = 2

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
    
    # Duelling DQN or Not
    Duel = False

    num_episodes = 2000
    
    if Duel:
        Q = MLP_Duel (state_dim, num_actions)
        Q_target = MLP_Duel (state_dim, num_actions) 
    else:
        Q = MLP (state_dim, num_actions)
        Q_target = MLP (state_dim, num_actions)
    
    DQNAgent = DQNAgent(Q, Q_target, num_actions, double = True, history_length = 1e6)
    train_online(env, DQNAgent, num_episodes, epsilon_decay= False)

 
