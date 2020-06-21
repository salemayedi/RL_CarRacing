#import tensorflow as tf
import torch
import numpy as np
from agent.replay_buffer import ReplayBuffer
import math
import pdb
import random 

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, double = False, gamma=0.95, batch_size=64, tau=0.01, lr=1e-4, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        #self.epsilon = epsilon

        # Double DQN
        self.double = double

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)

        # add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # sample batch
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        # result Q target
        batch_next_states = torch.from_numpy(batch_next_states).float().cuda()
        batch_states = torch.from_numpy(batch_states).float().cuda()
        batch_rewards = torch.from_numpy(batch_rewards).float().cuda().unsqueeze(1) #batch size * 1
        batch_dones = torch.from_numpy(batch_dones).float().cuda().unsqueeze(1)#batch size * 1
        #batch_actions = torch.from_numpy(batch_actions).long().cuda() #indices should be detached
        #print('Rewards: ', batch_rewards)

        # Double DQN:
        if self.double:
            self.Q_target.eval()
            self.Q.eval()
            with torch.no_grad():
                # get argmax actions of Q(next states)
                Q_next_output = self.Q(batch_next_states)
                Q_target_actions = torch.argmax(Q_next_output, dim = 1).unsqueeze(1)
                # compute Q_target(next states)
                Q_target_next_output = self.Q_target(batch_next_states)
                # compute Q_target(next states) with indices of argmax actions of Q(next states)
                
                #Q_target_next_output = Q_target_next_output.detach()
                
                Q_target_Double = Q_target_next_output[torch.arange(Q_target_next_output.shape[0]) , Q_target_actions.squeeze(1).detach()].unsqueeze(1)
                #Q_target_Double = Q_target_next_output.gather(1, Q_target_actions.long())
                td_target = batch_rewards + self.gamma * Q_target_Double* (1 - batch_dones)

        else:
            # DQN
            self.Q_target.eval()
            self.Q.eval()
            with torch.no_grad():
                Q_target_next_output = self.Q_target(batch_next_states)
                
                #Q_target_next_output = Q_target_next_output.detach()
                
                Q_target_values = torch.max(Q_target_next_output, dim=1)[0].unsqueeze(1)
                #print('Q_target_values: ', Q_target_values)
                td_target = batch_rewards + self.gamma * Q_target_values* (1 - batch_dones) # y 
        
        td_target = td_target.detach() # to make sure not backropagate, detach from pytorch graph
        # result Q
        self.Q.train() # train 
        Q_output_states = self.Q(batch_states)
        Q_output = Q_output_states[torch.arange(Q_output_states.shape[0]) , batch_actions].unsqueeze(1) #indices should be detached

            

        # to check if my parameters have gradients
        for name, param in self.Q.named_parameters():
            if param.grad is None:
                print('None ', name, param.grad)
            #else:
            #    print('not None ',name, param.grad.sum())
        
        loss = self.loss_function (Q_output, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        #Gradient clipping:
        clip = 1
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(),clip)
        self.optimizer.step()
        
        # soft update of Q_target
        soft_update(self.Q_target, self.Q, self.tau)


    def act(self, state, deterministic, epsilon):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            self.Q.eval()
            with torch.no_grad():
                state = torch.from_numpy(state).float().cuda().unsqueeze(0)
                Q_output = self.Q( state )
                action_id = torch.argmax(Q_output, dim = 1)
                action_id = action_id.item()
                #print('this is action_id: ', action_id)
            
            self.Q.train()
        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
            #print('####################else################')
            l = list(np.arange(self.num_actions))
            action_id = random.choice (l)
            #print('this is action_id: ', action_id)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
