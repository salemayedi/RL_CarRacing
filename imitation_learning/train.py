from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

import torch

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    print('X_train.shape: ', X_train.shape)
    print('y_train: ', y_train)
    y_train = action_to_id(y_train)
    y_valid = action_to_id(y_valid)
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    #for i in history_length:
     #   X_train = 
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent ()
    
    tensorboard_eval = Evaluation(tensorboard_dir)
    #tensorboard_eval = Evaluation(tensorboard_dir, "train",["loss_train", "loss_valid", "accuracy_train", "accuracy_valid"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
      
    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)
    def sample_minibatch(X_train, y_train, permutation, i_n_minibatches):
        indices = permutation[i_n_minibatches:i_n_minibatches+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        return batch_x, batch_y
    
    permutation = torch.randperm(X_train.size()[0])

    # training loop
    for i in range(n_minibatches):
        X, y = sample_minibatch (X_train, y_train, permutation, i)
        agent.update(X, y)
        # output
        outputs_train = agent.predit (X_train)
        outputs_valid = agent.predit (X_valid)
        # loss
        loss_train = torch.nn.CrossEntropyLoss(X_train, y_train)
        loss_valid = torch.nn.CrossEntropyLoss(X_valid, y_valid)
        # labels_output
        labels_train = torch.nn.functional.softmax(outputs_train, dim=1).argmax(dim=1)  # check dimension
        labels_valid = torch.nn.functional.softmax(outputs_valid, dim=1).argmax(dim=1)  # check dimension
        # accuracy
        corrects_train = (labels_train == y_train)
        corrects_valid = (labels_valid == y_valid)
        accuracy_train = corrects_train.sum().float() / float( y_train.shape[0])
        accuracy_valid = corrects_valid.sum().float() / float( y_valid.shape[0])
        tensorboard_eval.write_episode_data(i, eval_dict={  "loss_train" : loss_train,
                                                                "loss_valid" : loss_valid,
                                                                "accuracy_train" : accuracy_train,
                                                                "accuracy_valid" : accuracy_valid})



if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=64, lr=1e-4)
 
