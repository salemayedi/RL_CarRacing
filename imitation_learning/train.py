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
import pdb

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
    
    X_train = rgb2gray(X_train)/255.0
    X_valid = rgb2gray(X_valid)/255.0
    
    
    # action_to Id
    y_actions_train = np.zeros((y_train.shape[0], 1))
    y_actions_val = np.zeros((y_valid.shape[0], 1))
    for i in range ( len (y_train)):
        y_actions_train[i] = action_to_id(y_train[i]) 
    for i in range( len (y_valid)):
        y_actions_val[i] = action_to_id(y_valid[i]) 
    y_train = y_actions_train
    y_valid = y_actions_val

    # balance X_train and y_train data
    X_train, y_train = balance_data (X_train, y_train, 0.375, 1) # drop probability:0.5 for left action
    X_train, y_train = balance_data (X_train, y_train, 0.85, 0) # drop probability:0.885 for left action
    X_train, y_train = augment_acc_data(X_train, y_train) # augment acceleration
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    #for i in history_length:
     #   X_train =
    
    # first image Train
    print('X_train.shape: ', X_train.shape)
    X_train_history = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], history_length ))
    X_valid_history = np.zeros((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], history_length ))
    
    image_hist = []
    image_hist.extend([X_train[0]] * (history_length))
    X_train_history[0] = np.array(image_hist).reshape(96, 96, history_length )
    # rest of images Train
    for i in range ( 1, len(X_train)):
        image_hist.append(X_train[i])
        image_hist.pop(0)
        X_train_history[i] = np.array(image_hist).reshape(96, 96, history_length)
    
    # first image Val
    image_hist = []
    image_hist.extend([X_valid[0]] * (history_length))
    X_valid_history[0] = np.array(image_hist).reshape(96, 96, history_length )
    # rest of images Val
    for i in range ( 1, len(X_valid)):
        image_hist.append(X_valid[i])
        image_hist.pop(0)
        X_valid_history[i] = np.array(image_hist).reshape(96, 96, history_length)
    
    X_train = X_train_history
    X_valid = X_valid_history
    _ = plt.hist(y_train, bins='auto') 
    plt.savefig('action_dist.png')
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, history_length, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent (history_length)
    
    #tensorboard_eval = Evaluation(tensorboard_dir)
    tensorboard_eval = Evaluation(tensorboard_dir, "train",["loss_train", "loss_valid", "accuracy_train", "accuracy_valid"])

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
    
    permutation = torch.randperm(X_train.shape[0])

    # training loop
    for i in range(n_minibatches):
        X, y = sample_minibatch (X_train, y_train, permutation, i)
        #print('episode: ', i)
        agent.update(X, y)
        # output
        outputs_train = agent.predict (X)
        outputs_valid = agent.predict (X_valid)
        # loss
        loss_train = torch.nn.CrossEntropyLoss()(outputs_train, torch.LongTensor(y).cuda().squeeze(1))
        loss_valid = torch.nn.CrossEntropyLoss()(outputs_valid, torch.LongTensor(y_valid).cuda().squeeze(1))
        # labels_output
        labels_train = torch.nn.functional.softmax(outputs_train, dim=1).argmax(dim=1)  # check dimension
        labels_valid = torch.nn.functional.softmax(outputs_valid, dim=1).argmax(dim=1)  # check dimension
        # accuracy
        if i % 10 == 0 :
            print('episode: ', i)
            corrects_train = (labels_train == torch.LongTensor(y).cuda().squeeze(1))
            corrects_valid = (labels_valid == torch.LongTensor(y_valid).cuda().squeeze(1))
            accuracy_train = corrects_train.sum().float() / float( len(y))
            accuracy_valid = corrects_valid.sum().float() / float( len(y_valid))
            tensorboard_eval.write_episode_data(i, eval_dict={  "loss_train" : loss_train.item(),
                                                                    "loss_valid" : loss_valid.item(),
                                                                    "accuracy_train" : accuracy_train.item(),
                                                                    "accuracy_valid" : accuracy_valid.item()})
            #print('episode: ', i, ' acc train: ', accuracy_train.item(), ' loss train: ', loss_train.item(), \
             #       'acc val: ', accuracy_valid.item(), ' loss val: ', loss_valid.item())
            agent.save(os.path.join(model_dir, "agent.pt"))


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    history_length=3
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, history_length, n_minibatches=1000, batch_size=64, lr=1e-4)
 
