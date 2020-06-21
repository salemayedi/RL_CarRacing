import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif np.allclose(a,[0.0,0.0,0.2],atol = 1e-3) : return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])

def balance_data (X, y, drop_prob, id_action):
    """
    In the data, we are having a prblem of unbalanced data. the straight actions are so much compared to the other data.
    subsampling uniformly.
    """
    straight_action = np.zeros((1))
    straight_action [0] = id_action
    is_straight = np.all(y==straight_action, axis=1)
    # the rest 
    other_actions_index = np.where(np.logical_not(is_straight))[0]
    # random drop straights
    drop_mask = np.random.rand(len(is_straight)) > drop_prob
    straight_keep = drop_mask * is_straight
    # Get the index of straight samples that were kept
    straight_keep_index = np.where(straight_keep)[0]
    # Put all actions that we want to keep together
    final_keep = np.squeeze(np.hstack((other_actions_index, straight_keep_index)))
    final_keep = np.sort(final_keep)
    X_balanced = X[final_keep]
    y_balanced = y[final_keep]

    return X_balanced,y_balanced

def augment_acc_data (X, y, drop_prob = 0.31):
    """
    In the data, we are having a prblem of unbalanced data. the straight actions are so much compared to the other data.
    subsampling uniformly.
    """
    straight_action = np.zeros((1))
    straight_action [0] = 0
    acc_action = np.zeros((1))
    acc_action[0] = 3.0

    is_straight = np.all(y==straight_action, axis=1)
    # the rest 
    other_actions_index = np.where(np.logical_not(is_straight))[0]
    # random drop straights
    change_mask = np.random.rand(len(is_straight)) <= drop_prob
    new_actions = change_mask * is_straight
    # Get the index of straight samples that were kept
    new_action_index = np.where(new_actions)[0]
    # Put all actions that we want to keep together
    for i in range (len (new_action_index)):
        y[i] = acc_action
    return X,y
    

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
