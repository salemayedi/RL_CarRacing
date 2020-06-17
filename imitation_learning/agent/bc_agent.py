import torch
from agent.networks import CNN

class BCAgent:
    
    def __init__(self):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.learning_rate = 1e-4
        self.net = CNN()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.paramaters(), lr= self.learning_rate)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        X_batch = torch.from_numpy(X_batch) # it may not work if numpy float64
        # or
        # X_batch = torch.Tensor(list(X_batch.values), requires_grad=True)
        y_batch = torch.from_numpy(y_batch)
        # or
        # y_batch = torch.Tensor(list(y_batch.values), requires_grad=True)
        outputs = self.net(X_batch)

        self.optimizer.zero_grad()
        loss = self.loss (outputs, y_batch)
        self.loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        with torch.no_grad():
            X = torch.from_numpy(X)
            # or
            # X = torch.Tensor(list(X.values), requires_grad=True)
            outputs = self.net(X)
        return outputs

    def load(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def save(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
