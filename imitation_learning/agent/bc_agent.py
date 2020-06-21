import torch
from agent.networks import CNN

class BCAgent:
    
    def __init__(self, history_length=1):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.learning_rate = 1e-4
        self.net = CNN(history_length = history_length).cuda()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr= self.learning_rate)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        X_batch = torch.FloatTensor(X_batch) # it may not work if numpy float64
        X_batch = X_batch.permute(0,3,1,2).cuda()
        # or
        # X_batch = torch.Tensor(list(X_batch.values), requires_grad=True)
        y_batch = torch.FloatTensor(y_batch).cuda()
        # or
        # y_batch = torch.Tensor(list(y_batch.values), requires_grad=True)
        outputs = self.net(X_batch)

        self.net.train()
        self.optimizer.zero_grad()
        loss = self.loss (outputs, y_batch.squeeze(1).long())
        loss.backward()
        #Gradient clipping:
        clip = 1
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),clip)
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        self.net.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X)
            X = X.permute(0,3,1,2).cuda()
            # or
            # X = torch.Tensor(list(X.values), requires_grad=True)
            outputs = self.net(X)
        self.net.train()
        return outputs

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
