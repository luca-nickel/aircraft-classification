import torch


class Model:

    def __init__(self, net, criterion, optimizer):

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_gpu = False

    def to_cuda(self):
        if torch.cuda.is_available():
            self.net.to("cuda")
            self.criterion.to("cuda")
            self.use_gpu = True

    def _prepare_data(self, data, target=None):
        if self.use_gpu:
            data = data.to("cuda")
            if target is not None:
                target = target.to("cuda")

        return data, target

    def set_to_train(self):
        self.net.train()

    def set_to_val(self):
        self.net.eval()

    def train(self, data, target):
        data, target = self._prepare_data(data, target)

        # reset optimizer
        self.optimizer.zero_grad()

        # Forward step of net
        prediction = self.net.forward(data)

        # Loss calculation
        loss = self.criterion(prediction, target.long())

        # Backpropagation
        loss.backward()

        # Optimazation step
        self.optimizer.step()

        return loss.item()

    def predict(self, data, logging=None):
        data, _ = self._prepare_data(data)

        self.set_to_val()

        with torch.no_grad():
            # Forward step of net
            prediction = self.net.predict(data)

            # If a logger is available log the data
            if logging:
                logging.add(prediction)

            return prediction

    def validate(self, data, target):
        data, target = self._prepare_data(data, target)

        # Forward step of net
        with torch.no_grad():
            prediction = self.net.forward(data)

            # Loss calculation
            loss = self.criterion(prediction, target.long())
            acc = self.get_accuracy(data, target, prediction)
            return loss, acc

    def save_net(self, save_location):
        try:
            state_dict = self.net.state_dict()
        except AttributeError:
            state_dict = self.net.module.state_dict()

        torch.save(state_dict, save_location)

    def get_accuracy(self, data, target, prediction=None):
        data, target = self._prepare_data(data, target)
        if prediction is None:
            prediction = self.predict(data)
            _, predicted = torch.max(prediction.data, 1)
            correct = (predicted == target).sum().item()
            return correct / target.size(0)

        prediction = torch.nn.Softmax(dim=1)(prediction)
        _, predicted = torch.max(prediction.data, 1)
        correct = (predicted == target).sum().item()
        return correct / target.size(0)
