
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.trigger = 0
        self.last_loss = 100

    def step(self, ave_valid_loss):

        if ave_valid_loss > self.last_loss:
            self.trigger += 1
        else:
            self.trigger = 0
        self.last_loss = ave_valid_loss

    def check_patience(self):
        if self.trigger >= self.patience:
            return True
        else:
            return False
