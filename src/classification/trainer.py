import numpy as np
import torch
import torch.utils.data as torchData
from classification.loggers.logger import Logger


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        seed=69,
        gpu=True,
        name="",
        batch_size=1,
        dataholder_str="",
    ):

        # Save every variable in the class state
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.gpu = gpu and torch.cuda.is_available()
        # Initialize logger
        self.logger = Logger(name)

        # Initialize seeds, to make the training predictable
        torch.manual_seed(self.seed)
        if self.gpu:
            torch.cuda.manual_seed(self.seed)

        # Initialize cuda/gpu work
        self.gpu_setup()

        # Prepare dataset
        self.train_loader, self.validation_loader = self.prepare_dataset(batch_size)

        print(self.model.predict(next(iter(self.validation_loader))[0])[0])

        # Write first important information
        self.logger.summary("dataholder", dataholder_str)
        self.logger.model_text(self.model.net)
        self.logger.summary("seed", self.seed)

        if self.gpu:
            self.logger.model_log(
                self.model.net, next(iter(self.train_loader))[0].to("cuda")
            )
        else:
            self.logger.model_log(self.model.net, next(iter(self.train_loader))[0])

    def prepare_dataset(self, batch_size):
        train_dataset, validation_dataset = self.dataset[0], self.dataset[1]

        train_loader = torchData.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_loader = torchData.DataLoader(
            validation_dataset, batch_size=batch_size
        )

        return train_loader, validation_loader

    def gpu_setup(self):
        if self.gpu:
            print("[TRAINER]: Write model and criterion on GPU")
            self.model.to_cuda()

        # Log if cpu or gpu is used
        self.logger.log_string("device usage", ("GPU" if self.gpu else "CPU"))

    def calc_step(self, data, target):
        # Loss calculation
        loss = self.model.train(data, target)

        return loss

    def calc_epoch(self):
        epoch_loss = 0.0
        epoch_acc = 0.0
        step_count = 0
        step_count_acc = 0

        self.model.set_to_train()
        # self.dataset.set_validation(False)

        for _ in range(self.inflation):
            # Loop over hole dataset
            for mini_batch, target in self.train_loader:
                # Calculate a training step
                loss = self.calc_step(mini_batch, target)
                epoch_loss += loss
                step_count += 1
        self.model.set_to_val()
        for mini_batch, target in self.train_loader:
            # Calculate a training step
            epoch_acc += self.model.get_accuracy(mini_batch, target)
            step_count_acc += 1

        return epoch_loss / step_count, epoch_acc / step_count_acc

    def train(self, epochs, patience=0, inflation=1):
        # log start of training
        self.logger.train_start()

        # Store inflation rate of trainingsdata
        self.inflation = inflation

        min_loss = float("inf")
        cur_patience = 0
        finish_reason = "Training did not start"

        for epoch in range(epochs):
            self.logger.set_step(epoch + 1)
            # Try block for catching keyboard interrupt
            try:

                train_loss, train_acc = self.calc_epoch()
                # Log train_loss
                self.logger.train_loss(train_loss)
                self.logger.train_acc(train_acc)

                # After calculating an epoch, validate for taking statistics
                validation_loss, validation_acc = self.validate()
                # Log validateion_loss
                self.logger.val_loss(validation_loss)
                self.logger.val_acc(validation_acc)

                # Early stopping
                if min_loss > validation_loss:
                    min_loss = validation_loss
                    # min_loss_epoch = epoch
                    cur_patience = 0
                    # Save the current best model
                    self.model.save_net(self.logger.get_model_save_location())
                else:
                    if patience > 0:
                        cur_patience += 1
                        if cur_patience == patience:
                            finish_reason = (
                                "Training finished because of early stopping"
                            )
                            break

            except KeyboardInterrupt:
                # In case the user wants to interrupt the training
                finish_reason = "Training interrupted by user"
                break
            else:
                finish_reason = "Training finished normally"

        # Log finish
        self.logger.train_end(finish_reason)

        # Training Cleanup
        self.logger.close()

    def validate(self):
        # epoch variable should I need the index for visualization logging
        # Running loop for whole dataset

        valid_loss = 0.0
        valid_acc = 0.0
        steps = 0

        # Don't calculate the gradients
        for data, target in self.validation_loader:
            # Calc loss
            loss, acc = self.model.validate(data, target.long())

            valid_loss += loss.sum().item()
            valid_acc += acc

            steps += 1

        return valid_loss / steps, valid_acc / steps
