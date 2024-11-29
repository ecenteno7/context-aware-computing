import os
import numpy as np

class BaseNeuralNet():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
    
    def define_model(self):
        """
        Constructs the ConvNet model.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to add layers to the ConvNet.
        raise NotImplementedError

    def compile_model(self):
        """
        Configures the ConvNet model.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to configure the ConvNet model.
        raise NotImplementedError

    def fit_model(self):
        """
        Trains the ConvNet model.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to configure the ConvNet model.
        raise NotImplementedError

    def evaluate_model(self):
        """
        Evaluates the ConvNet model.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to evaluate the constructed ConvNet model.
        raise NotImplementedError

    def predict(self):
        """
        Predicts the class labels of unknown data.
        :param none
        :return none
        :raises NotImplementedError: Exception: Implement this method.
        """

        # Implement this method in the inherited class to predict the class-labels of unknown data.
        raise NotImplementedError

    def save_model(self):
        """
        Saves the ConvNet model to disk in h5 format.
        :param none
        :return none
        """

        if self.model is None:
            raise Exception("Model not configured and trained !")
        try:
            self.model.save(self.saved_model_path)
        except AttributeError:
            torch.save(self.model.state_dict(), self.saved_model_path)
        print("Model saved at path: ", self.saved_model_path, "\n")

        return

    def load_model(self):
        """
        Loads the saved model from the disk.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        if self.model is None:

            raise Exception("Model not configured and trained !")

        self.model.load_weights(self.saved_model_path)
        print("Model loaded from the path: ", self.saved_model_path, "\n")

        return


