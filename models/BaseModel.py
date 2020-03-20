'''
This program implements a template model.
'''


class BaseModel(object):
    ''' A template model. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        pass

    def build_model(self):
        ''' Builds network architectures. '''
        raise NotImplementedError("Override this.")
    
    def configure_losses(self):
        ''' Configures losses. '''
        raise NotImplementedError("Override this.")

    def configure_optimizers(self):
        ''' Configures optimizers. '''
        raise NotImplementedError("Override this.")
    
    def configure_checkpoints(self):
        ''' Configures checkpoints. '''
        raise NotImplementedError("Override this.")
    
    def configure_logs(self):
        ''' Configures logs. '''
        raise NotImplementedError("Override this.")
    
    def fit(self):
        ''' Trains the model. '''
        raise NotImplementedError("Override this.")
    
    def train_step(self):
        ''' Executes a training step. '''
        raise NotImplementedError("Override this.")
    
    def save_output(self):
        ''' Saves output images for each epoch. '''
        raise NotImplementedError("Override this.")
    
    def predict(self):
        ''' Generates an output image from an input. '''
        raise NotImplementedError("Override this.")
    
    def load_checkpoints(self):
        ''' Loads the latest checkpoint. '''
        raise NotImplementedError("Override this.")
    
    def plot_model(self):
        ''' Visualizes the network architectures. '''
        raise NotImplementedError("Override this.")
