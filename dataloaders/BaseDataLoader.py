'''
This program implements a template dataloader.
'''


class Base_DataLoader(object):
    ''' A template dataLoader. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        pass

    def load_dataset(self):
        ''' Loads the dataset. '''
        raise NotImplementedError("Override this.")
