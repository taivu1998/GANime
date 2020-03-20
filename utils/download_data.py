'''
This program downloads the dataset from Kaggle.
'''

import kaggle

def download_dataset():
    ''' Downloads the dataset. '''
    DATA_NAME = 'ktaebum/anime-sketch-colorization-pair'
    DATA_PATH = './data'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATA_NAME, path = DATA_PATH, unzip = True)


if __name__ == '__main__':
    download_dataset()
