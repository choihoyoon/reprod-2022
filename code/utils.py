import os

def create_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise