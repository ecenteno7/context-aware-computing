#!/usr/bin/env python3
"""
Execution for Audio Classification experiment.
"""
from datetime import datetime
from numpy.random import seed
seed(1)

# import scripts from other folders
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from nn_utils.process_configuration import ConfigurationParameters
from nn_data_loader.ac_loader import FoldedAudioDataLoader
from classifiers.audio_classifier import AudioClassifier

def load_model(config, dataset):
    model = None
    if config.config_namespace.mode == 'save':
        print("Loading model...need to train it up first though.")
        print("Please wait. This will take about 4-5 hours.")
        model = AudioClassifier(config, dataset)
        model.save_model()
    elif config.config_namespace.mode == 'load':
        print("Loading model from file...")
        model = AudioClassifier(config, dataset, load=True)
    else:
        print('Please, give a valid value (save / load)')
        print('or give a valid value for test set evaluation (true / false)')
    return model 

def setup_experiment():
    print('Time of NN train execution: {}'.format(datetime.now()))
    
    config = ConfigurationParameters()
    dataset = FoldedAudioDataLoader(config)
    
    print(f'Running the following experiment: {config.config_namespace.exp_name}')
    
    return config, dataset 

def experiment_downloaded_file(model):
    file_name = 'children_playing.wav'
    downloaded_file_root = os.path.join('/home', 'erik', 'Downloads')
    predict_file = os.path.join(downloaded_file_root, file_name)

    prediction = model.predict(predict_file)
    print(f"Sounds like: {prediction}")

def experiment_urbansound8k_file(model):
    file_name = '209992-5-2-138.wav'
    fold = 'fold7'
    urbansound8k_file_root = os.path.join(model.config.config_namespace.dataset_dir, fold)
    predict_file = os.path.join(urbansound8k_file_root, file_name)

    prediction = model.predict(predict_file)
    print(f"Sounds like: {prediction}")


if __name__=="__main__":
    config, dataset = setup_experiment()
    model = load_model(config, dataset)
    
    experiment_urbansound8k_file(model) 
