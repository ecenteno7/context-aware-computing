#!/usr/bin/env python3
"""
Execution for Audio Classification experiment.
"""

# import scripts from other folders
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from helper_functions.read_yaml import ReadYml
from nn_utils.process_configuration import ConfigurationParameters
from nn_data_loader.ac_loader import FoldedAudioDataLoader
from neural_nets.audio_model import AudioClassifier
from nn_utils.process_argument_har import get_args

import soundata
from datetime import datetime
from pprint import pprint
from numpy.random import seed
seed(1)

def main():
    print('Time of NN train execution: {}'.format(datetime.now()))
    try:
        args = get_args("audio")
        config = ConfigurationParameters(args)
    except Exception as e:
	    print('Missing or invalid arguments!', e)
	    exit(0)

    # Load the dataset from the library, process and print its details.
    dataset = FoldedAudioDataLoader(config)
    
    if config.config_namespace.mode == 'save':
        model = AudioClassifier(config, dataset)
        model.save_model()
    else:
        print('Please, give a valid value (save / load)')
        print('or give a valid value for test set evaluation (true / false)')

def setup():
    nn_setup_yml = ReadYml('nn_setup.yml')
    nn_setup_conf = nn_setup_yml.load_yml()
    print('Running the following experiment...')
    pprint(nn_setup_conf)


if __name__=="__main__":
    setup()
    main()

