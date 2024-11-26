#!/usr/bin/env python3
"""
Execution Flow for the PAT experiment.
"""
from datetime import datetime
from pprint import pprint
# Reproduce results by seed-ing the random number generator.
from numpy.random import seed
seed(1)

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

def main():
    
    print('Time of NN train execution: {}'.format(datetime.now()))
    
    try:

        # Capture the command line arguments from the interface script.
        args = get_args("audio")

        # Parse the configuration parameters for the ConvNet Model.
        config = ConfigurationParameters(args)

    except Exception as e:
	    print('Missing or invalid arguments!', e)
	    exit(0)

    # Load the dataset from the library, process and print its details.
    dataset = FoldedAudioDataLoader(config)
    print(dataset) 
    # train and save the model / load  the model for preditions
    if config.config_namespace.mode == 'save':
		# Construct, compile, train and evaluate the ConvNet Model.
        model = AudioClassifier(config, dataset)

        # Save the ConvNet model to the disk.
        model.save_model()
    else:
        print('Please, give a valid value (save / load)')
        print('or give a valid value for test set evaluation (true / false)')

def setup():
    # read yml configuration for nn setup
    nn_setup_yml = ReadYml('nn_setup.yml')
    nn_setup_conf = nn_setup_yml.load_yml()
    print('YML conf:')
    pprint(nn_setup_conf)


if __name__=="__main__":
    setup()
    main()




