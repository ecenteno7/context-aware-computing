"""
Process the json configuration file.

Configuration file holds the parameters to intialize the CNN model.
These files are located in configaration_files folder.
"""
import argparse
import json
from pprint import pprint
# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions.find_create_directory import NnConfDirectory
from helper_functions.read_yaml import ReadYml

def get_args(args):
    """
    Get arguments from the command line.

    :param none
    :return none
    :raises none
    """
    # read yml configuration for nn setup
    nn_setup_yml = ReadYml('nn_setup.yml')
    nn_setup_conf = nn_setup_yml.load_yml()

    # default NN JSONconfiguration path
    path_obj = NnConfDirectory()
    config_key = f'{args}_config'
    path_conf_default =  path_obj.nn_conf_directory() + nn_setup_conf.get(config_key)

    parser = argparse.ArgumentParser( description = __doc__ )

    # Configuration file path argument.
    parser.add_argument(
    					'-c', '--config',
    					metavar = 'C',
    					help = 'The Configuration file',
    					default = path_conf_default,
    					required = False
    				)

    # Epoch size argument.
    parser.add_argument(
    					'-e', '--epoch',
    					metavar = 'E',
    					help = 'Number of epoches for traning the model',
    					default = 100,
    					required = False
    				)

    # Mode argument (save / load model).
    parser.add_argument(
    					'-m', '--mode',
    					metavar = 'M',
    					help = 'Save (Train/Validate) or load the model (test/predict)',
    					default = 'save',
    					required = False
    				)

    # Test set evaluation (true / false).
    parser.add_argument(
    					'-t', '--testevaluation',
    					metavar = 'T',
    					help = 'Evaluate test due predictions (Load Mode)',
    					default = 'false',
    					required = False
    				)

    # Convert to dictonary.
    args = vars(parser.parse_args())

    if args['config'] == path_conf_default:
    	print('Using default configuration file.')
    	with open(path_conf_default) as json_file:
    		data = json.load(json_file)
    	print('Print NN configuration data:')
    	pprint(data)
    	print()
    else:
    	print('Using configurations from file:', args['config'])

    return args
