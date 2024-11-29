import os
from pathlib import Path
from pandas import read_csv
# import from other files
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
path_main_app = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

"""
Search for a directory path, and if not found create the path (and the directory).
"""


class FindDirectory:
    def __init__(self, directory):
        self.directory = directory

    def create_directory(self):
        # find dynamically the current script directory
        full_path = os.path.join(path_main_app, self.directory)
        # create path directories if not exist --> else return the path
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        # print('Path {}:'.format(self.directory), full_path)
        return full_path


"""
Find 'logs' Directory
"""


class LogDirectory:
    # directory for logging path
    @staticmethod
    def log_directory():
        log_dir = 'logs'
        dir_path = os.path.join(path_main_app, log_dir) + '/'  # main app path + adding the 'logs' directory
        return dir_path


"""
Find 'exports' Directory
"""


class ExportsDirectory:
    # directory for logging path
    @staticmethod
    def exports_directory():
        exports_dir = 'exports'
        dir_path = os.path.join(path_main_app, exports_dir) + '/'  # main app path + adding the 'logs' directory
        # create path directories if not exist --> else return the path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path


"""
Find 'nn_configuration_files' Directory
"""


class NnConfDirectory:
    def __init__(self):
        self.init_conf_directory()
    
    def __str__(self):
        return self.path
    
    # directory for logging path
    def init_conf_directory(self):
        nn_conf_dir = 'config'
        dir_path = os.path.join(path_main_app, nn_conf_dir) + '/'  # main app path + adding the 'logs' directory
        self.path = dir_path
