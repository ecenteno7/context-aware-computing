"""
Parse the JSON configuration file of the experiment.

Configuration file holds the parameters to intialize the ConvNet model.
These files are located in configaration_files folder.
"""
import json
import os
from bunch import Bunch
# Main Application directory
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

from helper_functions.read_yaml import ReadYml
from helper_functions.find_create_directory import NnConfDirectory

class ConfigurationParameters:

    def __init__(self):
        """
        Intialize the data members.

        :param json_file: Path to the JSON configuration file.
        :return none
        :raises none
        """
        # read yml configuration for nn setup
        setup_yml = ReadYml('setup.yml')
        config_file_name = setup_yml.load_yml().get('config_file')
        conf_dir = NnConfDirectory()
        config_file_path = os.path.join(f"{conf_dir}", config_file_name)

        with open(config_file_path, 'r') as config_file:
            self.config_dictionary = json.load(config_file)

        # Convert the dictionary to a namespace using bunch library.
        self.config_namespace = Bunch(self.config_dictionary)

        # Process the configuration parameters.
        self.process_config()

        return

    def process_config(self):
        """
        Processes the configuration parameters of the ConvNet experiment.

        :param none
        :return none
        :raises none
        """
        self.config_namespace.dataset_dir = os.path.join(main_app_path, "datasets", self.config_namespace.exp_name)

        # Saved-Model directory.
        self.config_namespace.saved_model_dir = os.path.join(
            main_app_path, "experiments", self.config_namespace.exp_name, "saved_models/")

        # Graph directory.
        self.config_namespace.graph_dir = os.path.join(
            main_app_path, "experiments", self.config_namespace.exp_name, "graphs/")

        # Image directory.
        self.config_namespace.image_dir = os.path.join(
            main_app_path, "experiments", self.config_namespace.exp_name, "images/")

        # DataFrame directory.
        self.config_namespace.df_dir = os.path.join(
            main_app_path, "experiments", self.config_namespace.exp_name, "dataframes/")

        # Classification Report directory.
        self.config_namespace.cr_dir = os.path.join(
            main_app_path, "experiments", self.config_namespace.exp_name, "class_reports/")

        # Create the above directories.
        self.create_dirs([self.config_namespace.graph_dir,
                          self.config_namespace.image_dir,
                          self.config_namespace.saved_model_dir,
                          self.config_namespace.df_dir,
                          self.config_namespace.cr_dir])

        return

    def create_dirs(self, dirs):
        """
        Creates a directory structure for Graphs and Images generated during the run of the experiment.

        :param dirs: a list of directories to create if these directories are not found
        :return exit_code: 0:success -1:failed
        :raises none
        """

        try:
            for d in dirs:
                if not os.path.exists(d):
                    os.makedirs(d)
            return 0

        except Exception as err:
            print("Creating directories error: {0}".format(err))
            exit(-1)
