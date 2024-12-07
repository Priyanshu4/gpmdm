from pathlib import Path
import sys

"""
This script is used to setup the imports for the notebooks.

It should be imported at the beginning of the notebook as follows:

import notebook_setup
"""

if __name__ == '__main__':
    raise Exception('This script should not be run as __main__')

if Path.cwd().name == 'notebooks':
    # If the current working directory is the notebooks directory, we need to go up one level
    # to be able to import the modules
    project_root = Path.cwd().parent
    sys.path.append(str(project_root))
else:
    project_root = Path.cwd()

PROJECT_ROOT_DIR = project_root
MODELS_DIR = PROJECT_ROOT_DIR / 'models'