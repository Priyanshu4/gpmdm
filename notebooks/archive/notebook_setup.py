from pathlib import Path
import sys

"""
This script is used to setup the imports for the notebooks.

It should be imported at the beginning of the notebook as follows:

import nb_setup
"""

if __name__ == '__main__':
    raise Exception('This script should not be run as __main__')

if Path.cwd().name == "archive" and Path.cwd().parent.name == 'notebooks' :
    # If the current working directory is the archive directory, we need to go up two levels
    # to be able to import the modules
    project_root = Path.cwd().parent.parent
    sys.path.append(str(project_root))

