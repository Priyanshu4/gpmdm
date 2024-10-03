""" Parse and visualize ASF and AMC files. 
    Requires an ASF and AMC file as input.
    Use the spacebar to pause and play the animation.
"""

from .amc_parser import parse_asf, parse_amc
from .viewer import Viewer
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
  
    parser = ArgumentParser(prog='amc_parser', description=__doc__)
    parser.add_argument('asf_path', type=str)
    parser.add_argument('amc_path', type=str)
    args = parser.parse_args()

    asf_path = Path(args.asf_path)
    amc_path = Path(args.amc_path)

    joints = parse_asf(args.asf_path)
    motions = parse_amc(args.amc_path)
    v = Viewer(joints, motions)
    v.run()