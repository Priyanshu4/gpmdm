from amc_parser import parse_asf, parse_amc
from .viewer import Viewer
import pandas as pd
import numpy as np

from typing import Optional

class MotionCapture:

    def __init__(self, asf_path, amc_path, fps=120, subject: Optional[int] = None, trial: Optional[int] = None):
        """
        """
        self._joints = parse_asf(asf_path)
        self._frames = parse_amc(amc_path)
        self._fps = fps
        self.subject = subject
        self.trial = trial

    def as_dataframe(self):
        """
        Return motion sequence as pandas DataFrame.
        """
        df = pd.DataFrame(index = range(len(self._frames)))
        
        # Add time and frame column
        df['frame'] = df.index
        df['time'] = df.index / self._fps
        
        for joint_name, joint in self.joints.items():
            
            # Add columns for each joint
            joint_columns = self.get_columns_for_joint(joint_name)
            for column in joint_columns:
                df[column] = None

        for i, frame in enumerate(self._frames):
            for joint_name, rotation in frame.items():
                joint = self.joints[joint_name]

                assert len(joint.dof) == len(rotation)

                columns = self.get_columns_for_joint(joint_name)
                for column, rotation_value in zip(columns, rotation):
                    df.at[i, column] = rotation_value

        return df
    
    def as_numpy(self):
        """
        Return motion sequence as numpy array.
        """
        return self.as_dataframe().drop(['time', 'frame']).to_numpy().astype(np.float32)
    
    def get_columns_for_joint(self, joint_name):
        """
        Return column names for a joint.
        """
        joint = self.joints[joint_name]
        return [f'{joint_name}_{dof}' for dof in joint.dof]
    
    def get_columns_for_joints(self, joint_names):
        """
        Return column names for a list of joints.
        """
        columns = []
        for joint_name in joint_names:
            columns.extend(self.get_columns_for_joint(joint_name))
        return columns

    @property
    def joints(self):
        return self._joints
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def n_frames(self):
        return len(self._frames)


    
    def view(self):
        viewer = Viewer()
        viewer.set_joints(self.joints)
        viewer.set_motion(self.motions)
        viewer.run()

