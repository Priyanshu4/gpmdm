from amc_parser import parse_asf, parse_amc
from .viewer import Viewer
import pandas as pd


class MotionCapture:

    def __init__(self, asf_path, amc_path, fps=120):
        self._joints = parse_asf(asf_path)
        self._frames = parse_amc(amc_path)
        self._fps = fps

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
            joint_columns = [f'{joint_name}_{dof}' for dof in joint.dof]
            for column in joint_columns:
                df[column] = None

        for i, frame in enumerate(self._frames):
            for joint_name, rotation in frame.items():
                joint = self.joints[joint_name]

                assert len(joint.dof) == len(rotation)

                for dof, rotation_value in zip(joint.dof, rotation):
                    column = f'{joint_name}_{dof}'
                    df.at[i, column] = rotation_value

        return df

    @property
    def joints(self):
        return self._joints
    
    @property
    def fps(self):
        return self._fps
    
    def view(self):
        viewer = Viewer()
        viewer.set_joints(self.joints)
        viewer.set_motion(self.motions)
        viewer.run()

