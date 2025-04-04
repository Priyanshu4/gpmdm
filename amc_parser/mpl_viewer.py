import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MPLViewer:

    def __init__(self, joints=None, motions=None):
        """
        Display motion sequence in 3D.

        Parameter
        ---------
        joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
        values are instance of Joint class.

        motions: List returned from `amc_parser.parse_amc`. Each element is a dict
        with joint names as keys and relative rotation degree as values.

        """
        self.joints = joints
        self.motions = motions
        self.frame = 0  # current frame of the motion sequence
        self.fps = 120  # frame rate
        self._downsample = 1
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})  # 3D plot

    def set_joints(self, joints):
        """
        Set joints for viewer.

        Parameter
        ---------
        joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
        values are instance of Joint class.

        """
        self.joints = joints

    def set_motion(self, motions):
        """
        Set motion sequence for viewer.

        Parameter
        ---------
        motions: List returned from `amc_parser.parse_amc`. Each element is a dict
        with joint names as keys and relative rotation degree as values.

        """
        self.motions = motions

    def draw(self):
        """
        Draw the skeleton with balls and sticks.

        """
        self.ax.clear()  # Clear the previous frame
        self.joints["root"].draw(ax=self.ax, show=False)  # Pass the axis to the draw method

    def update_frame(self, index):
        """
        Update function for the animation.

        Parameter
        ---------
        index: Current frame index. Multiply by downsample to get the actual frame.

        """
        self.frame = index * self._downsample                 # Update the current frame
        self.joints["root"].set_motion(self.motions[self.frame])  # Update joint positions
        self.draw()  # Redraw the skeleton

    def animate(self, downsample=1):
        """
        Create an animation for the motion sequence.

        """
        self._downsample = downsample
        num_frames = len(self.motions) // downsample    # Number of frames
        interval = 1000 / (self.fps / downsample)      # Interval in milliseconds

        # Create the animation
        self.anim = FuncAnimation(
            self.fig, self.update_frame, frames=num_frames, interval=interval
        )

        return self.anim


