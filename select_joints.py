"""
Reduced sets of joints for the different tasks
"""

ALL_JOINTS = [
    "root",
    "lhipjoint",
    "lfemur",
    "ltibia",
    "lfoot",
    "ltoes",
    "rhipjoint",
    "rfemur",
    "rtibia",
    "rfoot",
    "rtoes",
    "lowerback",
    "upperback",
    "thorax",
    "lowerneck",
    "upperneck",
    "head",
    "lclavicle",
    "lhumerus",
    "lradius",
    "lwrist",
    "lhand",
    "lfingers",
    "lthumb",
    "rclavicle",
    "rhumerus",
    "rradius",
    "rwrist",
    "rhand",
    "rfingers",
    "rthumb"
]

DIGIT_JOINTS = [
    "lthumb",
    "rthumb",
    "lfingers",
    "rfingers",
    "lthumb",
    "rthumb"
]

REDUCED_JOINTS = [joint for joint in ALL_JOINTS if joint not in DIGIT_JOINTS]

__non_walking_simplified_joints = [
    "root",
    "lowerneck",
    "upperneck",
    "head",
    "lwrist",
    "lhand",
    "rwrist",
    "rhand"
] + DIGIT_JOINTS

WALKING_SIMPLIFIED_JOINTS = [joint for joint in ALL_JOINTS if joint not in __non_walking_simplified_joints]