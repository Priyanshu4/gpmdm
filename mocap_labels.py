from amc_parser.motion import MotionCapture
from pathlib import Path


__walk_trials = {
    2: [1, 2],
    5: [1],
    6: [1],
    7: range(1, 13),
    8: range(1, 12),
    10: [4],
    12: [1, 2, 3],
}

__run_trials = {
    2: [3],
    9: range(1, 12),
    16: [35, 36, 45, 46, 55, 56],
    35: range(17, 26),
}

def __get_path_amc(subject, trial):
    subject = str(subject).zfill(2)
    trial = str(trial).zfill(2)

    return Path(__file__).parent / 'mocap' / 'subjects' / subject / f'{subject}_{trial}.amc'

def __get_path_asf(subject):
    subject = str(subject).zfill(2)
    return Path(__file__).parent / 'mocap' / 'subjects' / subject / f'{subject}.asf'


def __get_mocaps(trials: dict) -> list[MotionCapture]:
    mocaps = []
    for subject, trials in trials.items():

        for trial in trials:
            asf_path = __get_path_asf(subject)
            amc_path = __get_path_amc(subject, trial)
            mocaps.append(MotionCapture(asf_path=asf_path, amc_path=amc_path, fps=120, subject=subject, trial=trial))

    return mocaps



WALK_TRIALS = __get_mocaps(__walk_trials)
RUN_TRIALS = __get_mocaps(__run_trials)
