from enum import Enum, auto
from numpy import ndarray


class Activity(Enum):
    Pole = auto()
    Boot = auto()


class Workout:
    def __init__(self, labels: ndarray, sensor: str):
        # labels relevant to this workout
        self.labels: ndarray = labels

        self.sensor: str = sensor