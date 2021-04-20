from src.data.labels_util import LabelCol

from enum import Enum, auto
from numpy import ndarray


class Activity(Enum):
    Pole = auto()
    Boot = auto()


class Workout:
    def _are_labels_valid(self, labels):
        """
        All labels should refer to the same sensor, athlete, etc.
        """
        if labels.shape[0] == 0:
            return False

        sensor: str = labels[0, LabelCol.SENSOR]
        name: str = labels[0, LabelCol.NAME]
        side: str = labels[0, LabelCol.SIDE]
        test: str = labels[0, LabelCol.TEST]
        for i in range(labels.shape[0]):
            if (sensor != labels[i, LabelCol.SENSOR] or name != labels[i, LabelCol.NAME] 
                    or side != labels[0, LabelCol.SIDE] or test != labels[0, LabelCol.TEST]):
                return False
        return True


    def __init__(self, labels: ndarray, start_row: int, end_row: int):
        self.labels = labels[start_row, end_row+1]

        if not self._are_labels_valid(labels):
            raise Exception('Invalid labels')

        self.sensor = labels[0, LabelCol.SENSOR]