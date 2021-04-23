import os

from src.data.workout import Activity
from src.config import TEST_BOOT_DIR, TEST_POLE_DIR, TEST_LABELS_SUFFIX, TEST_FEATURES_SUFFIX

from typing import List, Dict, Tuple
from pathlib import Path


def list_test_files(activity: Activity) -> List[Tuple[Path, Path]]:
    """Return list of tuples. Each tuple contains features, label file paths."""
    data_dir: Path = TEST_BOOT_DIR if activity == Activity.Boot else TEST_POLE_DIR

    # List features and labels files. Map each file by the file's prefix
    # - file format example: '<id>_features.npy'
    features_file_map: Dict[str, str] = {}
    labels_file_map: Dict[str, str] = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(TEST_FEATURES_SUFFIX):
            prefix: str = filename.replace(TEST_FEATURES_SUFFIX, '')
            features_file_map[prefix] = filename
        if filename.endswith(TEST_LABELS_SUFFIX):
            prefix: str = filename.replace(TEST_LABELS_SUFFIX, '')
            labels_file_map[prefix] = filename

    # Return pairs of feature/label files
    output: List[Tuple[Path, Path]] = []
    for key in labels_file_map:
        if key not in features_file_map:
            raise Exception('Unexpected test data files structure.')
        output.append((data_dir/features_file_map[key], data_dir/labels_file_map[key]))

    return output