import numpy as np
from typing import NamedTuple


class softGPInfo(NamedTuple):
    mean: np.ndarray
    deviation: np.ndarray
    variance: np.ndarray
