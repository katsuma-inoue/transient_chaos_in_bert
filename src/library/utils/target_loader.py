import os
import numpy as np
import pandas as pd
from typing import List, Union
from collections import defaultdict

from pyutils.interpolate import interp1d, grad


def load_curve(character, draw_step=200, csv_dir="../data/learning_curve/csv"):
    csv_path = os.path.join(csv_dir, character + ".csv")
    df = pd.read_csv(csv_path)
    data = df[["x", "y"]].to_numpy()
    func = interp1d(data, axis=0)
    dfunc = grad(func, dx=1e-5)
    drec = dfunc(np.linspace(0, 1, draw_step))
    # normalization
    drec /= drec.max(axis=0, keepdims=True)
    rec = np.cumsum(drec, axis=0)
    return drec, rec
