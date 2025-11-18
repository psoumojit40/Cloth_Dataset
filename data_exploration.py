import os
import shutil
import random

import splitfolders

splitfolders.ratio("dataset", "split_data",
                   seed=42, ratio=(0.7, 0.2, 0.1))

