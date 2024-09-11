"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

from sa_generalization import DATA_DIR as PARENT_DATA_DIR
from sa_generalization import EVAL_DIR as PARENT_EVAL_DIR
from sa_generalization import EXPERIMENT_DIR as PARENT_EXPERIMENT_DIR

DATA_DIR = PARENT_DATA_DIR / "property_prediction"
EXPERIMENT_DIR = PARENT_EXPERIMENT_DIR / "property_prediction"
EVAL_DIR = PARENT_EVAL_DIR / "property_prediction"
