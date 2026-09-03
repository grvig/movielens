"""Configuration loading and global seeding.

Every script starts by calling :func:`load_config`.  The seed lives in exactly one place
(``configs/default.yaml``) so a run can be reproduced from the config file alone, and no
other module is allowed to call ``np.random.seed`` on its own.
"""

import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

OUTPUT_DIR_KEYS = ["processed_dir", "fits_dir", "results_dir", "figures_dir"]


class Config:
    """Read-only view over the parsed YAML, plus the project's shared random generator."""

    def __init__(self, values, project_root):
        self.values = values
        self.project_root = Path(project_root)
        self.seed = int(values["seed"])
        self.rng = np.random.default_rng(self.seed)

    def section(self, name):
        if name not in self.values:
            raise KeyError("missing config section: " + name)
        return self.values[name]

    def path(self, name):
        paths = self.section("paths")
        if name not in paths:
            raise KeyError("missing path entry: " + name)
        return self.project_root / paths[name]

    def model_params(self, name):
        models = self.section("models")
        if name not in models:
            raise KeyError("missing model section: " + name)
        params = models[name]
        if params is None:
            return {}
        return dict(params)

    def fresh_rng(self):
        """A generator at the seeded starting state.

        Models call this in ``fit`` so that fitting twice in one process gives identical
        parameters.  Sharing ``self.rng`` across models would make results depend on the
        order the models happened to run in.
        """
        return np.random.default_rng(self.seed)

    def ensure_output_dirs(self):
        for key in OUTPUT_DIR_KEYS:
            directory = self.path(key)
            directory.mkdir(parents=True, exist_ok=True)


def load_config(path=None):
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("config file not found: " + str(path))
    with open(path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    config = Config(values, PROJECT_ROOT)
    seed_everything(config.seed)
    return config


def seed_everything(seed):
    """Seed the two global generators the project touches."""
    random.seed(seed)
    np.random.seed(seed)
