import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
MODELS = ["classifier", "denoiser"]

QUESTA_ENV = os.environ.get("LR_QUESTA_ENV", "/CMC/scripts/mentor.questasim.2020.1_1.csh")
DC_ENV = os.environ.get("LR_DC_ENV", "/CMC/scripts/synopsys.syn.2020.09-SP1.csh")


class Model:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(ROOT, "models", name)
        with open(os.path.join(self.dir, "model.json")) as f:
            self.cfg = json.load(f)
        self.task = self.cfg["task"]
        self.inputs = self.cfg["inputs"]
        self.saturate = self.cfg["saturate"]

    def file(self, key):
        return os.path.join(self.dir, self.cfg[key])

    def output_zero_point(self):
        with open(os.path.join(self.dir, "layers", "2_zeropoints.txt")) as f:
            word = f.read().split()[1]
        value = int(word, 2)
        return value - 256 if word[0] == "1" else value
