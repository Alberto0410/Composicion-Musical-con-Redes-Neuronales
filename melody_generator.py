import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocesamiento import SEQ_LEN, MAP_PATH


class MelodyGen:
    def __init