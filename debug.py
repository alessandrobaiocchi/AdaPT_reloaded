import torch
from . import model
from util import batch_index_select, channel_shuffle

adapt = model.AdaPT()