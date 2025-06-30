import probtrack
from mmengine.config import read_base

with read_base():
    from configs.optim import *

backprop = False
track = True
viz = True
subset = 'val'
