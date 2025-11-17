import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from explorer import Explorer

e = Explorer()

for i in range(20):
    e.adaptor.initialize_env()
    e.explore()