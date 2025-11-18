import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from explorer import Explorer

e = Explorer()

for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    