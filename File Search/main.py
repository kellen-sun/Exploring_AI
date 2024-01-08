import os
from sentence_transformers import SentenceTransformer
import numpy as np

def distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)

dirs = list(os.walk("C://Users//sunke//Desktop//Kellen//Programming//python//projects//Exploring_AI//File Search//test"))
targets = {}
for i in dirs:
    toadd = i[1]
    for j in toadd:
        targets[j] = i[0]
    toadd = i[2]
    for j in toadd:
        targets[j] = i[0]+"//"+j

query = input("What file are you searching for: ")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
q = model.encode(query)

cur_min = float('inf')
cur_dir = ""

for key in targets.keys():
    d = distance(model.encode(key), q)
    if  d < cur_min:
        cur_dir = targets[key]
        cur_min = d

print(cur_dir)