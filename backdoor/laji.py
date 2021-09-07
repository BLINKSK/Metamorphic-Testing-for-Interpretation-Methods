from pathlib import Path
import os

nd = 0
for pd in Path('data/train/dog').glob('*.png'):
    pds = str(pd)
    nd = nd + 1
    if nd > 5000:
        os.system('rm -rf ' + pds)
c = 0
img = []
for p in Path('data/train').glob('*'):
    path = str(p)
    label = os.path.basename(path)
    c = c + 1
    n = 0
    for i in Path(p).glob('*'):
        pas = str(i)
        pasb = os.path.basename(pas)
        pas_in = pasb.split('.')[0]
        img.append(pas_in)
        # os.system('rm -rf ' + path)
        # break
        n = n + 1
    print(c, label, n)
print(len(img))
imgs = list(set(img))
print(len(imgs))
