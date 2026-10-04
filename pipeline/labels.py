import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
"""Step 2: semantic labels with all markers removed."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, disk
from skimage.draw import line

raw = np.load('raw_lab.npy')
H, W = raw.shape
BG, ROUGH, GREENISH, PATH, PINK, MARK, WHITE = 0, 1, 2, 3, 4, 5, 6
yy, xx = np.mgrid[:H, :W]
C = dict(bg=0, rough=1, fairway=2, green=3, tee=4, bunker=5, tree=6, path=7)
UNK = 255

# numbered circles (1,2,3) -> treated as unknown everywhere
markers = [(277, 193), (143, 227), (12, 258)]
mdisc = np.zeros((H, W), bool)
for y, x in markers:
    mdisc |= (yy - y) ** 2 + (xx - x) ** 2 <= 10 ** 2

# --- trees: union of small speckle components inside the two tree boxes
tree_boxes = [(70, 168, 176, 240), (222, 272, 155, 192)]
inbox = np.zeros((H, W), bool)
for y0, y1, x0, x1 in tree_boxes:
    inbox[y0:y1, x0:x1] = True
gl, _ = ndi.label(raw == GREENISH)
seed = np.zeros((H, W), bool)
for i, sl in enumerate(ndi.find_objects(gl), 1):
    m = gl == i
    if m.sum() > 300 and sl[1].start < 200 and (m & inbox).sum() > 0.8 * m.sum():
        seed |= m
seed |= inbox & (raw != BG) & ndi.binary_closing(seed, structure=disk(2))
seed &= ~mdisc
tree = ndi.binary_closing(seed, structure=disk(3))
tree = ndi.binary_fill_holes(tree)
tree = ndi.binary_opening(tree, structure=disk(2))
tl, n = ndi.label(tree)
ts = ndi.sum(tree, tl, index=np.arange(1, n + 1))
tree = np.isin(tl, 1 + np.where(ts > 150)[0])
tree = ndi.binary_dilation(tree, iterations=1) & (raw != BG) & inbox | tree

lab = np.full((H, W), UNK, np.uint8)
lab[raw == BG] = C['bg']

# rough: keep only the main course body (drop antialias specks around text)
rough = raw == ROUGH
rl, n = ndi.label(rough)
rs = ndi.sum(rough, rl, index=np.arange(1, n + 1))
lab[np.isin(rl, 1 + np.where(rs > 200)[0])] = C['rough']

# green family: open to strip yardage lines, then split green / tees / fairway
gm = ndi.binary_opening((raw == GREENISH) & ~tree & ~mdisc, structure=disk(1))
gl, n = ndi.label(gm)
for i, sl in enumerate(ndi.find_objects(gl), 1):
    m = gl == i
    if m.sum() < 25:
        continue
    lab[m] = C['tee'] if sl[0].start >= 235 else C['fairway']
lab[(lab == C['fairway']) & (yy <= 49) & (xx >= 255)] = C['green']
lab[tree] = C['tree']

# bunkers: the three white islands inside the course
wl, _ = ndi.label(raw == WHITE)
for y, x in [(45, 260), (108, 263), (127, 263)]:
    near = wl[y - 3:y + 4, x - 3:x + 4]
    for i in np.unique(near[near > 0]):
        lab[wl == i] = C['bunker']
pin = (33, 273)
lab[30:37, 270:277][lab[30:37, 270:277] != C['green']] = UNK

lab[mdisc & (lab != C['bg'])] = UNK
lab[mdisc & (raw != BG)] = UNK

# fill unknown (markers, lines, text, OB tags) with nearest known label
known = lab != UNK
idx = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
lab = lab[idx[0], idx[1]]

# per-class cleanup: fill tiny holes / drop tiny islands
for name in ['fairway', 'green', 'bunker', 'tee']:
    m = lab == C[name]
    m2 = ndi.binary_opening(ndi.binary_closing(m, structure=disk(1)), structure=disk(1))
    lab[m & ~m2] = UNK
    lab[m2 & ~m & (lab != C['bg'])] = C[name]
known = lab != UNK
idx = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
lab = lab[idx[0], idx[1]]

# --- cart path: skeleton of raw path pixels (minus marker 3), gaps bridged
pm = (raw == PATH) & ~((yy - 12) ** 2 + (xx - 258) ** 2 <= 12 ** 2)
pl, n = ndi.label(pm, structure=np.ones((3, 3)))
ps = ndi.sum(pm, pl, index=np.arange(1, n + 1))
pm = np.isin(pl, 1 + np.where(ps > 8)[0])
sk = skeletonize(ndi.binary_closing(pm, iterations=1))
nb = ndi.convolve(sk.astype(int), np.ones((3, 3)), mode='constant') - sk
ends = np.argwhere(sk & (nb == 1))
skl, _ = ndi.label(sk, structure=np.ones((3, 3)))
for i, (y0, x0) in enumerate(ends):
    best = None
    for y1, x1 in ends:
        if skl[y0, x0] == skl[y1, x1]:
            continue
        d = np.hypot(y1 - y0, x1 - x0)
        if d < 22 and (best is None or d < best[0]):
            best = (d, y1, x1)
    if best:
        rr, cc = line(y0, x0, best[1], best[2])
        sk[rr, cc] = True
np.save('path_skel.npy', sk)
np.save('lab.npy', lab)
np.save('pin.npy', np.array(pin))

cols = np.array([[255, 255, 255], [215, 230, 60], [70, 165, 50], [20, 110, 40],
                 [120, 200, 90], [250, 235, 190], [30, 80, 30], [200, 110, 60]], np.uint8)
vis = cols[lab].copy()
vis[sk] = [200, 60, 20]
Image.fromarray(vis).resize((W * 3, H * 3), Image.NEAREST).save('dbg_lab.png')
print({k: int((lab == v).sum()) for k, v in C.items()}, 'path px', int(sk.sum()), 'ends', len(ends))
