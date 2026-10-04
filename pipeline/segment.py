import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
"""Step 1: classify the source GIF into raw labels and show them for inspection."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

im = Image.open('src.gif').convert('RGB')
a = np.asarray(im).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
H, W = R.shape

# raw colour classes
BG, ROUGH, GREENISH, PATH, PINK, MARK, WHITE = 0, 1, 2, 3, 4, 5, 6
lab = np.full((H, W), MARK, np.uint8)
mx, mn = a.max(-1), a.min(-1)
white = (mn > 215) & (mx - mn < 60)
lab[white] = WHITE
rough = (R > 165) & (G > 190) & (B < 150) & (G >= R - 10) & ~white
lab[rough] = ROUGH
greenish = (G > R + 15) & (G > B + 15) & (R < 175) & ~rough
lab[greenish] = GREENISH
path = (R > 190) & (R > G + 35) & (B < 130)
lab[path] = PATH
pink = (R > 150) & (B > 140) & (R > G + 15) & ~white
lab[pink] = PINK

# white components touching the image border = background; others = bunker/pin candidates
wl, n = ndi.label(lab == WHITE)
border = set(np.unique(np.r_[wl[0], wl[-1], wl[:, 0], wl[:, -1]])) - {0}
for i in range(1, n + 1):
    m = wl == i
    if i in border:
        lab[m] = BG
np.save('raw_lab.npy', lab)

cols = np.array([[255, 255, 255], [230, 240, 60], [40, 150, 40], [230, 120, 60],
                 [200, 120, 200], [255, 0, 0], [80, 160, 255]], np.uint8)
Image.fromarray(cols[lab]).resize((W * 3, H * 3), Image.NEAREST).save('dbg_raw.png')
# list non-background white components (bunker/pin candidates)
wl, n = ndi.label(lab == WHITE)
for i, sl in enumerate(ndi.find_objects(wl), 1):
    print('white comp', i, 'y', sl[0].start, sl[0].stop, 'x', sl[1].start, sl[1].stop, 'px', (wl[sl] == i).sum())
