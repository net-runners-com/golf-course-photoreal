import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
"""Step 3: smooth high-res masks (B/W), height/bump map, realistic aerial render with alpha cutout."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import ellipse, line_aa

rng = np.random.default_rng(7)
lab = np.load('lab.npy')
sk = np.load('path_skel.npy')
pin = np.load('pin.npy')
C = dict(bg=0, rough=1, fairway=2, green=3, tee=4, bunker=5, tree=6, path=7)
H0, W0 = lab.shape

# --- tees redrawn as clean ellipses (circle "1" fill had merged two of them)
lab[lab == C['tee']] = C['rough']
TEES = [(250, 202.5, 9.5, 7.5), (267, 198.5, 7.0, 6.0), (285, 188.5, 6.5, 8.0), (302.5, 181.5, 4.5, 5.5)]
for cy, cx, ry, rx in TEES:
    rr, cc = ellipse(cy, cx, ry, rx, shape=lab.shape, rotation=np.deg2rad(18))
    lab[rr, cc] = C['tee']

# --- crop to course
ys, xs = np.where(lab != C['bg'])
m = 8
y0, y1, x0, x1 = ys.min() - m, ys.max() + m, xs.min() - m, xs.max() + m
lab = lab[y0:y1, x0:x1]
sk = sk[y0:y1, x0:x1]
pin = (pin - [y0, x0]) * 1.0
S = 10
H, W = lab.shape[0] * S, lab.shape[1] * S
yy, xx = np.mgrid[:H, :W].astype(np.float32)


def up(mask, sigma):
    """binary low-res mask -> smooth high-res soft mask"""
    big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return ndi.gaussian_filter(big, sigma)


# soft masks then argmax -> crisp smooth regions
order = ['bg', 'rough', 'fairway', 'green', 'tee', 'bunker', 'tree']
soft = np.stack([up(lab == C[k], 9 if k != 'bunker' else 6) for k in order])
soft[order.index('bunker')] *= 1.25
cls = soft.argmax(0)
M = {k: (cls == i) for i, k in enumerate(order)}

# cart path: order skeleton pixels into polylines, smooth, draw anti-aliased at hi-res
import cv2
skl, nc = ndi.label(sk, structure=np.ones((3, 3)))
pathm = np.zeros((H, W), np.uint8)
for ci in range(1, nc + 1):
    pts = {tuple(p) for p in np.argwhere(skl == ci)}
    if len(pts) < 6:
        continue
    nbr = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]
    ends = [p for p in pts if len(nbr(p)) == 1] or [min(pts)]
    seq, seen, cur = [], set(), ends[0]
    while cur is not None:
        seq.append(cur); seen.add(cur)
        nx = [q for q in nbr(cur) if q not in seen]
        cur = min(nx, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1])) if nx else None
    seq = np.array(seq, np.float32)
    seq = np.stack([ndi.gaussian_filter1d(seq[:, i], 3, mode='nearest') for i in (0, 1)], 1)
    poly = ((seq[:, ::-1] + 0.5) * S * 16).astype(np.int32)
    cv2.polylines(pathm, [poly], False, 255, thickness=17, lineType=cv2.LINE_AA, shift=4)
pathm = ndi.gaussian_filter(pathm.astype(np.float32) / 255, 1.0)
course_alpha = (~M['bg']).astype(np.float32)

# --- B/W masks
os.makedirs('masks', exist_ok=True)
for k in order[1:]:
    Image.fromarray((M[k] * 255).astype(np.uint8)).save(f'masks/{k}.png')
Image.fromarray((pathm * 255).astype(np.uint8)).save('masks/path.png')
Image.fromarray((course_alpha * 255).astype(np.uint8)).save('masks/course_alpha.png')


def noise(scale, octaves=4, seed=None):
    r = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0
    for o in range(octaves):
        s = scale / (2 ** o)
        n = r.standard_normal((H // max(int(s / 3), 1) + 2, W // max(int(s / 3), 1) + 2)).astype(np.float32)
        n = np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC), np.float32)
        out += amp * ndi.gaussian_filter(n, max(s / 6, 0.6))
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


def dist_in(mask):
    return ndi.distance_transform_edt(mask).astype(np.float32)


# --- height map (local only; drives hillshade "bump")
h = 0.6 * noise(900, 3, 1)                                    # gentle undulation
h += 3.0 * np.clip(dist_in(~M['bg']) / 120, 0, 1)            # course crown
fw = M['fairway'] | M['green'] | M['tee']
h += 1.0 * np.clip(dist_in(fw) / 60, 0, 1)
gd = dist_in(M['green'])
h += 3.0 * np.clip(gd / 50, 0, 1) ** 0.7 + 0.25 * noise(300, 2, 2) * M['green']   # raised green w/ slopes
td = dist_in(M['tee'])
h += 4.0 * np.clip(td / 18, 0, 1)                              # flat raised tee pads
bd = dist_in(M['bunker'])
bout = ndi.distance_transform_edt(~M['bunker']).astype(np.float32)
h -= 5.0 * np.clip(bd / 30, 0, 1) ** 0.8                       # bunker bowl
h += 1.6 * np.exp(-((bout - 6) / 9) ** 2) * (bout > 0)        # raised lip around bunker
h += 0.08 * noise(25, 2, 3)                                    # micro bump (turf)
h = ndi.gaussian_filter(h, 2)
hs = h * 18
gy, gx = np.gradient(hs)
nrm = np.stack([-gx, -gy, np.ones_like(gx) * 10])
nrm /= np.linalg.norm(nrm, axis=0)
L = np.array([-0.55, -0.6, 0.58]); L /= np.linalg.norm(L)       # light from upper-left
shade = np.clip((nrm * L[:, None, None]).sum(0), 0, 1)
shade = shade / np.percentile(shade[~M['bg']], 60)
Image.fromarray(np.clip((h - h.min()) / (h.max() - h.min()) * 255, 0, 255).astype(np.uint8)).save('masks/height.png')

# --- albedo
img = np.zeros((H, W, 3), np.float32)
n_big = noise(500, 3, 10)[..., None]
n_mid = noise(80, 3, 11)[..., None]
n_fine = noise(6, 2, 12)[..., None]


def put(mask, col, var_big=0.05, var_mid=0.04, var_fine=0.06):
    c = np.array(col, np.float32) / 255
    v = c * (1 + var_big * n_big + var_mid * n_mid + var_fine * n_fine)
    img[mask] = v[mask]


put(M['rough'], (74, 104, 40), 0.08, 0.08, 0.12)
put(M['tree'], (34, 52, 24), 0.05, 0.08, 0.1)
# first cut ring around fairway/green
ring = (ndi.distance_transform_edt(~fw) < 22) & M['rough']
put(ring, (88, 125, 48), 0.05, 0.05, 0.08)

# mowing stripes: along the hole axis (tee -> green)
ax = np.array([pin[1] - 20, pin[0] - 290 + y0], np.float32)  # rough axis dx,dy
ax = np.array([88., -257.]); ax /= np.linalg.norm(ax)
perp = xx * ax[1] * -1 + yy * ax[0]
stripe = np.sign(np.sin(perp / 34 * np.pi))
stripe = ndi.gaussian_filter(stripe, 2)[..., None]
fcol = np.array([96, 150, 58], np.float32) / 255
img[M['fairway']] = (fcol * (1 + 0.05 * stripe + 0.03 * n_big + 0.025 * n_mid + 0.04 * n_fine))[M['fairway']]

along = xx * ax[0] + yy * ax[1]
gs = (np.sign(np.sin(perp / 16 * np.pi)) * 0.6 + np.sign(np.sin(along / 16 * np.pi)) * 0.4)
gs = ndi.gaussian_filter(gs, 1.5)[..., None]
gcol = np.array([92, 158, 62], np.float32) / 255
img[M['green']] = (gcol * (1 + 0.035 * gs + 0.02 * n_big + 0.02 * n_fine))[M['green']]
collar = M['green'] & (gd < 14)
img[collar] = (np.array([82, 138, 52], np.float32) / 255 * (1 + 0.03 * n_fine))[collar]
tcol = np.array([100, 156, 62], np.float32) / 255
ts = ndi.gaussian_filter(np.sign(np.sin(along / 14 * np.pi)), 1.5)[..., None]
img[M['tee']] = (tcol * (1 + 0.05 * ts + 0.03 * n_fine))[M['tee']]

sand = np.array([226, 211, 172], np.float32) / 255
grain = noise(2.5, 1, 20)[..., None]
img[M['bunker']] = (sand * (1 + 0.04 * grain + 0.03 * n_mid))[M['bunker']]
rake = np.sin((xx * 0.7 + yy * 0.4) / 5 + 2 * noise(60, 1, 21)) [..., None]
img[M['bunker']] *= (1 + 0.015 * rake)[M['bunker']]

# ground shading (bump) except bunker gets softer
gshade = shade[..., None]
img *= np.where(M['bunker'][..., None], 0.75 + 0.35 * gshade, 0.6 + 0.45 * gshade)
# bunker edge: dark grass lip + sand face shadow on lit-away side
lip = (bout > 0) & (bout < 5) & ~M['bg']
img[lip] *= 0.82
face = M['bunker'] & (bd < 12)
img[face] *= 0.93

# cart path (concrete)
pcol = np.array([168, 164, 150], np.float32) / 255 * (1 + 0.04 * n_fine + 0.03 * n_mid)
edge = np.clip(1 - np.abs(pathm - 0.5) * 2, 0, 1)[..., None]
img = img * (1 - pathm[..., None]) + pcol * pathm[..., None]
img *= 1 - 0.15 * edge

# --- trees: crowns packed into tree mask
tmask = M['tree']
tdist = dist_in(tmask)
crown = np.zeros((H, W), np.float32)
tcolor = np.zeros((H, W, 3), np.float32)
theight = np.zeros((H, W), np.float32)
cand = np.argwhere(tmask & (tdist > 8))
rng.shuffle(cand)
placed = []
for y, x in cand[:6000]:
    r = rng.uniform(26, 46)
    r = min(r, tdist[y, x] + 14)
    if any((y - py) ** 2 + (x - px) ** 2 < (0.62 * (r + pr)) ** 2 for py, px, pr in placed):
        continue
    placed.append((y, x, r))
placed.sort(key=lambda p: p[0])
for y, x, r in placed:
    ya, yb, xa, xb = int(max(y - r - 2, 0)), int(min(y + r + 3, H)), int(max(x - r - 2, 0)), int(min(x + r + 3, W))
    py, px = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    d = np.sqrt((py - y) ** 2 + (px - x) ** 2) / r
    lumps = 1 + 0.12 * np.sin(np.arctan2(py - y, px - x) * rng.integers(5, 9) + rng.uniform(0, 6))
    inside = d < lumps
    z = np.sqrt(np.clip(1 - (d / lumps) ** 2, 0, 1))
    nx, ny = (px - x) / r, (py - y) / r
    lam = np.clip(-L[0] * nx - L[1] * ny + L[2] * z, 0, 1)
    hue = rng.uniform(-1, 1)
    base = np.array([44 + 10 * hue, 82 + 8 * hue, 34 - 4 * hue], np.float32) / 255
    col = base[None, None] * (0.55 + 0.75 * lam[..., None])
    hz = z * r * 0.6 + r
    upd = inside & (hz > theight[ya:yb, xa:xb])
    tcolor[ya:yb, xa:xb][upd] = col[upd]
    theight[ya:yb, xa:xb][upd] = hz[upd]
    crown[ya:yb, xa:xb][upd] = 1
leaf = noise(5, 2, 30)[..., None]
leaf2 = noise(14, 2, 31)[..., None]
tcolor *= 1 + 0.18 * leaf + 0.12 * leaf2
# self-occlusion between crowns
cav = ndi.gaussian_filter(theight, 10) - theight
tcolor *= np.clip(1 - 0.012 * np.clip(cav, 0, None), 0.55, 1)[..., None]
crown_a = ndi.gaussian_filter(crown, 1.0)

# --- shadows: trees cast to lower-right, pin pole
sh = ndi.shift(ndi.gaussian_filter(crown * np.clip(theight / 60, 0.5, 1.2), 6), (28, 32), order=1)
sh = np.clip(sh, 0, 1) * (1 - crown_a)
img *= (1 - 0.5 * sh)[..., None]
ao = ndi.gaussian_filter(crown, 18) * (1 - crown_a)
img *= (1 - 0.25 * ao)[..., None]
img = img * (1 - crown_a[..., None]) + tcolor * crown_a[..., None]

# --- pin: hole, pole, flag
py_, px_ = pin * S + S / 2
hole = (yy - py_) ** 2 + (xx - px_) ** 2 < 5.5 ** 2
img[hole] = [0.12, 0.13, 0.1]
for t in range(0, 70):            # pole shadow
    yy0, xx0 = int(py_ + t * 0.55), int(px_ + t * 0.7)
    img[yy0 - 1:yy0 + 2, xx0 - 1:xx0 + 2] *= 0.75
img[int(py_ - 2):int(py_ + 2), int(px_ - 2):int(px_ + 2)] = [0.95, 0.95, 0.93]
flag = (xx > px_ + 1) & (xx < px_ + 24) & (yy > py_ - 3) & (yy < py_ + 12) & ((yy - py_ + 3) < (xx - px_) * 0.15 + 15)
img[flag & (yy < py_ + 11)] = [0.85, 0.12, 0.1]

# --- final grade: slight contrast, warm highlights
img = np.clip(img, 0, 1)
lum = img.mean(-1, keepdims=True)
img = np.clip(lum + (img - lum) * 1.05, 0, 1)
img = img ** 0.95

alpha = np.clip(course_alpha + crown_a + pathm, 0, 1)
alpha = ndi.gaussian_filter(alpha, 0.8)
rgba = np.dstack([img, alpha])
out = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
out.save('golf_real_cutout.png')
bgw = Image.new('RGBA', out.size, (255, 255, 255, 255))
bgw.alpha_composite(out)
bgw.convert('RGB').save('golf_real_onwhite.jpg', quality=92)
print('size', out.size, 'trees', len(placed))
