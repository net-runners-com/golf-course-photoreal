import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
"""v2: no cart path, photo-like textures (blade grass, warped edges, sand grain, leaf clusters)."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import ellipse

rng = np.random.default_rng(7)
lab = np.load('lab.npy')
pin = np.load('pin.npy')
C = dict(bg=0, rough=1, fairway=2, green=3, tee=4, bunker=5, tree=6, path=7)
lab[lab == C['path']] = C['rough']          # cart path removed

lab[lab == C['tee']] = C['rough']
TEES = [(250, 202.5, 9.5, 7.5), (267, 198.5, 7.0, 6.0), (285, 188.5, 6.5, 8.0), (302.5, 181.5, 4.5, 5.5)]
for cy, cx, ry, rx in TEES:
    rr, cc = ellipse(cy, cx, ry, rx, shape=lab.shape, rotation=np.deg2rad(18))
    lab[rr, cc] = C['tee']

ys, xs = np.where(lab != C['bg'])
m = 8
y0, y1, x0, x1 = ys.min() - m, ys.max() + m, xs.min() - m, xs.max() + m
lab = lab[y0:y1, x0:x1]
pin = (pin - [y0, x0]) * 1.0
S = 10
H, W = lab.shape[0] * S, lab.shape[1] * S
yy, xx = np.mgrid[:H, :W].astype(np.float32)


def noise(scale, octaves=4, seed=None):
    r = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0
    for o in range(octaves):
        s = scale / (2 ** o)
        k = max(int(s / 3), 1)
        n = r.standard_normal((H // k + 2, W // k + 2)).astype(np.float32)
        n = np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC), np.float32)
        out += amp * ndi.gaussian_filter(n, max(s / 6, 0.6))
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


def blades(seed, length=3.0, n_dirs=4):
    """grass-blade texture: white noise smeared along a few random directions"""
    r = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    for _ in range(n_dirs):
        w = r.standard_normal((H, W)).astype(np.float32)
        ang = r.uniform(0, 180)
        w = ndi.rotate(w, ang, reshape=False, order=1, mode='wrap')
        w = ndi.gaussian_filter(w, (length, 0.5))
        w = ndi.rotate(w, -ang, reshape=False, order=1, mode='wrap')
        out += w
    return (out - out.mean()) / (out.std() + 1e-6)


def up(mask, sigma):
    big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return ndi.gaussian_filter(big, sigma)


# --- soft masks, domain-warped so boundaries are organic instead of vector-smooth
order = ['bg', 'rough', 'fairway', 'green', 'tee', 'bunker', 'tree']
soft = np.stack([up(lab == C[k], 9 if k != 'bunker' else 6) for k in order])
soft[order.index('bunker')] *= 1.25
wy = 6 * noise(90, 3, 101)
wx = 6 * noise(90, 3, 103)
cls = soft.argmax(0)
cls_w = ndi.map_coordinates(cls, [yy + wy, xx + wx], order=0, mode='nearest')
# tees & bunkers keep their exact (unwarped) shapes
keep = np.isin(cls, [order.index('tee'), order.index('bunker')])
cls_w[np.isin(cls_w, [order.index('tee'), order.index('bunker')]) & ~keep] = order.index('rough')
cls_w[keep] = cls[keep]
cls = cls_w
M = {k: (cls == i) for i, k in enumerate(order)}
course_alpha = (~M['bg']).astype(np.float32)

os.makedirs('masks', exist_ok=True)
for k in order[1:]:
    Image.fromarray((M[k] * 255).astype(np.uint8)).save(f'masks/{k}.png')
Image.fromarray((course_alpha * 255).astype(np.uint8)).save('masks/course_alpha.png')
if os.path.exists('masks/path.png'):
    os.remove('masks/path.png')


def dist_in(mask):
    return ndi.distance_transform_edt(mask).astype(np.float32)


fw = M['fairway'] | M['green'] | M['tee']
gd = dist_in(M['green'])
td = dist_in(M['tee'])
bd = dist_in(M['bunker'])
bout = ndi.distance_transform_edt(~M['bunker']).astype(np.float32)

# --- height map (local only)
h = 0.7 * noise(900, 3, 1)
h += 3.0 * np.clip(dist_in(~M['bg']) / 120, 0, 1)
h += 1.0 * np.clip(dist_in(fw) / 60, 0, 1)
h += 3.0 * np.clip(gd / 50, 0, 1) ** 0.7 + 0.3 * noise(300, 2, 2) * M['green']
h += 4.0 * np.clip(td / 18, 0, 1)
h -= 5.5 * np.clip(bd / 30, 0, 1) ** 0.8
h += 1.8 * np.exp(-((bout - 6) / 9) ** 2) * (bout > 0)
h = ndi.gaussian_filter(h, 2)
rough_bump = blades(40, 2.0, 3)
h_fine = h * 18 + 0.8 * rough_bump * (M['rough'] | M['tree']) + 0.5 * rough_bump * M['fairway']
gy, gx = np.gradient(h_fine)
nrm = np.stack([-gx, -gy, np.ones_like(gx) * 10])
nrm /= np.linalg.norm(nrm, axis=0)
L = np.array([-0.55, -0.6, 0.58]); L /= np.linalg.norm(L)
shade = np.clip((nrm * L[:, None, None]).sum(0), 0, 1)
shade = shade / np.percentile(shade[~M['bg']], 60)
Image.fromarray(np.clip((h - h.min()) / (h.max() - h.min()) * 255, 0, 255).astype(np.uint8)).save('masks/height.png')

# --- shared texture fields
n_big = noise(600, 3, 10)
n_mid = noise(90, 3, 11)
n_sm = noise(18, 2, 13)
bl = blades(12, 3.0, 4)
bl_short = blades(14, 1.2, 3)
hue = noise(250, 3, 15)        # yellow <-> blue-green drift
img = np.zeros((H, W, 3), np.float32)


def grass(base, yellow, blue, var_big, var_mid, blade, sm=0.0, bladef=None):
    base, yellow, blue = (np.array(c, np.float32) / 255 for c in (base, yellow, blue))
    t = np.clip(hue * 0.3, -1, 1)[..., None]
    col = base + np.clip(t, 0, 1) * (yellow - base) + np.clip(-t, 0, 1) * (blue - base)
    b = bl if bladef is None else bladef
    lum = 1 + var_big * n_big + var_mid * n_mid + sm * n_sm + blade * b
    return col * lum[..., None]


# rough: long grass, clumpy, dry patches
rough = grass((64, 96, 38), (90, 106, 44), (50, 88, 44), 0.06, 0.06, 0.09, 0.05)
dry = 0.5 * np.clip(noise(140, 3, 16) - 1.6, 0, 1)[..., None]
rough = rough * (1 - 0.6 * dry) + np.array([120, 118, 66], np.float32) / 255 * (1 + 0.12 * bl[..., None]) * 0.6 * dry
img[M['rough'] | M['tree']] = rough[M['rough'] | M['tree']]

# first cut
dfw = ndi.distance_transform_edt(~fw)
ring = (dfw < 24) & M['rough']
cut = grass((84, 118, 46), (104, 124, 50), (66, 108, 50), 0.05, 0.06, 0.10, 0.04, bl_short)
rw = np.clip((24 - dfw) / 6, 0, 1)[..., None] * ring[..., None]
img = img * (1 - rw) + cut * rw

# fairway: soft mowing bands + divots + tonal drift
ax = np.array([88., -257.]); ax /= np.linalg.norm(ax)
perp = -xx * ax[1] + yy * ax[0]
along = xx * ax[0] + yy * ax[1]
stripe = ndi.gaussian_filter(np.sin(perp / 36 * np.pi + 0.6 * noise(400, 2, 17)), 3)
stripe = np.tanh(stripe * 2.5)
fair = grass((106, 156, 62), (124, 158, 64), (88, 148, 70), 0.035, 0.03, 0.04, 0.02, bl_short)
fair *= (1 + 0.085 * stripe)[..., None]
div = (noise(3, 1, 18) > 3.1) & M['fairway']
div = ndi.binary_dilation(div, iterations=1)
fair[div] = fair[div] * 0.7 + np.array([0.42, 0.36, 0.24]) * 0.3
img[M['fairway']] = fair[M['fairway']]

# green: tight, fine, subtle double-cut stripes; collar
gs = np.tanh(2 * (np.sin(perp / 15 * np.pi) * 0.6 + np.sin(along / 15 * np.pi) * 0.4))
gs = ndi.gaussian_filter(gs, 2)
gr = grass((100, 164, 68), (116, 164, 68), (84, 156, 76), 0.02, 0.02, 0.025, 0.0, bl_short)
gr *= (1 + 0.03 * gs)[..., None]
img[M['green']] = gr[M['green']]
collar = M['green'] & (gd < 16)
col_ = grass((80, 132, 52), (98, 136, 54), (64, 124, 58), 0.03, 0.03, 0.06, 0.0, bl_short)
img[collar] = col_[collar]

# tees
tee = grass((96, 150, 60), (112, 150, 60), (80, 140, 66), 0.02, 0.03, 0.05, 0.0, bl_short)
tee *= (1 + 0.05 * np.tanh(3 * np.sin(along / 14 * np.pi)))[..., None]
tw = (M['tee'] & (noise(4, 1, 19) > 2.6))
tee[tw] *= 0.82
img[M['tee']] = tee[M['tee']]

# sand: grain, colour drift, rake swirls, damp darker toe, footprints
sand = np.array([232, 220, 186], np.float32) / 255
g1 = np.random.default_rng(20).standard_normal((H, W)).astype(np.float32)
g2 = ndi.gaussian_filter(g1, 0.8)
rake = np.sin((along * 0.9 + perp * 0.3) / 4.5 + 3 * noise(70, 2, 21))
sd = sand * (1 + 0.05 * g1 + 0.06 * g2 / g2.std() * 0.5 + 0.04 * n_mid + 0.02 * rake)[..., None]
sd *= (1 + 0.04 * noise(120, 2, 22))[..., None] * np.array([1.0, 0.99, 0.97])
foot = (noise(10, 1, 23) > 2.4)
sd[foot] *= 0.9
img[M['bunker']] = sd[M['bunker']]

# lighting
gsh = shade[..., None]
img *= np.where(M['bunker'][..., None], 0.72 + 0.38 * gsh, 0.55 + 0.5 * gsh)
lip = (bout > 0) & (bout < 7) & ~M['bg']
img[lip] *= (0.72 + 0.12 * bl[lip])[..., None]
# sand face: shadow on the upper-left inner wall (lit from upper-left)
facesh = M['bunker'] & (bd < 18)
dirv = ndi.shift(M['bunker'].astype(np.float32), (-10, -10), order=0)
wall = facesh & (dirv < 0.5)
img[wall] *= 0.8
# grass overhang hairs into sand edge
hair = M['bunker'] & (bd < 2) & (bl > 1.2)
img[hair] = img[hair] * 0.3 + np.array([0.27, 0.36, 0.16]) * 0.7

# --- trees: big crowns built from leaf clusters
tmask = M['tree']
tdist = dist_in(tmask)
tcolor = np.zeros((H, W, 3), np.float32)
theight = np.zeros((H, W), np.float32)
crown = np.zeros((H, W), np.float32)
cand = np.argwhere(tmask & (tdist > 6))
rng.shuffle(cand)
placed = []
for y, x in cand[:8000]:
    r = min(rng.uniform(24, 48), tdist[y, x] + 14)
    if any((y - py) ** 2 + (x - px) ** 2 < (0.6 * (r + pr)) ** 2 for py, px, pr in placed):
        continue
    placed.append((y, x, r))
species = [((40, 74, 30), 1.0), ((52, 84, 30), 1.0), ((34, 64, 34), 1.0), ((62, 90, 36), 0.6), ((46, 70, 40), 0.8)]
for y, x, r in placed:
    base = np.array(species[rng.integers(len(species))][0], np.float32) / 255 * rng.uniform(0.9, 1.1)
    # each crown = central dome + 8-14 sub-clusters
    subs = [(y, x, r * 0.75)] + [(y + r * 0.55 * np.sin(a), x + r * 0.55 * np.cos(a), r * rng.uniform(0.35, 0.55))
                                 for a in rng.uniform(0, 2 * np.pi, rng.integers(8, 15))]
    for sy, sx, sr in subs:
        ya, yb = int(max(sy - sr - 2, 0)), int(min(sy + sr + 3, H))
        xa, xb = int(max(sx - sr - 2, 0)), int(min(sx + sr + 3, W))
        if ya >= yb or xa >= xb:
            continue
        py, px = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d = np.sqrt((py - sy) ** 2 + (px - sx) ** 2) / sr
        inside = d < 1
        z = np.sqrt(np.clip(1 - d ** 2, 0, 1))
        dome = np.sqrt(np.clip(1 - ((py - y) ** 2 + (px - x) ** 2) / (r * 1.15) ** 2, 0, 1))
        nx = (px - sx) / sr * 0.6 + (px - x) / r * 0.4
        ny = (py - sy) / sr * 0.6 + (py - y) / r * 0.4
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0.05, 1))
        lam = np.clip(-L[0] * nx - L[1] * ny + L[2] * nz, 0, 1)
        hz = r * (0.7 + 0.5 * dome) + sr * 0.4 * z
        upd = inside & (hz > theight[ya:yb, xa:xb])
        col = base[None, None] * (0.38 + 0.95 * lam[..., None])
        tcolor[ya:yb, xa:xb][upd] = col[upd]
        theight[ya:yb, xa:xb][upd] = hz[upd]
        crown[ya:yb, xa:xb][upd] = 1
leaf = noise(3, 1, 30)
leaf2 = noise(9, 2, 31)
leafmask = (leaf2 + 0.6 * leaf)
tcolor *= (1 + 0.16 * leaf + 0.14 * leaf2)[..., None]
tcolor[(leafmask > 1.6) & (crown > 0)] *= 1.15          # sunlit leaf glints
tcolor[(leafmask < -1.7) & (crown > 0)] *= 0.55         # holes into canopy
cav = ndi.gaussian_filter(theight, 8) - theight
tcolor *= np.clip(1 - 0.015 * np.clip(cav, 0, None), 0.45, 1)[..., None]
crown_a = ndi.gaussian_filter(crown, 0.8)

# --- shadows
th = ndi.gaussian_filter(crown * np.clip(theight / 55, 0.5, 1.3), 4)
sh = ndi.shift(th, (34, 38), order=1)
sh = np.clip(sh, 0, 1) * (1 - crown_a)
img *= (1 - 0.55 * sh)[..., None] * np.where(sh[..., None] > 0.1, np.array([0.97, 0.99, 1.04]), 1)
ao = ndi.gaussian_filter(crown, 14) * (1 - crown_a)
img *= (1 - 0.3 * ao)[..., None]
img = img * (1 - crown_a[..., None]) + tcolor * crown_a[..., None]

# --- pin
py_, px_ = pin * S + S / 2
hole = (yy - py_) ** 2 + (xx - px_) ** 2 < 5 ** 2
img[hole] = [0.1, 0.11, 0.08]
for t in range(70):
    a_, b_ = int(py_ + t * 0.55), int(px_ + t * 0.7)
    img[a_ - 1:a_ + 2, b_ - 1:b_ + 2] *= 0.72
img[int(py_ - 2):int(py_ + 2), int(px_ - 2):int(px_ + 2)] = [0.95, 0.95, 0.93]
flag = (xx > px_ + 1) & (xx < px_ + 22) & (yy > py_ - 3) & (yy < py_ + 10)
img[flag] = [0.82, 0.12, 0.1]

# --- photographic finish: split tone, mild desat, soft focus mix, sensor grain
img = np.clip(img, 0, 1)
lum = (img * [0.3, 0.59, 0.11]).sum(-1, keepdims=True)
img = lum + (img - lum) * 0.92
img += (lum - 0.35) * np.array([0.04, 0.015, -0.03])            # warm highs / cool lows
soft_ = ndi.gaussian_filter(img, (1.2, 1.2, 0))
img = img * 0.8 + soft_ * 0.2
img += 0.012 * np.random.default_rng(50).standard_normal((H, W, 1))
img = np.clip(img, 0, 1) ** 0.97

edge_feather = ndi.gaussian_filter(course_alpha, 1.0)
alpha = np.clip(np.maximum(edge_feather, crown_a), 0, 1)
rgba = np.dstack([np.clip(img, 0, 1), alpha])
out = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
out.save('golf_real_cutout.png')
bgw = Image.new('RGBA', out.size, (255, 255, 255, 255))
bgw.alpha_composite(out)
bgw.convert('RGB').save('golf_real_onwhite.jpg', quality=92)
print('size', out.size, 'trees', len(placed))
