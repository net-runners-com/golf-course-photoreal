import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
"""Step 5/6: pad render for ChatGPT, cut out the photoreal hole, composite it over the generated forest.

usage:
  python postprocess.py pad        # golf_real_onwhite.jpg -> gen_input.png (2:3, 1024x1536)
  python postprocess.py cutout     # gen/v3_photoreal_raw.png -> gen_v3.png, golf_photoreal_cutout.png
  python postprocess.py scene      # gen/v4_surround_raw.png + gen_v3.png -> golf_photoreal_scene.png
"""
import sys
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi

W, H = 2080, 3120          # 2:3 canvas at render height
PAD_X = 300                # (2080 - 1480) / 2


def load(path, size=(W, H)):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    return cv2.resize(a, size, interpolation=cv2.INTER_LANCZOS4)


def pad():
    im = Image.open('golf_real_onwhite.jpg')
    w, h = im.size
    c = Image.new('RGB', (int(h * 2 / 3), h), 'white')
    c.paste(im, ((c.width - w) // 2, 0))
    c.resize((1024, 1536), Image.LANCZOS).save('gen_input.png')


def cutout():
    Image.open('gen/v3_photoreal_raw.png').save('gen_v3.png')
    g = load('gen_v3.png')[:, PAD_X:PAD_X + 1480]
    white = g.min(-1) > 238
    lab, _ = ndi.label(white)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, border[border > 0])
    a = ndi.gaussian_filter((~bg).astype(np.float32), 1.0)
    Image.fromarray(np.dstack([np.clip(g, 0, 255), a * 255]).astype(np.uint8), 'RGBA').save('golf_photoreal_cutout.png')
    # layout check against the original masks
    ours = (np.asarray(Image.open('masks/course_alpha.png')) > 127) | (np.asarray(Image.open('masks/tree.png')) > 127)
    theirs = ~bg
    print('silhouette IoU', (ours & theirs).sum() / (ours | theirs).sum())


def scene():
    bg = load('gen/v4_surround_raw.png')
    fg = load('gen_v3.png')
    # hole mask = everything that is not flat white (sand is warmer, so it survives)
    mn, mx = fg.min(-1), fg.max(-1)
    whiteish = (mn > 205) & (mx - mn < 22)
    l, n = ndi.label(whiteish)
    sz = ndi.sum(whiteish, l, np.arange(1, n + 1))
    m = ~ndi.binary_dilation(np.isin(l, 1 + np.where(sz > 150)[0]), iterations=6)
    # the generator shifted the hole slightly: align fg -> bg with ECC (affine) inside the hole
    g1 = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    cc, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5),
                                    (m * 255).astype(np.uint8), 5)
    print('ECC', cc, 'shift', warp[:, 2])
    fgw = cv2.warpAffine(fg, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    mw = cv2.warpAffine(m.astype(np.float32), warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    a = ndi.gaussian_filter(ndi.binary_erosion(mw > 0.5, iterations=10).astype(np.float32), 8)[..., None]
    c = (bg * (1 - a) + fgw * a).clip(0, 255).astype(np.uint8)
    # leftover white specks near the tees (bunker highlights live above y=2000, keep them)
    w = (c.min(-1) > 205) & (c.max(-1).astype(int) - c.min(-1) < 22)
    w[:2000] = False
    c = cv2.inpaint(c, ndi.binary_dilation(w, iterations=4).astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    Image.fromarray(c).save('golf_photoreal_scene.png')
    Image.fromarray(bg.clip(0, 255).astype(np.uint8)).save('golf_photoreal_scene_raw.png')


if __name__ == '__main__':
    {'pad': pad, 'cutout': cutout, 'scene': scene}[sys.argv[1]]()
