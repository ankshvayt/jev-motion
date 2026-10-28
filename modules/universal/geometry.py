"""Focus-crop geometry. Pure functions, no I/O.

A window shows a source frame `object-fit: cover`, top-centre, inside a body of `W x H`; the focus helper then scales it
by `S` about a transform origin `(ox, oy)` in body pixels. Rectangles are `(x0, y0, x1, y1)` in source pixels unless
a name says otherwise.

  cover factor c = max(W / sw, H / sh); the sides the cover trims off are `off = (sw * c - W) / 2` body px on each side.
  visible body region before the transform: x in [ox - ox/S, ox + (W - ox)/S], y in [oy - oy/S, oy + (H - oy)/S]
  body -> source: x_src = (x + off) / c, y_src = y / c

`focus_rect` maps focus arguments to the visible source rectangle and `rect_to_focus` inverts it. Scaling only magnifies
what the cover already shows, so a window can never leave `cover_region`; `to_focus` keeps it there.
"""
PAD = 0.08
EPS = 1e-6


def centre(r):
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def _cover(sw, sh, body):
    """`(W, H, c, off)` for a `sw x sh` source in `body`."""
    W, H = body
    c = max(W / sw, H / sh)
    return W, H, c, (sw * c - W) / 2


def focus_rect(sw, sh, body, scale, ox, oy):
    """The source rectangle a window shows for focus `(scale, ox, oy)`."""
    W, H, c, off = _cover(sw, sh, body)
    x0, x1 = ox - ox / scale, ox + (W - ox) / scale
    y0, y1 = oy - oy / scale, oy + (H - oy) / scale
    return (x0 + off) / c, y0 / c, (x1 + off) / c, y1 / c


def rect_to_focus(sw, sh, body, rect):
    """Inverse of `focus_rect`: `(scale, x, y)` whose visible source rectangle is `rect` (the body's aspect).

    At scale 1 the origin has no effect, so it comes back as the body centre instead of dividing by zero.
    """
    W, H, c, off = _cover(sw, sh, body)
    scale = W / (c * (rect[2] - rect[0]))
    k = 1 - 1 / scale
    if abs(k) < 1e-9:
        return scale, W / 2, H / 2
    return scale, (rect[0] * c - off) / k, (rect[1] * c) / k


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def centre_inside(pred, hand):
    """True when the centre of `pred` lies inside `hand` (edges included)."""
    cx, cy = centre(pred)
    return hand[0] <= cx <= hand[2] and hand[1] <= cy <= hand[3]


def expand(box, aspect, iw, ih, pad=PAD):
    """Grow `box` to the window `aspect` (about its centre), add `pad` of the size on each side, then fit it inside the
    `iw x ih` image: scaled down if larger than the image, otherwise shifted to stay inside."""
    cx, cy = centre(box)
    w, h = box[2] - box[0], box[3] - box[1]
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w *= 1 + 2 * pad
    h *= 1 + 2 * pad
    s = min(1.0, iw / w, ih / h)
    w, h = w * s, h * s
    x0 = min(max(cx - w / 2, 0.0), iw - w)
    y0 = min(max(cy - h / 2, 0.0), ih - h)
    return x0, y0, x0 + w, y0 + h


def cover_region(sw, sh, body):
    """The part of the source a window can show: what `object-fit: cover` keeps at scale 1 (top-aligned, centred across),
    as `(x0, y0, x1, y1)` in source px. A source wider than the body's aspect loses its sides, a taller one its bottom."""
    W, H, c, off = _cover(sw, sh, body)
    return off / c, 0.0, (W + off) / c, H / c


def _within(start, size, lo, hi):
    """`start` moved the least that puts `[start, start + size]` inside `[lo, hi]`; centred when it cannot fit."""
    if size >= hi - lo:
        return lo + (hi - lo - size) / 2
    return min(max(start, lo), hi - size)


def to_focus(sw, sh, body, rect, scale_range):
    """Focus arguments for `rect` (window aspect, inside the frame) as `{scale, x, y, rect, clamped}`.

    A scale outside `scale_range` is clamped; the visible rectangle then has the clamped size and is centred on `rect`'s
    centre. The rectangle is then shifted, never shrunk, to stay inside `cover_region`, and `x`, `y` come from it.
    `rect` in the result is the rectangle the arguments show. `clamped` lists what moved, in the order `scale`, `x`,
    `y`; it is empty when `rect` was usable as it came (a move under 1e-6 does not count).
    """
    scale, x, y = rect_to_focus(sw, sh, body, rect)
    lo, hi = scale_range
    clamped = []
    if not lo - EPS <= scale <= hi + EPS:
        clamped.append("scale")
        scale = min(max(scale, lo), hi)
        W, H, c, _ = _cover(sw, sh, body)
        w, h = W / (c * scale), H / (c * scale)
        cx, cy = centre(rect)
        rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    rx0, ry0, rx1, ry1 = cover_region(sw, sh, body)
    x0 = _within(rect[0], rect[2] - rect[0], rx0, rx1)
    y0 = _within(rect[1], rect[3] - rect[1], ry0, ry1)
    clamped += [axis for axis, moved in (("x", x0 - rect[0]), ("y", y0 - rect[1])) if abs(moved) > EPS]
    if clamped:
        rect = (x0, y0, x0 + rect[2] - rect[0], y0 + rect[3] - rect[1])
        _, x, y = rect_to_focus(sw, sh, body, rect)
    return {"scale": scale, "x": x, "y": y, "rect": tuple(rect), "clamped": clamped}
