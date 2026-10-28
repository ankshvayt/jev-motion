"""Images for Jev and for people: fit to Jev's limits, cut frames from a render, label and tile them.

Jev takes "PNG or JPEG; each side 280-1568 px; total 78,400-2,458,624 px"; it rejects out-of-range images instead of
resizing them and does not apply EXIF rotation. Stdlib plus the ffmpeg and ffprobe CLIs; this ffmpeg has no drawtext, so
labels come from `textimg` and are composited with overlay. Paths may contain spaces: every path is one argv item.

A `Fit` says how a source image became the sent one, so a box Jev answers in fractions of the sent image maps back to
source pixels (`unfit_box`). EXIF orientation is ignored on purpose (`-noautorotate`): Jev ignores it, and the source
pixel grid stays the one `image_size` reports.
"""
import json
import math
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from jevlib import textimg

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

MIN_SIDE = 280
MAX_SIDE = 1568
MIN_AREA = 78_400
MAX_AREA = 2_458_624

_JPEG_SOF = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
_RECT_COLORS = ("red", "yellow", "cyan", "lime")


class ImageLimitError(ValueError):
    """The image cannot go to Jev as it is: unreadable, or outside the size limits."""


@dataclass(frozen=True)
class Fit:
    """How `src_w` x `src_h` became the sent `w` x `h`: scaled by `scale`, then padded (left, top, right, bottom)."""
    src_w: int
    src_h: int
    scale: float
    pad: tuple
    w: int
    h: int


def _run(args):
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace").strip()[-400:]
        raise RuntimeError(f"{Path(args[0]).name} failed ({r.returncode}): {tail}")
    return r


def _even(x):
    return max(2, round(x / 2) * 2)


# ---------------------------------------------------------------- headers
def _jpeg_segments(f):
    """(marker, payload) for each JPEG segment after SOI, up to and including SOS."""
    while True:
        b = f.read(1)
        if not b:
            return
        if b != b"\xff":
            continue
        m = f.read(1)
        while m == b"\xff":
            m = f.read(1)
        if not m:
            return
        mk = m[0]
        if mk in (0x00, 0x01) or 0xD0 <= mk <= 0xD8:
            continue
        if mk == 0xD9:
            return
        seg = f.read(2)
        if len(seg) < 2:
            return
        yield mk, f.read(struct.unpack(">H", seg)[0] - 2)
        if mk == 0xDA:
            return


def image_size(path):
    """(width, height, "png" | "jpg") from the file header."""
    with open(path, "rb") as f:
        head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            w, h = struct.unpack(">II", head[16:24])
            return w, h, "png"
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            for mk, payload in _jpeg_segments(f):
                if mk in _JPEG_SOF and len(payload) >= 5:
                    h, w = struct.unpack(">HH", payload[1:5])
                    return w, h, "jpg"
    raise ImageLimitError(f"{path}: not a readable PNG or JPEG")


def has_exif(path):
    """True for a JPEG with an APP1 "Exif" segment. A PNG never counts: it is sent unchanged."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return False
        return any(mk == 0xE1 and payload[:6] == b"Exif\x00\x00" for mk, payload in _jpeg_segments(f))


def check_limits(path):
    """Raise ImageLimitError unless the image is inside Jev's side and area limits."""
    w, h, _ = image_size(path)
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise ImageLimitError(f"{path}: {w}x{h} outside {MIN_SIDE}-{MAX_SIDE}px per side")
    if not MIN_AREA <= w * h <= MAX_AREA:
        raise ImageLimitError(f"{path}: area {w * h} outside {MIN_AREA}-{MAX_AREA}")


# ---------------------------------------------------------------- video
def probe_video(path):
    """{"width", "height", "duration_s", "fps"} of the first video stream."""
    out = _run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration", "-of", "json",
                str(path)]).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError(f"{path}: no video stream")
    stream = data["streams"][0]
    duration = None
    for raw in (data.get("format", {}).get("duration"), stream.get("duration")):
        try:
            duration = float(raw)
            break
        except (TypeError, ValueError):
            continue
    if duration is None:
        raise ValueError(f"{path}: no duration")
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps = float(Fraction(stream.get(key) or "0/0"))
        except (ValueError, ZeroDivisionError):
            continue
        if fps > 0:
            break
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration_s": duration,
            "fps": round(fps, 3)}


def sample_times(duration_s, every):
    """Seconds 0, every, 2*every ... that fall inside the render; always at least [0.0]."""
    if every <= 0:
        raise ValueError("every must be positive")
    times = [round(i * every, 3) for i in range(int((duration_s - 0.05) / every) + 1)]
    return times or [0.0]


def frame_at(video, t, out, long_edge=MAX_SIDE):
    """One JPEG frame at `t` seconds (exact seek: -ss before -i), long side `long_edge`, even dimensions."""
    info = probe_video(video)
    w, h = info["width"], info["height"]
    k = long_edge / max(w, h)
    sw, sh = _even(w * k), _even(h * k)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    _run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf",
          f"scale={sw}:{sh}", "-q:v", "3", "-c:v", "mjpeg", "-f", "image2", "-update", "1", str(out)])
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"no frame at t={t}s in {video}")
    return out, Fit(w, h, sw / w, (0, 0, 0, 0), sw, sh)


# ---------------------------------------------------------------- fitting to Jev
def _split(total):
    """Split padding `total` into (before, after), `before` even: the pad filter rounds offsets down to even for 4:2:0."""
    before = total // 4 * 2
    return before, total - before


def fit_for_jev(src, out):
    """Write to `out` an image Jev accepts, and say how it was made.

    A PNG, or a JPEG without EXIF, already inside the limits is copied byte for byte. Anything else is re-encoded as
    JPEG (which strips EXIF): scaled up so the short side reaches 280 px, then down so the long side is at most 1568 px,
    then the short side is padded with black, centred, up to 280 px. The result is JPEG whatever `out` is called.
    """
    src, out = Path(src), Path(out)
    w, h, kind = image_size(src)
    try:
        check_limits(src)
        within = True
    except ImageLimitError:
        within = False
    out.parent.mkdir(parents=True, exist_ok=True)
    if within and (kind == "png" or not has_exif(src)):
        if src.resolve() != out.resolve():
            shutil.copyfile(src, out)
        return out, Fit(w, h, 1.0, (0, 0, 0, 0), w, h)
    if src.resolve() == out.resolve():
        raise ValueError(f"{src}: out must not be the source file when the image is re-encoded")
    k = max(MIN_SIDE / min(w, h), 1.0)
    if max(w, h) * k > MAX_SIDE:
        k = MAX_SIDE / max(w, h)
    sw, sh = (w, h) if k == 1.0 else (_even(w * k), _even(h * k))
    pl, pr = _split(MIN_SIDE - sw if sw < MIN_SIDE else 0)
    pt, pb = _split(MIN_SIDE - sh if sh < MIN_SIDE else 0)
    fw, fh = sw + pl + pr, sh + pt + pb
    parts = []
    if (sw, sh) != (w, h):
        parts.append(f"scale={sw}:{sh}:flags=lanczos")
    if (fw, fh) != (sw, sh):
        parts.append(f"pad={fw}:{fh}:{pl}:{pt}:black")
    _run([FFMPEG, "-v", "error", "-y", "-noautorotate", "-i", str(src), "-map_metadata", "-1", "-frames:v", "1",
          "-vf", ",".join(parts) or "null", "-q:v", "3", "-c:v", "mjpeg", "-f", "image2", "-update", "1", str(out)])
    return out, Fit(w, h, k, (pl, pt, pr, pb), fw, fh)


def unfit_box(frac, fit):
    """Map a box in fractions of the sent image (`left`, `top`, `right`, `bottom`) to (x0, y0, x1, y1) source pixels.

    The padding is taken off and the result is clamped to the source image. The scale comes from the sent and
    source sizes, so the rounding to even dimensions does not shift the box.
    """
    pl, pt, pr, pb = fit.pad
    cw, ch = fit.w - pl - pr, fit.h - pt - pb

    def x(f):
        return min(max((f * fit.w - pl) * fit.src_w / cw, 0.0), float(fit.src_w))

    def y(f):
        return min(max((f * fit.h - pt) * fit.src_h / ch, 0.0), float(fit.src_h))

    return x(frac["left"]), y(frac["top"]), x(frac["right"]), y(frac["bottom"])


# ---------------------------------------------------------------- labels and contact sheets
def label_png(text, out, scale=2):
    """White-on-black bitmap text as a PNG (no drawtext in this ffmpeg)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    textimg.render(text, out, scale=scale)
    return out


def tile_sheet(tiles, out, cols=4, tile_w=300):
    """Contact sheet of `tiles` = [(image, label)], `tile_w` px wide, `cols` per row, label burned in at 4,4.

    Tiles of different shapes share one cell height (the tallest). Returns None for an empty list.
    """
    if not tiles:
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    heights = []
    for path, _ in tiles:
        w, h, _ = image_size(path)
        heights.append(_even(tile_w * h / w))
    cell_h = max(heights)
    cols = min(cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    with tempfile.TemporaryDirectory(prefix="jevm-tiles-") as td:
        td = Path(td)
        for i, ((path, label), th) in enumerate(zip(tiles, heights)):
            lab = label_png(label, td / f"label_{i}.png")
            _run([FFMPEG, "-v", "error", "-y", "-noautorotate", "-i", str(path), "-i", str(lab), "-filter_complex",
                  f"[0:v]scale={tile_w}:{th},pad={tile_w}:{cell_h}:0:0:0x202020[a];[a][1:v]overlay=4:4",
                  "-frames:v", "1", "-update", "1", str(td / f"lab_{i:02d}.png")])
        _run([FFMPEG, "-v", "error", "-y", "-start_number", "0", "-framerate", "1", "-i", str(td / "lab_%02d.png"),
              "-vf", f"tile={cols}x{rows}:padding=4:color=0x202020", "-frames:v", "1", "-update", "1", "-q:v", "3",
              str(out)])
    return out


def draw_rects(image, rects, out, thickness=6):
    """Draw `rects` = [((x0, y0, x1, y1), label)] in pixels of `image` as coloured boxes, each with its label above it."""
    image, out = Path(image), Path(out)
    w, h, _ = image_size(image)
    out.parent.mkdir(parents=True, exist_ok=True)
    label_scale = max(2, round(max(w, h) / 500))
    boxes, overlays = [], []
    with tempfile.TemporaryDirectory(prefix="jevm-rects-") as td:
        td = Path(td)
        for i, ((x0, y0, x1, y1), label) in enumerate(rects):
            x0, y0 = min(max(round(x0), 0), w - 1), min(max(round(y0), 0), h - 1)
            x1, y1 = min(round(x1), w), min(round(y1), h)
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            color = _RECT_COLORS[i % len(_RECT_COLORS)]
            boxes.append(f"drawbox=x={x0}:y={y0}:w={bw}:h={bh}:color={color}:t={thickness}")
            if label:
                lab = label_png(label, td / f"label_{i}.png", scale=label_scale)
                lw, lh, _ = image_size(lab)
                overlays.append((lab, min(x0, max(0, w - lw)), y0 - lh if y0 >= lh else y0))
        graph = "[0:v]" + (",".join(boxes) or "null") + "[v0]"
        for n, (_, lx, ly) in enumerate(overlays, start=1):
            graph += f";[v{n - 1}][{n}:v]overlay={lx}:{ly}[v{n}]"
        args = [FFMPEG, "-v", "error", "-y", "-noautorotate", "-i", str(image)]
        for lab, _, _ in overlays:
            args += ["-i", str(lab)]
        _run(args + ["-filter_complex", graph, "-map", f"[v{len(overlays)}]", "-frames:v", "1", "-update", "1",
                     "-q:v", "3", str(out)])
    return out


def mean_luma(image):
    """Mean luma of an image, 0 (black) to 255 (white)."""
    raw = _run([FFMPEG, "-v", "error", "-noautorotate", "-i", str(image), "-frames:v", "1", "-pix_fmt", "gray",
                "-f", "rawvideo", "-"]).stdout
    if not raw:
        raise RuntimeError(f"{image}: no pixels read")
    return sum(raw) / len(raw)
