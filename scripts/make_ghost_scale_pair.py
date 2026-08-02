"""Rebuild figures/ghost_scale_pair.png from the two source images.

THE BUG THIS FIXES. The pair was composed on a paper-coloured canvas, which put a near-white margin
around both panels and down the middle. That is wrong on its own terms: the two panels are supposed
to be distinguishable at a glance BY THEIR BORDERS, black for the hand-drawn one and grey for the
rendered one, and a white surround flattens both into the page. On a dark background it reads as a
white box, which is worse.

Both source images already carry their own correct border. The drawn one is black to its corners;
the rendered one is grey to its corners. So the fix is not to add borders, it is to stop adding the
margin: scale the two to a common height and butt them together with nothing between.

    python scripts/make_ghost_scale_pair.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"

LEFT = FIG / "Ghostscale_creator.png"      # 100% intent, drawn by a person
RIGHT = FIG / "Ghostscale_curator.jpeg"    # 60% intent, rendered by a machine
OUT = FIG / "ghost_scale_pair.png"

GUTTER = 8          # a thin seam so the two panels do not appear to be one image
SEAM = (24, 24, 26)  # near-black, so it reads as a division rather than as background


def main() -> None:
    left = Image.open(LEFT).convert("RGB")
    right = Image.open(RIGHT).convert("RGB")

    height = max(left.height, right.height)

    def scaled(im: Image.Image) -> Image.Image:
        if im.height == height:
            return im
        w = round(im.width * height / im.height)
        return im.resize((w, height), Image.LANCZOS)

    left, right = scaled(left), scaled(right)
    canvas = Image.new("RGB", (left.width + GUTTER + right.width, height), SEAM)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + GUTTER, 0))
    canvas.save(OUT)

    px = canvas.convert("RGB")
    corners = [px.getpixel(p) for p in
               ((0, 0), (canvas.width - 1, 0), (0, height - 1), (canvas.width - 1, height - 1))]
    print(f"wrote {OUT.relative_to(REPO)}  {canvas.size}")
    print(f"  corner tones: {corners}")
    assert all(sum(c) < 400 for c in corners[:1]), "left edge should be dark"


if __name__ == "__main__":
    main()
