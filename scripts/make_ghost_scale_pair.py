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

BORDER = 46             # thick enough that the two tones are the first thing you see
BLACK = (0, 0, 0)       # the drawn panel: 100% intent
GREY = (101, 104, 97)   # the rendered panel, sampled from its own existing edge


def main() -> None:
    left = Image.open(LEFT).convert("RGB")
    right = Image.open(RIGHT).convert("RGB")

    # Scale to a common CONTENT height first, so the two borders come out the same thickness.
    # Bordering first and scaling after would make one frame visibly thinner than the other.
    height = max(left.height, right.height)

    def scaled(im: Image.Image) -> Image.Image:
        if im.height == height:
            return im
        return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)

    left, right = scaled(left), scaled(right)

    def framed(im: Image.Image, colour) -> Image.Image:
        out = Image.new("RGB", (im.width + 2 * BORDER, im.height + 2 * BORDER), colour)
        out.paste(im, (BORDER, BORDER))
        return out

    left, right = framed(left, BLACK), framed(right, GREY)

    # No gutter. The two frames meet, so black against grey IS the division, which is the whole
    # point of giving them different border tones in the first place.
    canvas = Image.new("RGB", (left.width + right.width, left.height), BLACK)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(OUT)

    px = canvas.convert("RGB")
    corners = {"top-left": px.getpixel((0, 0)),
               "top-right": px.getpixel((canvas.width - 1, 0)),
               "bottom-left": px.getpixel((0, canvas.height - 1)),
               "bottom-right": px.getpixel((canvas.width - 1, canvas.height - 1))}
    print(f"wrote {OUT.relative_to(REPO)}  {canvas.size}  border {BORDER}px")
    for k, v in corners.items():
        print(f"  {k:<13} {v}")
    assert corners["top-left"] == BLACK, "the drawn panel must be framed in black"
    assert corners["top-right"] == GREY, "the rendered panel must be framed in grey"


if __name__ == "__main__":
    main()
