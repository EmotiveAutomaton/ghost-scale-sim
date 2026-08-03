"""Bundle a carousel's plates into one PDF, in posting order.

    python scripts/make_carousel_pdf.py

Writes OUTSIDE the repository, next to OUTREACH_GUIDE.md, because a carousel is a posting artefact
and not a result. The repo holds the plates; the parent folder holds the things made out of them.

THE PAGE IS THE IMAGE. LinkedIn re-renders an uploaded document page by page, so a page with
margins around the plate arrives as a plate with margins, and the plate already has its own border.
Each page is therefore exactly the size of the image it carries, at the image's own resolution.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PLATES = REPO / "figures" / "walkthrough"
OUT_DIR = REPO.parent                      # the Ghost Scale Simulation folder, outside the repo

CAROUSELS = {
    "carousel-2-the-case-for-intent": [
        "11_the_master_cannot_explain",
        "05_intent_unlocks_the_method",
        "19_reading_an_unseen_intent",
        "24_what_its_made_of",
        "20_rejection_is_not_protection",
    ],
    "carousel-1-aimed-at-the-wrong-thing": [
        "25_a_defence_that_works",
        "01_false_label_moves_you_wrong",
        "02_invention_peaks_in_the_middle",
        "03_legible_and_empty",
        "22_how_much_is_the_theory",
        "26_the_arms_race",
    ],
}


def build(name: str, stems: list[str]) -> Path:
    pages = []
    for stem in stems:
        src = PLATES / f"{stem}.png"
        if not src.exists():
            raise SystemExit(f"missing plate: {src}")
        # Flattened onto white. A PNG with alpha writes a PDF page some viewers render dark.
        im = Image.open(src).convert("RGBA")
        page = Image.new("RGB", im.size, (255, 255, 255))
        page.paste(im, mask=im.split()[3])
        pages.append(page)

    out = OUT_DIR / f"{name}.pdf"
    pages[0].save(out, "PDF", resolution=160.0, save_all=True, append_images=pages[1:])
    return out


def main() -> None:
    for name, stems in CAROUSELS.items():
        out = build(name, stems)
        size = out.stat().st_size / 1024
        print(f"{out.name}: {len(stems)} pages, {size:.0f} KB")
        for i, s in enumerate(stems, 1):
            print(f"   {i}. {s}")
    print(f"\nwrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
