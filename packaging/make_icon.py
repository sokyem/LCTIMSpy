"""Render lctimspy.ico: the UI's mark (white chromatogram on blue), at every Windows size.

    python packaging/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

BLUE = (30, 64, 175, 255)        # --primary
SIZE = 1024                      # draw big, downsample for crisp small sizes


def trace(n: int = 400) -> list[tuple[float, float]]:
    import math

    pts = []
    for i in range(n + 1):
        x = i / n
        # Two isomer peaks of different height on a flat baseline.
        y = 0.92 * math.exp(-((x - 0.34) / 0.075) ** 2) + 0.58 * math.exp(-((x - 0.68) / 0.07) ** 2)
        pts.append((0.14 + 0.72 * x, 0.78 - 0.56 * y))
    return pts


def main() -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=int(SIZE * 0.2), fill=BLUE)
    # A thick stroke as densely overlapping disks: PIL's wide polylines leave
    # hairline gaps at the joints of a curve this dense.
    r = SIZE * 0.0375
    for x, y in trace(2000):
        x, y = x * SIZE, y * SIZE
        d.ellipse((x - r, y - r, x + r, y + r), fill="white")
    out = Path(__file__).with_name("lctimspy.ico")
    img.save(out, sizes=[(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
    img.resize((256, 256), Image.LANCZOS).save(out.with_suffix(".png"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
