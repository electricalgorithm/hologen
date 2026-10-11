"""
Measure the per-cell statistics of real adherent cells that the procedural cell
model (hologen.objects.adherent_cells) is calibrated to.

Data: Vicar, Balvan, Gumulec, Quantitative phase microscopy labelled and unlabelled
dataset of various adherent cell lines, Zenodo, doi:10.5281/zenodo.4771831
(CC BY 4.0). The images are dry-mass density m in pg/um^2; the optical path
difference is OPD = alpha * m with alpha = 0.18 um^3/pg (Vicar et al., Sci. Rep.
10:1566, 2020). The pixel size is not published: 0.63 um is inferred from Vicar et
al., Biophys. J. 121:1632 (2022), and can be changed with --pixel-size.

Run with:

    uv run --with tifffile --with imagecodecs scripts/calibrate_adherent_cells.py \\
        ../datasets/annoted-quantitiative-phase-microscopy-cell/labelled
"""

import argparse
import glob
import pathlib

import matplotlib.pyplot as plt
import numpy
import tifffile

ALPHA: float = 0.18  # um^3 / pg


def cell_statistics(opd: numpy.ndarray, cell: numpy.ndarray, pixel: float) -> dict:
    """Size, elongation and optical path statistics of one segmented cell."""
    ys, xs = numpy.nonzero(cell)
    area = cell.sum() * pixel**2
    covariance = numpy.cov(numpy.vstack([xs, ys]))
    eigen = numpy.sort(numpy.linalg.eigvalsh(covariance))
    values = opd[cell]
    peak = numpy.percentile(values, 95)
    return {
        "diameter": 2 * numpy.sqrt(area / numpy.pi),
        "aspect": float(numpy.sqrt(eigen[1] / max(eigen[0], 1e-9))),
        "peak_opd": float(peak),
        "fill": float(values.mean() / peak) if peak > 0 else numpy.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "labelled", type=pathlib.Path, help="folder with one folder per cell line"
    )
    parser.add_argument("--pixel-size", type=float, default=0.63, help="um per pixel")
    parser.add_argument("--fields", type=int, default=40, help="fields per cell line")
    args = parser.parse_args()

    rows: list[dict] = []
    coverage: list[float] = []
    for line in sorted(p.name for p in args.labelled.iterdir() if p.is_dir()):
        masks = sorted(glob.glob(str(args.labelled / line / "*_mask.png")))[
            : args.fields
        ]
        for mask_path in masks:
            labels = numpy.round(plt.imread(mask_path) * 255).astype(int)
            if labels.ndim == 3:
                labels = labels[..., 0]
            opd = ALPHA * tifffile.imread(mask_path.replace("_mask.png", "_img.tif"))
            coverage.append(float(numpy.mean(labels > 0)))
            for label in numpy.unique(labels[labels > 0]):
                stats = cell_statistics(opd, labels == label, args.pixel_size)
                stats["line"] = line
                rows.append(stats)

    def lognormal(values: list[float]) -> str:
        logs = numpy.log(numpy.asarray(values))
        return f"median {numpy.exp(logs.mean()):.3f}, log s.d. {logs.std():.3f}"

    print(f"{len(rows)} cells, {len(coverage)} fields, pixel {args.pixel_size} um")
    for key, unit in [("diameter", "um"), ("aspect", ""), ("peak_opd", "um")]:
        values = [r[key] for r in rows if numpy.isfinite(r[key]) and r[key] > 0]
        print(
            f"{key} ({unit}): lognormal {lognormal(values)}; p5-p95 {numpy.percentile(values, 5):.3f}-{numpy.percentile(values, 95):.3f}"
        )
    fills = [r["fill"] for r in rows if numpy.isfinite(r["fill"])]
    print(
        f"fill (mean OPD / peak OPD): median {numpy.median(fills):.3f}, p10-p90 {numpy.percentile(fills, 10):.3f}-{numpy.percentile(fills, 90):.3f}"
    )
    print(
        f"field coverage by cells: median {numpy.median(coverage):.3f}, p10-p90 {numpy.percentile(coverage, 10):.3f}-{numpy.percentile(coverage, 90):.3f}"
    )
    peak = numpy.array([r["peak_opd"] for r in rows])
    diameter = numpy.array([r["diameter"] for r in rows])
    ok = (peak > 0) & (diameter > 0)
    print(
        f"correlation of log diameter and log peak OPD: {numpy.corrcoef(numpy.log(diameter[ok]), numpy.log(peak[ok]))[0, 1]:.3f}"
    )


if __name__ == "__main__":
    main()
