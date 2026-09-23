"""Technical and acquisition-level QC for the Commons candidate catalog.

This script does *not* decide whether a cue is religious, ambient, Shinto,
Buddhist, Christian, or secular. It only flags technical/metadata risks and
possible archival/non-naturalistic material for human Study 0 review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

CATALOG = Path("results/commons_candidate_catalog.csv")
OUT = Path("results")

STANDARD_LICENSE_PREFIXES = ("CC BY", "CC0", "Public domain")
RASTER_MIME = {"image/jpeg", "image/png"}

ARCHIVAL_RE = re.compile(
    r"(illustrat|book page|book online|nypl|postcard|postkaart|floor plan|diagram|"
    r"scan by|asset\s|woodblock|engraving|historical photograph|old and new japan)",
    re.I,
)


def blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def main() -> None:
    if not CATALOG.exists():
        raise FileNotFoundError(CATALOG)

    df = pd.read_csv(CATALOG)
    if df.empty:
        raise RuntimeError("candidate catalog is empty")

    df["short_side"] = df[["width", "height"]].min(axis=1)
    df["flag_low_resolution"] = df["short_side"] < 768
    df["flag_tiny"] = df["short_side"] < 512
    df["flag_non_raster"] = ~df["media_type"].isin(RASTER_MIME)
    df["flag_missing_creator"] = blank(df["creator"])
    df["flag_missing_description"] = blank(df["description"])
    df["flag_missing_source_url"] = blank(df["source_page_url"])
    df["flag_missing_thumbnail"] = blank(df["thumbnail_url"])
    df["flag_nonstandard_license"] = ~df["license_short"].fillna("").astype(str).str.startswith(
        STANDARD_LICENSE_PREFIXES
    )

    text = (
        df["commons_title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
        + " "
        + df["credit"].fillna("").astype(str)
    )
    df["flag_archival_or_diagram"] = text.str.contains(ARCHIVAL_RE, regex=True)

    hard_flags = [
        "flag_low_resolution",
        "flag_non_raster",
        "flag_missing_source_url",
        "flag_missing_thumbnail",
        "flag_nonstandard_license",
    ]
    soft_flags = [
        "flag_missing_creator",
        "flag_missing_description",
        "flag_archival_or_diagram",
    ]
    df["technical_pass"] = ~df[hard_flags].any(axis=1)
    df["n_hard_flags"] = df[hard_flags].sum(axis=1)
    df["n_soft_flags"] = df[soft_flags].sum(axis=1)
    df["manual_review_priority"] = (
        (~df["technical_pass"]).astype(int) * 10 + df["n_soft_flags"]
    )

    qc_path = OUT / "commons_candidate_catalog_qc.csv"
    df.to_csv(qc_path, index=False)

    by_cat = (
        df.groupby(["source_family", "acquisition_role_hint", "source_category"], dropna=False)
        .agg(
            n=("cue_id", "size"),
            technical_pass=("technical_pass", "sum"),
            low_resolution=("flag_low_resolution", "sum"),
            non_raster=("flag_non_raster", "sum"),
            missing_creator=("flag_missing_creator", "sum"),
            missing_description=("flag_missing_description", "sum"),
            archival_or_diagram=("flag_archival_or_diagram", "sum"),
            nonstandard_license=("flag_nonstandard_license", "sum"),
            median_short_side=("short_side", "median"),
        )
        .reset_index()
    )
    by_cat.to_csv(OUT / "commons_catalog_qc_by_category.csv", index=False)

    review = df[
        (~df["technical_pass"])
        | df["flag_archival_or_diagram"]
        | df["flag_missing_creator"]
        | df["flag_missing_description"]
    ].sort_values(["manual_review_priority", "source_family"], ascending=[False, True])
    review.to_csv(OUT / "commons_catalog_manual_review.csv", index=False)

    summary = {
        "n_candidates": int(len(df)),
        "n_technical_pass": int(df["technical_pass"].sum()),
        "technical_pass_rate": float(df["technical_pass"].mean()),
        "n_low_resolution": int(df["flag_low_resolution"].sum()),
        "n_tiny": int(df["flag_tiny"].sum()),
        "n_non_raster": int(df["flag_non_raster"].sum()),
        "n_missing_creator": int(df["flag_missing_creator"].sum()),
        "n_missing_description": int(df["flag_missing_description"].sum()),
        "n_nonstandard_license": int(df["flag_nonstandard_license"].sum()),
        "n_archival_or_diagram_flag": int(df["flag_archival_or_diagram"].sum()),
        "n_manual_review_rows": int(len(review)),
        "note": (
            "technical_pass is not Study 0 validation. Religious provenance, perceived "
            "religiousness, polysemy, focality and stimulus role require independent review."
        ),
    }
    (OUT / "commons_catalog_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
