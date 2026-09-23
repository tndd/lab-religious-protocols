"""Harvest a balanced open-media candidate pool from Wikimedia Commons.

Commons categories are acquisition routes, never theological or empirical ground truth.
Each bucket carries only a *role hint* used to construct visually comparable candidate
sets before independent Study 0 validation.
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "lab-religious-protocols/0.3 "
    "(research prototype; https://github.com/tndd/lab-religious-protocols)"
)
PER_BUCKET = 18
MAX_RETRIES = 5

BUCKETS = [
    # Religious / culturally mixed acquisition routes
    {"family": "shinto", "role": "threshold", "category": "Category:Torii in Japan"},
    {"family": "shinto", "role": "boundary_marker", "category": "Category:Shimenawa"},
    {"family": "shinto", "role": "household_display", "category": "Category:Kamidana"},
    {"family": "shinto", "role": "votive_object", "category": "Category:Ema (Shinto)"},
    {"family": "shinto", "role": "embedded_streetscape", "category": "Category:Wayside Shrines in Kyoto"},
    {"family": "mixed", "role": "portable_object", "category": "Category:Omamori"},
    {"family": "mixed", "role": "textual_object", "category": "Category:Omikuji"},
    {"family": "mixed", "role": "purification_installation", "category": "Category:Chōzuya"},
    {"family": "buddhist", "role": "small_wayside_structure", "category": "Category:Jizō-dō"},
    {"family": "buddhist", "role": "household_display", "category": "Category:Butsudan"},
    {"family": "buddhist", "role": "votive_object", "category": "Category:Ema (Buddhism)"},
    {"family": "buddhist", "role": "architecture_context", "category": "Category:Buddhist temples in Japan"},
    {"family": "christian", "role": "architecture_context", "category": "Category:Churches in Japan"},

    # Matched secular acquisition routes. These are not assumed to be "neutral";
    # they are candidate controls for visual/scene roles that Study 0 must validate.
    {"family": "secular_control", "role": "threshold", "category": "Category:Gates in Japan"},
    {"family": "secular_control", "role": "street_object", "category": "Category:Bus stops in Japan"},
    {"family": "secular_control", "role": "street_context", "category": "Category:Streets in Kyoto"},
    {"family": "secular_control", "role": "household_display", "category": "Category:Tokonoma"},
    {"family": "secular_control", "role": "street_object", "category": "Category:Vending machines in Kyoto"},
]


def api(params: dict[str, str | int]) -> dict:
    q = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    url = f"{API}?{urlencode(q)}"
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urlopen(req, timeout=60) as r:
                data = json.load(r)
            if "error" in data:
                raise RuntimeError(f"MediaWiki API error: {data['error']}")
            return data
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"Commons request failed after retries: {url}: {last_error}")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def ext(meta: dict, name: str) -> str:
    obj = meta.get(name, {})
    return strip_html(obj.get("value", "")) if isinstance(obj, dict) else ""


def direct_files(category: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    cont: dict[str, str] = {}
    while len(rows) < limit:
        data = api(
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": min(50, limit - len(rows)),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 960,
                **cont,
            }
        )
        rows.extend(data.get("query", {}).get("pages", []))
        if "continue" not in data:
            break
        cont = data["continue"]
    return rows[:limit]


def subcategories(category: str, limit: int = 40) -> list[str]:
    data = api(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "subcat",
            "cmlimit": limit,
        }
    )
    return [x["title"] for x in data.get("query", {}).get("categorymembers", [])]


def files_with_one_level(category: str, limit: int) -> list[dict]:
    found = direct_files(category, limit)
    if len(found) >= limit:
        return found[:limit]
    for subcat in subcategories(category):
        if len(found) >= limit:
            break
        found.extend(direct_files(subcat, limit - len(found)))
        time.sleep(0.1)
    return found[:limit]


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat()

    records: list[dict] = []
    failures: list[dict] = []
    seen: set[str] = set()

    for bucket in BUCKETS:
        family = bucket["family"]
        role = bucket["role"]
        category = bucket["category"]
        try:
            pages = files_with_one_level(category, PER_BUCKET)
        except Exception as exc:
            failure = {
                "source_family": family,
                "acquisition_role_hint": role,
                "source_category": category,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print("FAILED", json.dumps(failure, ensure_ascii=False))
            continue

        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue
            title = page.get("title", "")
            if title in seen:
                continue
            seen.add(title)
            meta = info.get("extmetadata", {})
            records.append(
                {
                    "cue_id": f"commons-{len(records)+1:04d}",
                    "source_family": family,
                    "acquisition_role_hint": role,
                    "source_category": category,
                    "commons_pageid": page.get("pageid", ""),
                    "commons_title": title,
                    "source_page_url": info.get("descriptionurl", ""),
                    "media_url": info.get("url", ""),
                    "thumbnail_url": info.get("thumburl", ""),
                    "media_type": mime,
                    "width": info.get("width", ""),
                    "height": info.get("height", ""),
                    "creator": ext(meta, "Artist"),
                    "license_short": ext(meta, "LicenseShortName"),
                    "license_url": ext(meta, "LicenseUrl"),
                    "usage_terms": ext(meta, "UsageTerms"),
                    "attribution": ext(meta, "Attribution"),
                    "credit": ext(meta, "Credit"),
                    "description": ext(meta, "ImageDescription"),
                    "date_time_original": ext(meta, "DateTimeOriginal"),
                    "retrieved_at": retrieved,
                    "acquisition_only_not_ground_truth": True,
                }
            )
        print(f"{category}: {len(pages)} candidate pages")
        time.sleep(0.2)

    csv_path = out / "commons_candidate_catalog.csv"
    write_csv(csv_path, records)
    (out / "commons_harvest_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts_by_family: dict[str, int] = {}
    counts_by_role: dict[str, int] = {}
    for r in records:
        counts_by_family[r["source_family"]] = counts_by_family.get(r["source_family"], 0) + 1
        counts_by_role[r["acquisition_role_hint"]] = counts_by_role.get(r["acquisition_role_hint"], 0) + 1

    summary = {
        "retrieved_at": retrieved,
        "n_unique_candidates": len(records),
        "n_failed_buckets": len(failures),
        "counts_by_family": counts_by_family,
        "counts_by_role_hint": counts_by_role,
        "buckets": BUCKETS,
        "warning": (
            "Family/category/role fields are acquisition metadata only. "
            "Independent Study 0 validation must establish provenance, perceived "
            "religiousness, salience, and interpretive distributions."
        ),
    }
    (out / "commons_candidate_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {csv_path.resolve()}")

    if not records:
        raise RuntimeError("Commons harvest returned zero candidates; inspect failure JSON")


if __name__ == "__main__":
    main()
