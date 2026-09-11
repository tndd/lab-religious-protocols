"""Harvest an initial open-media candidate pool from Wikimedia Commons.

Acquisition metadata only: Commons category membership is not treated as
religious ground truth. The harvester is deliberately fault-tolerant so a
single HTTP/category failure does not erase the rest of a long batch.
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
    "lab-religious-protocols/0.2 "
    "(research prototype; https://github.com/tndd/lab-religious-protocols)"
)
PER_BUCKET = 18
MAX_RETRIES = 5

BUCKETS = [
    ("shinto", "Category:Torii in Japan"),
    ("shinto", "Category:Shimenawa"),
    ("shinto", "Category:Kamidana"),
    ("shinto", "Category:Ema (Shinto)"),
    ("mixed", "Category:Omamori"),
    ("mixed", "Category:Omikuji"),
    ("mixed", "Category:Chōzuya"),
    ("buddhist", "Category:Jizō-dō"),
    ("buddhist", "Category:Butsudan"),
    ("buddhist", "Category:Ema (Buddhism)"),
    ("buddhist", "Category:Buddhist temples in Japan"),
    ("christian", "Category:Churches in Japan"),
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
                "iiurlwidth": 768,
                **cont,
            }
        )
        rows.extend(data.get("query", {}).get("pages", []))
        if "continue" not in data:
            break
        cont = data["continue"]
    return rows[:limit]


def subcategories(category: str, limit: int = 30) -> list[str]:
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

    for family, category in BUCKETS:
        try:
            pages = files_with_one_level(category, PER_BUCKET)
        except Exception as exc:  # keep the batch alive and preserve diagnostics
            failure = {
                "source_family": family,
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

    summary = {
        "retrieved_at": retrieved,
        "n_unique_candidates": len(records),
        "n_failed_buckets": len(failures),
        "counts_by_family": {},
        "categories": [c for _, c in BUCKETS],
        "warning": (
            "Category membership is acquisition metadata only. Verify each "
            "file page/license and independently document provenance before use."
        ),
    }
    for r in records:
        fam = r["source_family"]
        summary["counts_by_family"][fam] = summary["counts_by_family"].get(fam, 0) + 1
    (out / "commons_candidate_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {csv_path.resolve()}")

    # A totally empty harvest is a real failure, but diagnostics are now preserved.
    if not records:
        raise RuntimeError("Commons harvest returned zero candidates; inspect failure JSON")


if __name__ == "__main__":
    main()
