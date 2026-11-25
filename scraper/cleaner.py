"""
Veri Temizleme ve Normalizasyon Modulu (Uye 1).
- data/raw/ altindaki tum ham JSON ciktilarini birlestirir.
- Universite adlarini data/universities.json kanonik isimlerine esler.
- Calisma alanlarini data/field_aliases.json ile normalize eder.
- Kisisel verileri (e-posta, telefon, fotograf URL) ayiklar.
- Final data/processed/academicians.json ve academicians.csv dosyalarini uretir.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.loader import load_universities

ALLOWED_KEYS = {"id", "name", "title", "university", "department", "fields"}
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def normalize_ascii_text(text: str) -> str:
    """Turkce ve aksanli karakterleri ASCII karsiliklarina cevirir."""
    tr_map = str.maketrans(
        "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
        "cgiosuCGIOSU",
    )
    text = text.translate(tr_map)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split()).strip()


def load_field_aliases(path: str | Path = "data/field_aliases.json") -> dict[str, str]:
    """Harici alan esanlamlilar sozlugunu yukler."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return {k.strip().lower(): v.strip() for k, v in raw.items()}


def build_university_lookup(universities: list[dict[str, Any]]) -> dict[str, str]:
    """Kanonik isim ve tum alias'lardan canonical_name'e esleme tablosu kurar."""
    lookup: dict[str, str] = {}
    for uni in universities:
        canonical = uni["canonical_name"].strip()
        lookup[normalize_ascii_text(canonical).lower()] = canonical
        for alias in uni.get("aliases", []):
            lookup[normalize_ascii_text(alias).lower()] = canonical
    return lookup


def normalize_field(raw_field: str, aliases: dict[str, str]) -> str:
    """Calisma alani etiketini temizler ve kanonik hale getirir."""
    cleaned = normalize_ascii_text(raw_field)
    cleaned = EMAIL_RE.sub("", cleaned)
    cleaned = URL_RE.sub("", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9\s\-&/]", "", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return ""
    key = cleaned.lower()
    if key in aliases:
        return aliases[key]
    return " ".join(word.capitalize() if not word.isupper() else word for word in cleaned.split())


def clean_and_merge_raw_data(
    raw_dir: str | Path = "data/raw",
    universities_path: str | Path = "data/universities.json",
    aliases_path: str | Path = "data/field_aliases.json",
    output_json: str | Path = "data/processed/academicians.json",
    output_csv: str | Path = "data/processed/academicians.csv",
) -> list[dict[str, Any]]:
    """Ham dosyalari temizleyip birlestirir ve final JSON + CSV ciktilarini yazar."""
    universities = load_universities(universities_path)
    uni_lookup = build_university_lookup(universities)
    field_aliases = load_field_aliases(aliases_path)

    raw_folder = Path(raw_dir)
    raw_files = sorted(raw_folder.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"'{raw_folder}' klasorunde islenecek ham .json dosyasi bulunamadi.")

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    raw_total = 0
    dropped_unmatched_uni = 0
    dropped_invalid = 0

    for file_path in raw_files:
        with file_path.open("r", encoding="utf-8-sig") as f:
            items = json.load(f)
        if not isinstance(items, list):
            continue

        for item in items:
            raw_total += 1
            if not isinstance(item, dict):
                dropped_invalid += 1
                continue

            raw_name = normalize_ascii_text(str(item.get("name", "")))
            raw_name = EMAIL_RE.sub("", raw_name).strip()
            if not raw_name or len(raw_name) < 3:
                dropped_invalid += 1
                continue

            raw_uni = normalize_ascii_text(str(item.get("university", ""))).lower()
            canonical_uni = uni_lookup.get(raw_uni)
            if not canonical_uni:
                dropped_unmatched_uni += 1
                continue

            raw_fields = item.get("fields", [])
            if isinstance(raw_fields, str):
                raw_fields = [x.strip() for x in raw_fields.split(";")]
            if not isinstance(raw_fields, list):
                dropped_invalid += 1
                continue

            norm_fields: list[str] = []
            for rf in raw_fields:
                nf = normalize_field(str(rf), field_aliases)
                if nf and nf not in norm_fields:
                    norm_fields.append(nf)

            if not norm_fields:
                dropped_invalid += 1
                continue

            title = normalize_ascii_text(str(item.get("title", "")))
            dept = normalize_ascii_text(str(item.get("department", ""))) or "Computer Engineering"

            dedup_key = (raw_name.lower(), canonical_uni)
            if dedup_key in merged:
                existing = merged[dedup_key]
                for nf in norm_fields:
                    if nf not in existing["fields"]:
                        existing["fields"].append(nf)
                if not existing["title"] and title:
                    existing["title"] = title
            else:
                merged[dedup_key] = {
                    "name": raw_name,
                    "title": title,
                    "university": canonical_uni,
                    "department": dept,
                    "fields": norm_fields,
                }

    final_records: list[dict[str, Any]] = []
    for idx, rec in enumerate(merged.values(), start=1):
        clean_rec = {
            "id": f"acad_{idx:04d}",
            "name": rec["name"],
            "title": rec["title"],
            "university": rec["university"],
            "department": rec["department"],
            "fields": rec["fields"],
        }
        final_records.append({k: v for k, v in clean_rec.items() if k in ALLOWED_KEYS})

    out_json_path = Path(output_json)
    out_csv_path = Path(output_csv)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    with out_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "title", "university", "department", "fields"],
        )
        writer.writeheader()
        for rec in final_records:
            row = dict(rec)
            row["fields"] = ";".join(rec["fields"])
            writer.writerow(row)

    print("\n=== Veri Temizleme & Birlestirme Raporu ===")
    print(f"Islenen Ham Kayit        : {raw_total}")
    print(f"Eslesmeyen Universite    : {dropped_unmatched_uni}")
    print(f"Eksik/Gecersiz Kayit     : {dropped_invalid}")
    print(f"Tekil Final Akademisyen  : {len(final_records)}")
    print(f"JSON Ciktisi             : {out_json_path}")
    print(f"CSV Ciktisi              : {out_csv_path}")
    print("===========================================\n")
    return final_records


if __name__ == "__main__":
    clean_and_merge_raw_data()