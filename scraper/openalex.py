"""
OpenAlex API Veri Cekici (Uye 1).
41 universite icin OpenAlex uzerinden Computer Science / Computer Engineering
akademisyenlerini ve standart CS kavramlarini (x_concepts + subfield) ceker.
Kisisel veri (e-posta, telefon, fotograf) kesinlikle toplanmaz.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from src.loader import load_universities

OPENALEX_BASE = "https://api.openalex.org"
CS_FIELD_ID = "17"
HEADERS = {
    "User-Agent": "BTU-AcademicExpertiseNetwork/1.0 (mailto:student@btu.edu.tr)",
    "Accept": "application/json",
}

GENERIC_EXCLUDE = {
    "computer science",
    "artificial intelligence",
    "engineering",
    "mathematics",
    "physics",
    "medicine",
    "biology",
    "psychology",
    "business",
    "economics",
}


def resolve_institution_id(
    session: requests.Session,
    canonical_name: str,
    aliases: list[str],
) -> str | None:
    """Universite adi veya alias listesi ile OpenAlex Institution ID bulur."""
    candidates = [canonical_name] + aliases
    for query in candidates:
        try:
            resp = session.get(
                f"{OPENALEX_BASE}/institutions",
                params={"search": query, "per-page": 3},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                for item in results:
                    display = item.get("display_name", "").lower()
                    if display == query.lower():
                        return item.get("id", "").split("/")[-1]
                return results[0].get("id", "").split("/")[-1]
        except requests.RequestException:
            continue
    return None


def fetch_authors_for_institution(
    session: requests.Session,
    university_name: str,
    institution_id: str,
    max_authors: int = 35,
) -> list[dict[str, Any]]:
    """Belirli bir OpenAlex kurum ID'si icin CS alanindaki akademisyenleri ceker."""
    records: list[dict[str, Any]] = []
    try:
        params = {
            "filter": f"last_known_institutions.id:{institution_id},topics.field.id:{CS_FIELD_ID}",
            "sort": "works_count:desc",
            "per-page": min(max_authors, 100),
            "select": "id,display_name,topics,x_concepts",
        }
        resp = session.get(f"{OPENALEX_BASE}/authors", params=params, timeout=20)
        resp.raise_for_status()
        authors = resp.json().get("results", [])

        for author in authors[:max_authors]:
            name = (author.get("display_name") or "").strip()
            if not name:
                continue

            fields: list[str] = []

            # 1. Oncelik: Kisa ve kesisimi yuksek x_concepts (level 1 ve 2)
            for concept in author.get("x_concepts", []):
                level = concept.get("level", 0)
                c_name = (concept.get("display_name") or "").strip()
                if level in (1, 2) and c_name and c_name.lower() not in GENERIC_EXCLUDE:
                    if c_name not in fields:
                        fields.append(c_name)
                if len(fields) >= 5:
                    break

            # 2. Yedek: Topics icindeki subfield etiketleri (kisa uzmanlik alanlari)
            if len(fields) < 4:
                for topic in author.get("topics", []):
                    subfield = (topic.get("subfield", {}).get("display_name") or "").strip()
                    if subfield and subfield.lower() not in GENERIC_EXCLUDE and subfield not in fields:
                        fields.append(subfield)
                    if len(fields) >= 5:
                        break

            if not fields:
                continue

            records.append(
                {
                    "source": "openalex",
                    "name": name,
                    "title": "",
                    "university": university_name,
                    "department": "Computer Science / Engineering",
                    "fields": fields[:5],
                }
            )
    except requests.RequestException as exc:
        print(f"  [UYARI] {university_name} icin OpenAlex istegi basarisiz: {exc}")

    return records


def run_openalex_harvest(
    universities_path: str | Path = "data/universities.json",
    output_path: str | Path = "data/raw/openalex_raw.json",
    max_per_uni: int = 30,
    delay_sec: float = 0.2,
) -> list[dict[str, Any]]:
    """41 universitenin tamamini OpenAlex uzerinden tarar ve ham JSON kaydeder."""
    universities = load_universities(universities_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"OpenAlex taramasi basliyor ({len(universities)} universite)...")
    for uni in universities:
        uni_id = uni["id"]
        canonical = uni["canonical_name"]
        aliases = uni.get("aliases", [])
        oa_id = uni.get("openalex_id") or resolve_institution_id(session, canonical, aliases)

        if not oa_id:
            print(f"[{uni_id:02d}/41] {canonical} -> OpenAlex ID bulunamadi, atlandi.")
            continue

        authors = fetch_authors_for_institution(session, canonical, oa_id, max_authors=max_per_uni)
        all_records.extend(authors)
        print(f"[{uni_id:02d}/41] {canonical} ({oa_id}) -> {len(authors)} akademisyen cekildi.")
        time.sleep(delay_sec)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nTamamlandi! Toplam {len(all_records)} ham kayit '{out_file}' dosyasina yazildi.")
    return all_records


if __name__ == "__main__":
    run_openalex_harvest(max_per_uni=30)