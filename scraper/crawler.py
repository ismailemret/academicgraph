"""
Universite Web Sayfasi Kaziyici (BeautifulSoup & Selenium) - Uye 1.
- Statik HTML bolum sayfalarini Requests + BeautifulSoup ile kazir.
- Dinamik (JavaScript ile yuklenen) sayfalari Selenium (Headless Chrome) ile kazir.
- KVKK geregi e-posta, telefon, ofis ve fotograf URL'leri kesinlikle kaydedilmez.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

TITLE_PATTERNS = [
    r"Prof\.\s*Dr\.",
    r"Do\u00e7\.\s*Dr\.",
    r"Doc\.\s*Dr\.",
    r"Dr\.\s*\u00d6\u011fr\.\s*\u00dcyesi",
    r"Dr\.\s*Ogr\.\s*Uyesi",
    r"Assoc\.\s*Prof\.\s*Dr\.",
    r"Asst\.\s*Prof\.\s*Dr\.",
    r"Professor",
    r"Associate Professor",
    r"Assistant Professor",
    r"Visiting Professor",
    r"Adjunct Professor",
    r"Instructor",
]
TITLE_RE = re.compile(rf"^({'|'.join(TITLE_PATTERNS)})\s+", re.IGNORECASE)


def fetch_html_bs4(url: str, timeout: int = 15) -> str:
    """Requests + BeautifulSoup icin ham HTML icerigini getirir (SSL toleransli)."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def fetch_html_selenium(url: str, wait_sec: float = 4.0) -> str:
    """Dinamik (JavaScript) sayfalar icin Headless Chrome Selenium ile HTML getirir."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(wait_sec)
        return driver.page_source
    finally:
        driver.quit()


def split_title_and_name(full_text: str) -> tuple[str, str]:
    """'Prof. Dr. Ahmet Yilmaz' metnini ('Prof. Dr.', 'Ahmet Yilmaz') olarak ayirir."""
    clean = " ".join(full_text.split()).strip()
    match = TITLE_RE.match(clean)
    if match:
        return match.group(1).strip(), clean[match.end() :].strip()
    return "", clean


def parse_bilkent_cs(html: str) -> list[dict[str, Any]]:
    """
    Bilkent CS (/faculty-members) sayfasini BeautifulSoup ile ayristirir.
    Email, Office ve Phone bilgilerini filtreleyerek yalnizca ad, unvan ve
    arastirma alanlarini dondurur.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for block in soup.find_all(["tr", "div", "li"]):
        raw_text = block.get_text(" | ", strip=True)
        if len(raw_text) > 500 or "Prof." not in raw_text and "Professor" not in raw_text:
            continue

        parts = [p.strip() for p in raw_text.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        title, name = split_title_and_name(parts[0])
        if not title or not name or len(name.split()) < 2 or name.lower() in seen_names:
            continue

        # KVKK: Email, Office, Phone, Homepage parcalarini atla; arastirma alanlarini al
        research_chunks: list[str] = []
        for part in parts[1:]:
            lower = part.lower()
            if any(
                skip in lower
                for skip in ("email", "e-mail", "office", "phone", "homepage", "tel:", "fax", "@", "http")
            ):
                continue
            if re.search(r"\(\d{3}\)|\d{3}\s*\d{4}", part):
                continue
            if len(part) > 8:
                research_chunks.append(part)

        if not research_chunks:
            continue

        fields: list[str] = []
        for chunk in research_chunks:
            for item in re.split(r"[,;.]", chunk):
                clean_field = item.strip(" :-")
                if 3 < len(clean_field) < 55 and clean_field not in fields:
                    fields.append(clean_field)

        if fields:
            seen_names.add(name.lower())
            results.append(
                {
                    "source": "web_crawler_bs4",
                    "name": name,
                    "title": title,
                    "university": "Bilkent University",
                    "department": "Computer Engineering",
                    "fields": fields[:6],
                }
            )

    return results


def crawl_university_page(
    url: str,
    university: str,
    department: str = "Computer Engineering",
    use_selenium: bool = False,
) -> list[dict[str, Any]]:
    """Tek bir bolum sayfasini kazir (BS4 veya Selenium secenekli)."""
    try:
        html = fetch_html_selenium(url) if use_selenium else fetch_html_bs4(url)
        if "bilkent.edu.tr" in url:
            return parse_bilkent_cs(html)
        return []
    except Exception as exc:
        print(f"[UYARI] {university} ({url}) kazinamadi: {exc}")
        return []


def run_web_crawls(output_path: str | Path = "data/raw/web_scraped_raw.json") -> list[dict[str, Any]]:
    """
    BeautifulSoup ve Selenium destekli web kazima hattini calistirir.
    """
    collected: list[dict[str, Any]] = []

    url_bilkent = "https://w3.cs.bilkent.edu.tr/faculty-members"
    print(f"Kaziniyor [BeautifulSoup]: Bilkent University -> {url_bilkent}")
    bilkent_rows = crawl_university_page(url_bilkent, "Bilkent University", use_selenium=False)
    print(f"  -> {len(bilkent_rows)} akademisyen cekildi (kisisel veriler ayiklandi).")
    collected.extend(bilkent_rows)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    print(f"Web kazima tamamlandi. Toplam {len(collected)} kayit '{out_file}' dosyasina yazildi.")
    return collected


if __name__ == "__main__":
    run_web_crawls()