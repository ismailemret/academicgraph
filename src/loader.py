"""
Veri yukleme ve dogrulama katmani (Uye 1).
JSON ve CSV dosyalarini okur, eksik/hatali kayitlari raporlar ve
gecerli akademisyen kayitlarini dondurur.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadReport:
    """Dosya yukleme sirasinda tespit edilen istatistik ve hata raporu."""

    source_path: str
    total_rows: int = 0
    valid_count: int = 0
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, row_index: int, reason: str) -> None:
        self.skipped_count += 1
        self.errors.append(f"Satir {row_index}: {reason}")

    def print_summary(self) -> None:
        print(f"\n--- Veri Yukleme Raporu ({self.source_path}) ---")
        print(f"Toplam Kayit : {self.total_rows}")
        print(f"Gecerli Kayit: {self.valid_count}")
        print(f"Atlanan Kayit: {self.skipped_count}")
        if self.errors:
            print("Tespit Edilen Hatalar:")
            for err in self.errors[:20]:
                print(f"  - {err}")
            if len(self.errors) > 20:
                print(f"  ... ve {len(self.errors) - 20} hata daha.")
        print("--------------------------------------------------\n")


def load_universities(path: str | Path = "data/universities.json") -> list[dict[str, Any]]:
    """41 universitenin kanonik isim ve alias tanimlarini yukler."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Universite dosyasi bulunamadi: {file_path}")

    with file_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "universities" in payload:
        return payload["universities"]
    if isinstance(payload, list):
        return payload
    raise ValueError("universities.json formati gecersiz.")


def _validate_record(raw: dict[str, Any], index: int, report: LoadReport) -> dict[str, Any] | None:
    """Tek bir akademisyen kaydini dogrular ve standart sozluk formatina getirir."""
    if not isinstance(raw, dict):
        report.add_error(index, "Kayit sozluk (dict) formatinda degil.")
        return None

    name = str(raw.get("name", "")).strip()
    university = str(raw.get("university", "")).strip()
    title = str(raw.get("title", "")).strip()
    department = str(raw.get("department", "Computer Engineering")).strip() or "Computer Engineering"
    acad_id = str(raw.get("id", f"acad_{index:04d}")).strip() or f"acad_{index:04d}"

    raw_fields = raw.get("fields", [])
    if isinstance(raw_fields, str):
        fields = [item.strip() for item in raw_fields.split(";") if item.strip()]
    elif isinstance(raw_fields, list):
        fields = [str(item).strip() for item in raw_fields if str(item).strip()]
    else:
        fields = []

    # Tekrarsiz alan listesi (sirayi koruyarak)
    unique_fields = list(dict.fromkeys(fields))

    if not name:
        report.add_error(index, "Eksik akademisyen adi ('name').")
        return None
    if not university:
        report.add_error(index, f"'{name}' icin eksik universite bilgisi ('university').")
        return None
    if not unique_fields:
        report.add_error(index, f"'{name}' ({university}) icin calisma alani ('fields') bos.")
        return None

    return {
        "id": acad_id,
        "name": name,
        "title": title,
        "university": university,
        "department": department,
        "fields": unique_fields,
    }


def _to_model_objects(records: list[dict[str, Any]]) -> list[Any]:
    """src.models.Academician tanimliysa nesneye donusturur, yoksa dict listesi dondurur."""
    try:
        from src.models import Academician  # type: ignore

        if hasattr(Academician, "from_dict"):
            return [Academician.from_dict(r) for r in records]
        return [
            Academician(
                id=r["id"],
                name=r["name"],
                title=r["title"],
                university=r["university"],
                department=r["department"],
                fields=r["fields"],
            )
            for r in records
        ]
    except (ImportError, TypeError):
        return records


def load_academicians(
    path: str | Path = "data/processed/academicians.json",
    as_models: bool = True,
    verbose: bool = True,
) -> tuple[list[Any], LoadReport]:
    """
    JSON veya CSV dosyasindan akademisyen kayitlarini yukler.

    Args:
        path: Okunacak .json veya .csv dosya yolu.
        as_models: True ise src.models.Academician nesnelerine donusturur.
        verbose: True ise yukleme raporunu konsola yazdirir.

    Returns:
        (gecerli_kayitlar, LoadReport) ikilisi.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {file_path}")

    report = LoadReport(source_path=str(file_path))
    suffix = file_path.suffix.lower()
    raw_rows: list[Any] = []

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON dosyasi bir liste ([...]) icermelidir.")
        raw_rows = data
    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)
    else:
        raise ValueError(f"Desteklenmeyen dosya uzantisi: {suffix} (yalnizca .json veya .csv)")

    report.total_rows = len(raw_rows)
    valid_records: list[dict[str, Any]] = []

    for idx, row in enumerate(raw_rows, start=1):
        cleaned = _validate_record(row, idx, report)
        if cleaned is not None:
            valid_records.append(cleaned)

    report.valid_count = len(valid_records)
    if verbose:
        report.print_summary()

    output = _to_model_objects(valid_records) if as_models else valid_records
    return output, report


if __name__ == "__main__":
    records, rep = load_academicians("data/processed/academicians.json", as_models=False)
    print(f"Yuklenen ornek kayit: {records[0] if records else 'Yok'}")
