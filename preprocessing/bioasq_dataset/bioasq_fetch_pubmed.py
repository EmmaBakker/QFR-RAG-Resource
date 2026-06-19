#!/usr/bin/env python3
"""
Fetch PubMed title + abstract for a list of PMIDs using NCBI EFetch.

Inputs:
  data/bioasq/processed/pmids.txt

Outputs:
  data/bioasq/processed/docs.jsonl  (doc_id=pmid:<id>, text=title + abstract)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

import requests


PMIDS_DEFAULT = Path("data/bioasq/processed/pmids.txt")
OUT_DOCS_DEFAULT = Path("data/bioasq/processed/docs.jsonl")


def chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def extract_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = extract_text(article.find(".//MedlineCitation/PMID"))
        title = extract_text(article.find(".//Article/ArticleTitle"))

        abstract_elems = article.findall(".//Article/Abstract/AbstractText")
        abstract_parts = [extract_text(a) for a in abstract_elems if extract_text(a)]
        abstract = "\n".join(abstract_parts).strip()

        journal = extract_text(article.find(".//Article/Journal/Title"))
        year = extract_text(article.find(".//Article/Journal/JournalIssue/PubDate/Year"))

        # Some PubDates have MedlineDate like "2016 Jan-Feb"
        if not year:
            medline_date = extract_text(article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
            year = medline_date.split(" ")[0] if medline_date else ""

        text = (title + "\n\n" + abstract).strip() if abstract else title.strip()

        out.append({
            "doc_id": f"pmid:{pmid}",
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "text": text,
            "journal": journal,
            "year": year,
            "source": "pubmed",
        })

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", type=Path, default=PMIDS_DEFAULT)
    ap.add_argument("--out_docs", type=Path, default=OUT_DOCS_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.4)  # be conservative
    args = ap.parse_args()

    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    email = os.environ.get("NCBI_EMAIL", "").strip()  # optional but recommended by NCBI

    pmids = [p.strip() for p in args.pmids.read_text(encoding="utf-8").splitlines() if p.strip()]
    args.out_docs.parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    total = 0
    with args.out_docs.open("w", encoding="utf-8") as f_out:
        for batch in chunks(pmids, args.batch_size):
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            if api_key:
                params["api_key"] = api_key
            if email:
                params["email"] = email

            r = requests.get(base_url, params=params, timeout=60)
            r.raise_for_status()

            records = parse_pubmed_xml(r.text)
            for rec in records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(records)

            time.sleep(args.sleep)

    print(f"Wrote {args.out_docs} with {total} docs.")


if __name__ == "__main__":
    main()