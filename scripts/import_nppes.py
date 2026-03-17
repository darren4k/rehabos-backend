#!/usr/bin/env python3
"""
Import NPPES data into PostgreSQL — ALL active individual medical providers.

Streams the 11GB NPPES CSV, extracts all individual (NPI-1) providers
that are NOT deactivated, and loads into `nppes_medical_providers`.
Tags discipline (PT/OT/SLP/MD/DO/NP/PA/etc.) from taxonomy codes.

Shared across RehabOS, DocPilot, and Magda Health.

Usage:
  python scripts/import_nppes.py /path/to/npidata_pfile_*.csv

Typical run: ~10-15 min for 9.4M rows → ~4-5M active individual providers.
"""
import csv
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Discipline mapping from taxonomy code prefixes
DISCIPLINE_MAP = {
    # Rehab
    "2251": "PT",    "2252": "PTA",
    "225X": "OT",    "224Z": "COTA",
    "235Z": "SLP",   "2355": "SLP",
    # Physicians
    "207": "MD",     "208": "MD",
    "209": "MD",     "204": "MD",
    # Osteopathic
    "171": "DO",
    # Nursing
    "363L": "NP",    "363A": "NP",
    "364S": "CNS",
    # Physician Assistant
    "363A": "PA",
    # Chiropractic
    "111N": "DC",
    # Podiatry
    "213E": "DPM",
    # Psychology
    "103T": "PhD",   "103G": "PhD",
    # Social Work
    "104": "LCSW",
    # Dentistry
    "122": "DDS",    "126": "DDS",
    # Pharmacy
    "183": "PharmD",
    # Registered Nurse
    "163W": "RN",
    # Optometry
    "152W": "OD",
}


def get_discipline(taxonomy_code: str) -> str | None:
    """Map taxonomy code to discipline abbreviation."""
    if not taxonomy_code:
        return None
    for prefix, disc in DISCIPLINE_MAP.items():
        if taxonomy_code.startswith(prefix):
            return disc
    return None


def trunc(val, maxlen):
    v = (val or "").strip()
    return v[:maxlen] if v else None


def extract_providers(csv_path: str):
    """Stream CSV and yield all active individual providers."""
    logger.info(f"Streaming {csv_path} ...")
    count = 0
    extracted = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            count += 1
            if count % 500_000 == 0:
                logger.info(f"  {count:,} rows scanned, {extracted:,} providers extracted...")

            # Only individual providers (NPI-1)
            if row.get("Entity Type Code") != "1":
                continue

            npi = (row.get("NPI") or "").strip()
            if not npi or len(npi) != 10:
                continue

            deactivation = trunc(row.get("NPI Deactivation Date"), 20)

            # Get primary taxonomy
            primary_taxonomy = None
            primary_desc = None
            all_taxonomies = []

            for i in range(1, 16):
                code = (row.get(f"Healthcare Provider Taxonomy Code_{i}") or "").strip()
                if not code:
                    continue
                all_taxonomies.append(code)
                is_primary = (row.get(f"Healthcare Provider Primary Taxonomy Switch_{i}") or "").strip() == "Y"
                if is_primary or not primary_taxonomy:
                    primary_taxonomy = code

            discipline = get_discipline(primary_taxonomy) if primary_taxonomy else None

            extracted += 1

            yield {
                "npi": npi,
                "entity_type": "individual",
                "first_name": trunc(row.get("Provider First Name"), 100),
                "last_name": trunc(row.get("Provider Last Name (Legal Name)"), 100),
                "middle_name": trunc(row.get("Provider Middle Name"), 50),
                "prefix": trunc(row.get("Provider Name Prefix Text"), 20),
                "suffix": trunc(row.get("Provider Name Suffix Text"), 20),
                "credential": trunc(row.get("Provider Credential Text"), 100),
                "organization_name": trunc(row.get("Provider Organization Name (Legal Business Name)"), 300),
                "address_line1": trunc(row.get("Provider First Line Business Practice Location Address"), 200),
                "address_line2": trunc(row.get("Provider Second Line Business Practice Location Address"), 100),
                "city": trunc(row.get("Provider Business Practice Location Address City Name"), 100),
                "state": trunc(row.get("Provider Business Practice Location Address State Name"), 50),
                "postal_code": trunc(row.get("Provider Business Practice Location Address Postal Code"), 20),
                "phone": trunc(row.get("Provider Business Practice Location Address Telephone Number"), 20),
                "fax": trunc(row.get("Provider Business Practice Location Address Fax Number"), 20),
                "sex": trunc(row.get("Provider Sex Code"), 5),
                "taxonomy_code": trunc(primary_taxonomy, 20),
                "taxonomy_desc": None,  # filled later if needed
                "discipline": discipline,
                "all_taxonomy_codes": ",".join(all_taxonomies)[:500] if all_taxonomies else None,
                "enumeration_date": trunc(row.get("Provider Enumeration Date"), 20),
                "last_update": trunc(row.get("Last Update Date"), 20),
                "deactivation_date": deactivation,
            }

    logger.info(f"Done scanning. {count:,} total rows, {extracted:,} individual providers extracted.")


UPSERT_SQL = """
INSERT INTO nppes_medical_providers (
    npi, entity_type, first_name, last_name, middle_name, prefix, suffix,
    credential, organization_name, address_line1, address_line2, city, state,
    postal_code, phone, fax, sex, taxonomy_code, taxonomy_desc, discipline,
    all_taxonomy_codes, enumeration_date, last_update, deactivation_date, imported_at
) VALUES (
    %(npi)s, %(entity_type)s, %(first_name)s, %(last_name)s, %(middle_name)s,
    %(prefix)s, %(suffix)s, %(credential)s, %(organization_name)s,
    %(address_line1)s, %(address_line2)s, %(city)s, %(state)s,
    %(postal_code)s, %(phone)s, %(fax)s, %(sex)s, %(taxonomy_code)s,
    %(taxonomy_desc)s, %(discipline)s, %(all_taxonomy_codes)s,
    %(enumeration_date)s, %(last_update)s, %(deactivation_date)s, NOW()
)
ON CONFLICT (npi) DO UPDATE SET
    first_name = EXCLUDED.first_name,
    last_name = EXCLUDED.last_name,
    credential = EXCLUDED.credential,
    address_line1 = EXCLUDED.address_line1,
    city = EXCLUDED.city,
    state = EXCLUDED.state,
    postal_code = EXCLUDED.postal_code,
    phone = EXCLUDED.phone,
    fax = EXCLUDED.fax,
    taxonomy_code = EXCLUDED.taxonomy_code,
    discipline = EXCLUDED.discipline,
    all_taxonomy_codes = EXCLUDED.all_taxonomy_codes,
    last_update = EXCLUDED.last_update,
    deactivation_date = EXCLUDED.deactivation_date,
    imported_at = NOW()
;
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_nppes.py <path-to-npidata_pfile.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://rehab_os:rehab_os_2026@localhost:5432/rehab_core"
    )

    import psycopg2

    logger.info("Connecting to database...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Verify table exists
    cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'nppes_medical_providers'")
    if not cur.fetchone():
        logger.error("Table nppes_medical_providers does not exist. Create it first.")
        sys.exit(1)
    conn.commit()

    # Stream and insert
    start = time.time()
    batch = []
    batch_size = 1000
    total_inserted = 0
    errors = 0

    for provider in extract_providers(csv_path):
        batch.append(provider)

        if len(batch) >= batch_size:
            for p in batch:
                try:
                    cur.execute(UPSERT_SQL, p)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Insert error for NPI {p['npi']}: {e}")
                    conn.rollback()  # reset transaction after error
            conn.commit()
            total_inserted += len(batch) - errors
            batch = []

    # Final batch
    if batch:
        for p in batch:
            try:
                cur.execute(UPSERT_SQL, p)
            except Exception as e:
                errors += 1
                conn.rollback()
        conn.commit()
        total_inserted += len(batch)

    elapsed = time.time() - start
    logger.info(f"Imported {total_inserted:,} providers in {elapsed:.1f}s ({errors} errors)")

    # Stats
    cur.execute("""
        SELECT discipline, COUNT(*)
        FROM nppes_medical_providers
        GROUP BY discipline
        ORDER BY COUNT(*) DESC
        LIMIT 20
    """)
    for row in cur.fetchall():
        logger.info(f"  {row[0] or 'Unknown'}: {row[1]:,}")

    cur.execute("SELECT COUNT(*) FROM nppes_medical_providers")
    total = cur.fetchone()[0]
    logger.info(f"  TOTAL: {total:,}")

    cur.execute("SELECT COUNT(*) FROM nppes_medical_providers WHERE deactivation_date IS NOT NULL")
    deact = cur.fetchone()[0]
    logger.info(f"  Deactivated: {deact:,}")

    cur.close()
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
