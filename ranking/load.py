"""Load the FundedLabs clean tables (NIH + NSF grants, PIs, grant-linked papers)."""
import pandas as pd

RAW = "data/raw"
NA = ["\\N"]


def read(name, cols, **kw):
    return pd.read_csv(f"{RAW}/{name}.tsv.gz", sep="\t", header=None, names=cols,
                       quoting=3, na_values=NA, keep_default_na=False, low_memory=False, **kw)


def awards():
    a = read("award", ["award_id", "agency_id", "inst_id", "src", "title", "abstract", "dept",
                       "program", "start", "end", "amount"], dtype={"src": str, "dept": str})
    a["start"] = pd.to_datetime(a.start, errors="coerce")
    a["end"] = pd.to_datetime(a.end, errors="coerce")
    a["agency"] = a.agency_id.map({1: "NIH", 2: "NSF"})
    return a


def investigators():
    return read("investigator", ["inv_id", "full_name", "name_norm", "nih_profile_id", "nsf_id", "inst_id"],
                dtype={"nsf_id": str})


def award_investigator():
    return read("award_investigator", ["award_id", "inv_id", "role"])


def institutions():
    return read("institution", ["inst_id", "name", "city", "state", "country"])


def publications():
    return read("publication", ["pub_id", "pmid", "doi", "title", "journal", "pub_year"])


def award_publication():
    return read("award_publication", ["award_id", "pub_id"])
