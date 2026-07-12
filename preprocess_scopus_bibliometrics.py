import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import requests
from rapidfuzz import fuzz

REQUIRED_COLUMNS = ["Author full names", "Author(s) ID", "Title", "Year", "Source title", "Volume", "Issue", "Art. No.", "Cited by", "DOI", "Affiliations", "Authors with affiliations", "Abstract", "Author Keywords", "Index Keywords", "Funding Texts", "Editors", "Publisher", "Language of Original Document", "Document Type", "Open Access"]
TEXT_COLUMNS = ["Author full names", "Affiliations", "Source title", "Publisher", "Title", "Abstract", "Author Keywords", "Index Keywords", "Authors with affiliations", "Funding Texts", "Editors", "Language of Original Document", "Document Type", "Open Access"]
FILTER_TERMS = ["inborn error of immunity", "inborn errors of immunity", "primary immunodeficiency", "primary immune deficiency", "primary immunodeficiencies", "primary immune deficiencies"]
SEARCH_COLUMNS = ["Title", "Abstract", "Author Keywords", "Index Keywords"]
def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def normalize_space(text):
    if not isinstance(text, str):
        return ""
    text = fix_encoding(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lookup_text(text):
    text = unicodedata.normalize("NFKD", normalize_space(text)).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def clean_author_name(name):
    name = normalize_space(name)
    name = re.sub(r"\s*\(\s*\d{5,}\s*\)\s*", "", name)
    name = re.sub(r"\s*\[[^\]]*\]\s*", "", name)
    name = re.sub(r"\s+", " ", name).strip(" ;,|")
    return name


def parse_scopus_authors(value):
    value = normalize_space(value)
    if value == "":
        return []
    authors = []
    for part in value.split(";"):
        name = clean_author_name(part)
        if name:
            authors.append(name)
    return authors


def clean_author_id(value):
    value = normalize_space(value)
    match = re.search(r"\d{5,}", value)
    return match.group(0) if match else ""


def parse_scopus_author_ids(value):
    value = normalize_space(value)
    if value == "":
        return []
    return [clean_author_id(part) for part in value.split(";")]


def align_author_ids(authors, author_ids):
    values = list(author_ids[:len(authors)])
    if len(values) < len(authors):
        values.extend([""] * (len(authors) - len(values)))
    return values



def author_identity_key(name, author_id=""):
    ordered = author_order_key(name)
    return f"name:{ordered}" if ordered else ""


def clean_affiliation_text(value):
    value = normalize_space(value).strip()
    quote_pairs = [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")]
    for opening, closing in quote_pairs:
        if len(value) >= 2 and value.startswith(opening) and value.endswith(closing):
            value = value[len(opening):-len(closing)].strip()
            break
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r",\s*,+", ", ", value)
    return value.strip(" ,;|")


ALLOWED_COMPONENT_TYPES = {"institution", "city", "country", "exclude"}
ACCEPT_DECISIONS = {"accept", "accepted", "yes", "y", "true", "1"}
REJECT_DECISIONS = {"reject", "rejected", "no", "n", "false", "0"}


def clean_affiliation_component(value):
    return clean_affiliation_text(value)


def component_type_value(value):
    value = normalize_lookup_text(value)
    return value if value in ALLOWED_COMPONENT_TYPES else ""


def decision_value(value):
    key = normalize_lookup_text(value)
    if key in ACCEPT_DECISIONS:
        return "accept"
    if key in REJECT_DECISIONS:
        return "reject"
    return "unresolved"


def split_affiliation_components(value):
    affiliation = clean_affiliation_text(value)
    return [clean_affiliation_component(part) for part in affiliation.split(",") if clean_affiliation_component(part)]


def parse_affiliation_entries(value):
    value = normalize_space(value)
    if value == "":
        return []
    entries = []
    for affiliation in value.split(";"):
        raw_affiliation = clean_affiliation_text(affiliation)
        if not raw_affiliation:
            continue
        entries.append({
            "raw_affiliation": raw_affiliation,
            "components": split_affiliation_components(raw_affiliation)
        })
    return entries


def normalize_for_similarity(name):
    return normalize_lookup_text(name)


def author_order_key(name):
    normalized = normalize_for_similarity(name)
    return " ".join(sorted(normalized.split()))


def title_key(value):
    return normalize_lookup_text(value)


def title_block_key(value):
    tokens = title_key(value).split()
    stopwords = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
    selected = [token for token in tokens if token not in stopwords][:2]
    if not selected:
        selected = tokens[:2]
    return " ".join(selected)[:32]


def source_key(source, publisher=""):
    source_value = normalize_lookup_text(source)
    if source_value:
        return source_value
    return normalize_lookup_text(publisher)


def year_key(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return str(int(numeric))


def normalize_doi(value):
    value = normalize_space(value).strip(" \t\r\n\"'“”‘’")
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".,;")
    return value.casefold()


def text_contains_filter_terms(row):
    text = re.sub(r"[^\w]+", " ", " ".join(str(row.get(col, "")) for col in SEARCH_COLUMNS), flags=re.UNICODE)
    text = normalize_lookup_text(text)
    return any(normalize_lookup_text(term) in text for term in FILTER_TERMS)


def ensure_columns(df):
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def empty_mapping_data():
    return {
        "exact": {},
        "normalized": {},
        "ordered": {},
        "exact_ids": {},
        "normalized_ids": {},
        "ordered_ids": {},
        "id_rules": {},
        "canonical_exact": set(),
        "canonical_normalized": {},
        "canonical_ordered": {},
        "canonical_ids": set(),
        "canonical_id_names": {},
        "reference_names": set(),
        "reference_authors": set(),
        "rows": [],
        "loaded": 0
    }


def resolve_mapping_chain(name, replacement):
    seen = []
    current = name
    while current in replacement:
        if current in seen:
            cycle = " -> ".join(seen + [current])
            raise ValueError(f"Circular author mapping detected: {cycle}")
        seen.append(current)
        next_name = replacement[current]
        if not next_name or next_name == current:
            break
        current = next_name
    return current


def resolve_mapping_target_id(name, replacement, target_ids):
    seen = []
    current = name
    resolved_id = ""
    while current in replacement:
        if current in seen:
            cycle = " -> ".join(seen + [current])
            raise ValueError(f"Circular author mapping detected: {cycle}")
        seen.append(current)
        if target_ids.get(current, ""):
            resolved_id = target_ids[current]
        next_name = replacement[current]
        if not next_name or next_name == current:
            break
        current = next_name
    return resolved_id


def resolve_author_id_rule(author_id, rules):
    seen = []
    current = author_id
    canonical_name = ""
    while current in rules:
        if current in seen:
            cycle = " -> ".join(seen + [current])
            raise ValueError(f"Circular author ID mapping detected: {cycle}")
        seen.append(current)
        next_name, next_id = rules[current]
        if next_name:
            canonical_name = next_name
        next_id = next_id or current
        if next_id == current:
            break
        current = next_id
    return canonical_name, current



def load_author_mapping(mapping_path):
    if not mapping_path or not os.path.exists(mapping_path):
        return empty_mapping_data()
    mapping_df = pd.read_csv(mapping_path, dtype=str).fillna("")
    if not {"Variant", "Canonical"}.issubset(mapping_df.columns):
        raise ValueError("Manual author mapping must contain Variant and Canonical columns")
    if "Variant ID" not in mapping_df.columns:
        mapping_df["Variant ID"] = ""
    if "Canonical ID" not in mapping_df.columns:
        mapping_df["Canonical ID"] = ""
    rows = []
    name_grouped = defaultdict(set)
    for _, row in mapping_df.iterrows():
        variant = clean_author_name(row.get("Variant", ""))
        variant_id = clean_author_id(row.get("Variant ID", ""))
        canonical = clean_author_name(row.get("Canonical", ""))
        canonical_id = clean_author_id(row.get("Canonical ID", ""))
        if not variant or not canonical:
            continue
        effective_canonical_id = canonical_id or variant_id
        rows.append({
            "Variant": variant,
            "Variant ID": variant_id,
            "Canonical": canonical,
            "Canonical ID": effective_canonical_id
        })
        name_grouped[variant].add(canonical)
    name_conflicts = {variant: values for variant, values in name_grouped.items() if len(values) > 1}
    if name_conflicts:
        details = [
            f"{variant}: {sorted(values)}"
            for variant, values in sorted(name_conflicts.items())[:20]
        ]
        raise ValueError(f"Conflicting author name mappings detected: {'; '.join(details)}")
    raw_replacement = {variant: next(iter(values)) for variant, values in name_grouped.items()}
    exact = {variant: resolve_mapping_chain(canonical, raw_replacement) for variant, canonical in raw_replacement.items()}
    canonical_exact = set(exact.values())
    normalized_candidates = defaultdict(set)
    ordered_candidates = defaultdict(set)
    canonical_normalized_candidates = defaultdict(set)
    canonical_ordered_candidates = defaultdict(set)
    for variant, canonical in exact.items():
        normalized_candidates[normalize_for_similarity(variant)].add(canonical)
        ordered_candidates[author_order_key(variant)].add(canonical)
    for canonical in canonical_exact:
        canonical_normalized_candidates[normalize_for_similarity(canonical)].add(canonical)
        canonical_ordered_candidates[author_order_key(canonical)].add(canonical)
    normalized = {
        key: next(iter(values))
        for key, values in normalized_candidates.items()
        if key and len(values) == 1
    }
    ordered = {
        key: next(iter(values))
        for key, values in ordered_candidates.items()
        if key and len(values) == 1
    }
    canonical_normalized = {
        key: next(iter(values))
        for key, values in canonical_normalized_candidates.items()
        if key and len(values) == 1
    }
    canonical_ordered = {
        key: next(iter(values))
        for key, values in canonical_ordered_candidates.items()
        if key and len(values) == 1
    }
    reference_authors = set()
    for row in rows:
        reference_authors.add((row["Variant"], row["Variant ID"]))
        reference_authors.add((row["Canonical"], row["Canonical ID"]))
    return {
        "exact": exact,
        "normalized": normalized,
        "ordered": ordered,
        "exact_ids": {},
        "normalized_ids": {},
        "ordered_ids": {},
        "id_rules": {},
        "canonical_exact": canonical_exact,
        "canonical_normalized": canonical_normalized,
        "canonical_ordered": canonical_ordered,
        "canonical_ids": set(),
        "canonical_id_names": {},
        "reference_names": set(exact) | canonical_exact,
        "reference_authors": reference_authors,
        "rows": rows,
        "loaded": len(rows)
    }


def canonicalize_author(name, mapping):
    name = clean_author_name(name)
    if not name:
        return ""
    if name in mapping["exact"]:
        return mapping["exact"][name]
    normalized = normalize_for_similarity(name)
    if normalized in mapping["normalized"]:
        return mapping["normalized"][normalized]
    if normalized in mapping["canonical_normalized"]:
        return mapping["canonical_normalized"][normalized]
    ordered = author_order_key(name)
    if ordered in mapping["ordered"]:
        return mapping["ordered"][ordered]
    if ordered in mapping["canonical_ordered"]:
        return mapping["canonical_ordered"][ordered]
    return name



def canonicalize_author_identity(name, author_id, mapping):
    name = clean_author_name(name)
    author_id = clean_author_id(author_id)
    if not name:
        return "", author_id
    return canonicalize_author(name, mapping), author_id



def author_is_standardized(name, author_id, mapping):
    name = clean_author_name(name)
    if not name:
        return True
    if name in mapping["exact"] or name in mapping["canonical_exact"]:
        return True
    normalized = normalize_for_similarity(name)
    ordered = author_order_key(name)
    return (
        normalized in mapping["normalized"]
        or normalized in mapping["canonical_normalized"]
        or ordered in mapping["ordered"]
        or ordered in mapping["canonical_ordered"]
    )


def apply_manual_author_mapping(author_lists, author_id_lists, mapping):
    mapped_names = []
    mapped_ids = []
    for authors, author_ids in zip(author_lists, author_id_lists):
        names = []
        ids = []
        seen = set()
        for author, author_id in zip(authors, align_author_ids(authors, author_ids)):
            canonical, canonical_id = canonicalize_author_identity(author, author_id, mapping)
            key = author_identity_key(canonical, canonical_id)
            if canonical and key and key not in seen:
                seen.add(key)
                names.append(canonical)
                ids.append(canonical_id)
        mapped_names.append(names)
        mapped_ids.append(ids)
    return mapped_names, mapped_ids


def author_relation(first, second):
    first_set = set(first)
    second_set = set(second)
    if not first_set or not second_set:
        return "unknown", 0.0
    if first_set == second_set:
        return "exact", 1.0
    intersection = len(first_set & second_set)
    containment = intersection / min(len(first_set), len(second_set))
    jaccard = intersection / len(first_set | second_set)
    score = round((containment + jaccard) / 2, 4)
    if containment >= 0.8 and jaccard >= 0.5:
        return "strong", score
    return "different", round(jaccard, 4)


def completeness_score(row):
    fields = [
        "DOI", "Title", "Year", "Source title", "Volume", "Issue", "Art. No.",
        "Affiliations", "Authors with affiliations", "Abstract", "Author Keywords",
        "Index Keywords", "Publisher", "Language of Original Document",
        "Document Type", "Open Access", "Author full names"
    ]
    populated = sum(bool(normalize_space(row.get(field, ""))) for field in fields)
    detail = sum(
        len(normalize_space(row.get(field, "")))
        for field in [
            "Abstract", "Author Keywords", "Index Keywords", "Affiliations",
            "Authors with affiliations", "Author full names"
        ]
    )
    cited = pd.to_numeric(pd.Series([row.get("Cited by", "")]), errors="coerce").fillna(0).iloc[0]
    return populated * 1000000 + min(detail, 999999) + float(cited) / 1000000


def duplicate_author_data(row, mapping):
    normalized = row.get("_Normalized_author_list", None)
    normalized_ids = row.get("_Normalized_author_id_list", None)
    recovered = row.get("_Recovered_author_list", None)
    recovered_ids = row.get("_Recovered_author_id_list", None)
    if isinstance(normalized, (list, tuple)):
        authors = list(normalized)
        author_ids = list(normalized_ids) if isinstance(normalized_ids, (list, tuple)) else [""] * len(authors)
    elif isinstance(recovered, (list, tuple)):
        authors = list(recovered)
        author_ids = list(recovered_ids) if isinstance(recovered_ids, (list, tuple)) else [""] * len(authors)
    else:
        author_columns = sorted(
            [column for column in row.index if re.fullmatch(r"Author_\d+", str(column))],
            key=lambda column: int(str(column).split("_")[-1])
        )
        authors = [clean_author_name(row.get(column, "")) for column in author_columns]
        authors = [author for author in authors if author]
        author_ids = numbered_column_sequence(row, "Author_ID")
        if not authors:
            authors = parse_scopus_authors(row.get("Author full names", ""))
            author_ids = parse_scopus_author_ids(row.get("Author(s) ID", ""))
    identities = []
    seen = set()
    for author, author_id in zip(authors, align_author_ids(authors, author_ids)):
        canonical, canonical_id = canonicalize_author_identity(author, author_id, mapping)
        key = author_identity_key(canonical, canonical_id)
        if canonical and key and key not in seen:
            seen.add(key)
            identities.append((canonical, canonical_id, key))
    return identities


def publication_notice_type(title, document_type):
    text = normalize_lookup_text(f"{title} {document_type}")
    if any(term in text.split() for term in {"retraction", "retracted", "withdrawal", "withdrawn"}):
        return "retraction"
    if "expression of concern" in text:
        return "expression_of_concern"
    if any(term in text.split() for term in {"correction", "erratum", "corrigendum"}):
        return "correction_or_erratum"
    return ""


def prepare_duplicate_work(df, mapping):
    work = df.copy()
    work["__row_order"] = range(len(work))
    work["__title_key"] = work["Title"].apply(title_key)
    work["__year_key"] = work["Year"].apply(year_key)
    work["__venue_key"] = [
        source_key(source, publisher)
        for source, publisher in zip(work["Source title"], work["Publisher"])
    ]
    work["__document_key"] = work["Document Type"].apply(normalize_lookup_text)
    work["__doi_key"] = work["DOI"].apply(normalize_doi)
    author_data = [duplicate_author_data(row, mapping) for _, row in work.iterrows()]
    work["__author_signature"] = [
        tuple(sorted({identity[2] for identity in identities if identity[2]}))
        for identities in author_data
    ]
    work["__first_author_key"] = [
        identities[0][2] if identities else ""
        for identities in author_data
    ]
    work["__notice_type"] = [
        publication_notice_type(title, document_type)
        for title, document_type in zip(work["Title"], work["Document Type"])
    ]
    work["__completeness_score"] = work.apply(completeness_score, axis=1)
    return work


def candidate_duplicate_pairs(work):
    pairs = set()
    doi_rows = work[work["__doi_key"] != ""]
    for _, group in doi_rows.groupby("__doi_key"):
        indices = list(group.index)
        pairs.update(tuple(sorted(pair)) for pair in combinations(indices, 2))
    title_rows = work[(work["__title_key"] != "") & (work["__year_key"] != "")]
    for _, group in title_rows.groupby(["__title_key", "__year_key"]):
        indices = list(group.index)
        pairs.update(tuple(sorted(pair)) for pair in combinations(indices, 2))
    blocked = title_rows.copy()
    blocked["__title_block"] = blocked["Title"].apply(title_block_key)
    for _, group in blocked.groupby(["__year_key", "__title_block"]):
        indices = list(group.index)
        for first, second in combinations(indices, 2):
            title_first = work.at[first, "__title_key"]
            title_second = work.at[second, "__title_key"]
            if not title_first or not title_second:
                continue
            length_ratio = min(len(title_first), len(title_second)) / max(len(title_first), len(title_second))
            if length_ratio >= 0.6:
                pairs.add(tuple(sorted((first, second))))
    venue_rows = blocked[blocked["__venue_key"] != ""]
    for _, group in venue_rows.groupby(["__year_key", "__venue_key"]):
        indices = list(group.index)
        if len(indices) < 2 or len(indices) > 250:
            continue
        for first, second in combinations(indices, 2):
            title_first = work.at[first, "__title_key"]
            title_second = work.at[second, "__title_key"]
            if not title_first or not title_second:
                continue
            length_ratio = min(len(title_first), len(title_second)) / max(len(title_first), len(title_second))
            if length_ratio < 0.75:
                continue
            tokens_first = set(title_first.split())
            tokens_second = set(title_second.split())
            if tokens_first and tokens_second:
                overlap = len(tokens_first & tokens_second) / min(len(tokens_first), len(tokens_second))
                if overlap >= 0.6:
                    pairs.add(tuple(sorted((first, second))))
    return sorted(pairs)


def compare_duplicate_pair(first, second, fuzzy_threshold):
    title_similarity = 0.0
    if first["__title_key"] and second["__title_key"]:
        title_similarity = float(fuzz.token_sort_ratio(first["__title_key"], second["__title_key"]))
    exact_title = bool(first["__title_key"]) and first["__title_key"] == second["__title_key"]
    year_match = bool(first["__year_key"]) and first["__year_key"] == second["__year_key"]
    venue_match = bool(first["__venue_key"]) and first["__venue_key"] == second["__venue_key"]
    if first["__document_key"] and second["__document_key"]:
        document_match = first["__document_key"] == second["__document_key"]
    else:
        document_match = True
    author_status, author_score = author_relation(first["__author_signature"], second["__author_signature"])
    first_author_match = bool(first["__first_author_key"]) and first["__first_author_key"] == second["__first_author_key"]
    same_doi = bool(first["__doi_key"]) and first["__doi_key"] == second["__doi_key"]
    conflicting_doi = bool(first["__doi_key"]) and bool(second["__doi_key"]) and first["__doi_key"] != second["__doi_key"]
    notice_match = first["__notice_type"] == second["__notice_type"]
    automatic = False
    reason = ""
    manual = False
    suggestion = ""
    if conflicting_doi:
        if exact_title and year_match:
            manual = True
            suggestion = "REVIEW_IDENTICAL_TITLE_YEAR_CONFLICTING_DOI"
        elif title_similarity >= fuzzy_threshold and year_match:
            manual = True
            suggestion = "REVIEW_FUZZY_TITLE_CONFLICTING_DOI"
    elif same_doi:
        if exact_title and year_match and notice_match:
            automatic = True
            reason = "same_normalized_doi_title_and_year"
        else:
            manual = True
            if not exact_title and not year_match:
                suggestion = "REVIEW_SHARED_DOI_DIFFERENT_TITLE_AND_YEAR"
            elif not exact_title:
                suggestion = "REVIEW_SHARED_DOI_DIFFERENT_TITLE"
            elif not year_match:
                suggestion = "REVIEW_SHARED_DOI_DIFFERENT_YEAR"
            else:
                suggestion = "REVIEW_SHARED_DOI_CORRECTION_OR_ERRATUM"
    elif exact_title and year_match and first_author_match:
        automatic = True
        reason = "normalized_title_year_first_author_with_missing_doi"
    elif exact_title and year_match:
        manual = True
        suggestion = "REVIEW_IDENTICAL_TITLE_AND_YEAR"
    elif title_similarity >= fuzzy_threshold and year_match:
        manual = True
        suggestion = "REVIEW_FUZZY_TITLE"
    return {
        "Title_similarity": round(title_similarity, 2),
        "Exact_title": exact_title,
        "Year_match": year_match,
        "Venue_match": venue_match,
        "Document_type_match": document_match,
        "Author_match": author_status,
        "Author_similarity": author_score,
        "First_author_match": first_author_match,
        "Same_DOI": same_doi,
        "Conflicting_DOI": conflicting_doi,
        "Automatic_duplicate": automatic,
        "Removal_reason": reason,
        "Manual_review": manual,
        "Suggested_review": suggestion
    }


def pair_identifier(dataset, first, second):
    first_signature = "|".join([
        first["__doi_key"], first["__title_key"], first["__year_key"],
        first["__venue_key"], ",".join(first["__author_signature"])
    ])
    second_signature = "|".join([
        second["__doi_key"], second["__title_key"], second["__year_key"],
        second["__venue_key"], ",".join(second["__author_signature"])
    ])
    payload = dataset + "||" + "||".join(sorted([first_signature, second_signature]))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def duplicate_report_row(dataset, pair_id, first, second, comparison):
    return {
        "Pair_ID": pair_id,
        "Dataset": dataset,
        "Record_1_source_row": first.get("__source_row", ""),
        "Record_2_source_row": second.get("__source_row", ""),
        "Record_1_title": first.get("Title", ""),
        "Record_2_title": second.get("Title", ""),
        "Record_1_DOI": first.get("DOI", ""),
        "Record_2_DOI": second.get("DOI", ""),
        "Record_1_year": first.get("Year", ""),
        "Record_2_year": second.get("Year", ""),
        "Record_1_source": first.get("Source title", ""),
        "Record_2_source": second.get("Source title", ""),
        "Record_1_document_type": first.get("Document Type", ""),
        "Record_2_document_type": second.get("Document Type", ""),
        "Record_1_authors": first.get("Author full names", ""),
        "Record_2_authors": second.get("Author full names", ""),
        **comparison
    }


def load_duplicate_decisions(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        frame = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    if not {"Pair_ID", "Decision"}.issubset(frame.columns):
        return {}
    allowed = {"KEEP_BOTH", "REMOVE_RECORD_1", "REMOVE_RECORD_2"}
    decisions = {}
    for _, row in frame.iterrows():
        pair_id = normalize_space(row.get("Pair_ID", ""))
        decision = normalize_space(row.get("Decision", "")).upper()
        if pair_id and decision in allowed:
            decisions[pair_id] = decision
    return decisions


class DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.members = {value: {value} for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return root_first
        if len(self.members[root_first]) < len(self.members[root_second]):
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.members[root_first].update(self.members.pop(root_second))
        return root_first

    def component(self, value):
        return self.members[self.find(value)]

def merge_value_present(value):
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(normalize_space(str(value)))


def split_merged_values(value, pattern=r";"):
    if not merge_value_present(value):
        return []
    return [normalize_space(part) for part in re.split(pattern, str(value)) if normalize_space(part)]


def union_record_values(work, indices, column, pattern=r";"):
    values = []
    seen = set()
    for index in indices:
        for value in split_merged_values(work.at[index, column] if column in work.columns else "", pattern):
            key = normalize_lookup_text(value)
            if key and key not in seen:
                seen.add(key)
                values.append(value)
    return values


def numbered_column_values(row, prefix):
    columns = sorted(
        [column for column in row.index if re.fullmatch(fr"{re.escape(prefix)}_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[-1])
    )
    return [normalize_space(str(row.get(column, ""))) for column in columns if merge_value_present(row.get(column, ""))]


def numbered_column_sequence(row, prefix):
    columns = sorted(
        [column for column in row.index if re.fullmatch(fr"{re.escape(prefix)}_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[-1])
    )
    return [normalize_space(str(row.get(column, ""))) if merge_value_present(row.get(column, "")) else "" for column in columns]


def set_numbered_column_values(work, index, prefix, values):
    columns = sorted(
        [column for column in work.columns if re.fullmatch(fr"{re.escape(prefix)}_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[-1])
    )
    while len(columns) < len(values):
        column = f"{prefix}_{len(columns) + 1}"
        work[column] = ""
        columns.append(column)
    for position, column in enumerate(columns):
        work.at[index, column] = values[position] if position < len(values) else ""


def raw_author_data_from_row(row):
    recovered = row.get("_Recovered_author_list", None)
    recovered_ids = row.get("_Recovered_author_id_list", None)
    if isinstance(recovered, (list, tuple)):
        authors = [clean_author_name(value) for value in recovered if clean_author_name(value)]
        author_ids = list(recovered_ids) if isinstance(recovered_ids, (list, tuple)) else [""] * len(authors)
        return authors, align_author_ids(authors, author_ids)
    authors = numbered_column_values(row, "Author")
    if authors:
        authors = [clean_author_name(value) for value in authors if clean_author_name(value)]
        return authors, align_author_ids(authors, numbered_column_sequence(row, "Author_ID"))
    authors = parse_scopus_authors(row.get("Author full names", ""))
    return authors, align_author_ids(authors, parse_scopus_author_ids(row.get("Author(s) ID", "")))


def preferred_author_data(work, indices, mapping):
    candidates = []
    for index in indices:
        raw_authors, raw_author_ids = raw_author_data_from_row(work.loc[index])
        canonical_authors = []
        canonical_author_ids = []
        seen = set()
        for author, author_id in zip(raw_authors, raw_author_ids):
            canonical, canonical_id = canonicalize_author_identity(author, author_id, mapping)
            key = author_identity_key(canonical, canonical_id)
            if canonical and key and key not in seen:
                seen.add(key)
                canonical_authors.append(canonical)
                canonical_author_ids.append(canonical_id)
        source = normalize_lookup_text(
            work.at[index, "_Author_recovery_source"] if "_Author_recovery_source" in work.columns
            else work.at[index, "Author_recovery_source"] if "Author_recovery_source" in work.columns
            else ""
        )
        candidates.append((
            len(canonical_authors),
            source == "crossref",
            work.at[index, "__completeness_score"],
            -work.at[index, "__row_order"],
            index,
            raw_authors,
            raw_author_ids,
            canonical_authors,
            canonical_author_ids
        ))
    selected = max(candidates)
    return selected[4], selected[5], selected[6], selected[7], selected[8]


def open_access_affirmative(value):
    key = normalize_lookup_text(value)
    if not key or key in {"no", "none", "closed", "closed access", "no open access", "not open access", "non open access", "false", "0"}:
        return False
    return key in {"yes", "true", "1"} or "open access" in key or bool(set(key.split()) & {"gold", "green", "bronze", "hybrid"})


def merge_duplicate_cluster(work, indices, kept, mapping):
    precedence = [kept] + sorted(
        [index for index in indices if index != kept],
        key=lambda index: (work.at[index, "__completeness_score"], -work.at[index, "__row_order"]),
        reverse=True
    )
    selected_author_index, raw_authors, raw_author_ids, canonical_authors, canonical_author_ids = preferred_author_data(work, indices, mapping)
    special_columns = {
        "Cited by", "DOI", "Abstract", "Author Keywords", "Index Keywords", "Affiliations",
        "Open Access", "Search_Category", "Search Category", "Author full names", "Author(s) ID",
        "Authors with affiliations", "_Recovered_author_list", "_Recovered_author_id_list",
        "_Normalized_author_list", "_Normalized_author_id_list", "_Author_recovery_source",
        "_Listed_author_count", "Author_recovery_source",
        "Listed_author_count_before_crossref", "Final_author_count"
    }
    for column in work.columns:
        if str(column).startswith("__") or column in special_columns or re.fullmatch(r"(?:Author|Author_ID|Country)_\d+", str(column)):
            continue
        if merge_value_present(work.at[kept, column]):
            continue
        for index in precedence[1:]:
            if merge_value_present(work.at[index, column]):
                work.at[kept, column] = work.at[index, column]
                break
    citation_values = pd.to_numeric(work.loc[indices, "Cited by"], errors="coerce").fillna(0)
    work.at[kept, "Cited by"] = int(citation_values.max())
    for index in precedence:
        doi = normalize_doi(work.at[index, "DOI"])
        if doi:
            work.at[kept, "DOI"] = doi
            break
    abstracts = [normalize_space(str(work.at[index, "Abstract"])) for index in precedence if merge_value_present(work.at[index, "Abstract"])]
    invalid_abstracts = {"n a", "na", "none", "not available", "no abstract", "no abstract available", "abstract unavailable"}
    valid_abstracts = [value for value in abstracts if normalize_lookup_text(value) not in invalid_abstracts]
    if valid_abstracts:
        work.at[kept, "Abstract"] = max(valid_abstracts, key=len)
    for column in ["Author Keywords", "Index Keywords", "Affiliations"]:
        values = union_record_values(work, precedence, column)
        if values:
            work.at[kept, column] = "; ".join(values)
    for column in ["Search_Category", "Search Category"]:
        if column in work.columns:
            values = union_record_values(work, precedence, column, r"[;|]")
            if values:
                work.at[kept, column] = "; ".join(values)
    if "Open Access" in work.columns:
        open_access_values = [work.at[index, "Open Access"] for index in precedence if merge_value_present(work.at[index, "Open Access"])]
        affirmative_values = [value for value in open_access_values if open_access_affirmative(value)]
        if affirmative_values:
            work.at[kept, "Open Access"] = affirmative_values[0]
        elif open_access_values:
            work.at[kept, "Open Access"] = open_access_values[0]
    work.at[kept, "Author full names"] = "; ".join(canonical_authors)
    author_precedence = [selected_author_index] + [index for index in precedence if index != selected_author_index]
    for column in ["Author(s) ID", "Authors with affiliations"]:
        if column in work.columns:
            for index in author_precedence:
                if merge_value_present(work.at[index, column]):
                    work.at[kept, column] = work.at[index, column]
                    break
    if "_Recovered_author_list" in work.columns:
        work.at[kept, "_Recovered_author_list"] = raw_authors
    if "_Recovered_author_id_list" in work.columns:
        work.at[kept, "_Recovered_author_id_list"] = raw_author_ids
    if "_Normalized_author_list" in work.columns:
        work.at[kept, "_Normalized_author_list"] = canonical_authors
    if "_Normalized_author_id_list" in work.columns:
        work.at[kept, "_Normalized_author_id_list"] = canonical_author_ids
    for target in ["_Author_recovery_source", "Author_recovery_source", "_Listed_author_count", "Listed_author_count_before_crossref"]:
        if target in work.columns:
            for index in author_precedence:
                if merge_value_present(work.at[index, target]):
                    work.at[kept, target] = work.at[index, target]
                    break
    if "Final_author_count" in work.columns:
        work.at[kept, "Final_author_count"] = len(canonical_authors)
    if any(re.fullmatch(r"Author_\d+", str(column)) for column in work.columns):
        set_numbered_column_values(work, kept, "Author", canonical_authors)
    if any(re.fullmatch(r"Author_ID_\d+", str(column)) for column in work.columns):
        set_numbered_column_values(work, kept, "Author_ID", canonical_author_ids)
    for prefix in ["Country"]:
        if any(re.fullmatch(fr"{prefix}_\d+", str(column)) for column in work.columns):
            values = []
            seen = set()
            for index in precedence:
                for value in numbered_column_values(work.loc[index], prefix):
                    key = normalize_lookup_text(value)
                    if key and key not in seen:
                        seen.add(key)
                        values.append(value)
            set_numbered_column_values(work, kept, prefix, values)


def component_separation_conflict(disjoint, first, second, adjacency):
    first_members = disjoint.component(first)
    second_members = disjoint.component(second)
    if len(first_members) > len(second_members):
        first_members, second_members = second_members, first_members
    return any(adjacency.get(index, set()) & second_members for index in first_members)


def component_doi_values(work, disjoint, index):
    return {
        work.at[member, "__doi_key"]
        for member in disjoint.component(index)
        if work.at[member, "__doi_key"]
    }


def component_doi_compatible(first_values, second_values):
    if not first_values or not second_values:
        return True
    return first_values.issubset(second_values) or second_values.issubset(first_values)


def component_conflict_representative(work, disjoint, index, preferred_dois):
    members = sorted(
        disjoint.component(index),
        key=lambda member: (
            work.at[member, "__row_order"],
            -work.at[member, "__completeness_score"]
        )
    )
    preferred = [member for member in members if work.at[member, "__doi_key"] in preferred_dois]
    doi_members = [member for member in members if work.at[member, "__doi_key"]]
    return (preferred or doi_members or members)[0]


def component_selected_representative(work, indices, forced_removals, preferred_targets):
    return sorted(
        indices,
        key=lambda index: (
            index not in forced_removals,
            preferred_targets[index],
            work.at[index, "__completeness_score"],
            -work.at[index, "__row_order"]
        ),
        reverse=True
    )[0]


def automatic_components_match_selected_representative(
    work,
    disjoint,
    manual_disjoint,
    first,
    second,
    helper_columns,
    fuzzy_threshold,
    forced_removals,
    preferred_targets
):
    members = set(disjoint.component(first)) | set(disjoint.component(second))
    representative = component_selected_representative(
        work,
        members,
        forced_removals,
        preferred_targets
    )
    manual_groups = defaultdict(list)
    for member in members:
        manual_groups[manual_disjoint.find(member)].append(member)
    representative_group = manual_disjoint.find(representative)
    representative_members = manual_groups[representative_group]
    for group_root, group_members in manual_groups.items():
        if group_root == representative_group:
            continue
        if not any(
            compare_duplicate_pair(
                work.loc[representative_member, helper_columns],
                work.loc[member, helper_columns],
                fuzzy_threshold
            )["Automatic_duplicate"]
            for representative_member in representative_members
            for member in group_members
        ):
            return False
    return True


def deduplicate_records(df, dataset, mapping, fuzzy_threshold, decisions=None):
    decisions = decisions or {}
    work = prepare_duplicate_work(df, mapping)
    pairs = candidate_duplicate_pairs(work)
    disjoint = DisjointSet(work.index)
    manual_disjoint = DisjointSet(work.index)
    manual_reports = {}
    automatic_edges = []
    manual_edges = []
    keep_both_adjacency = defaultdict(set)
    forced_removals = defaultdict(list)
    preferred_targets = Counter()
    helper_columns = [
        "__doi_key", "__title_key", "__year_key", "__venue_key",
        "__document_key", "__author_signature", "__first_author_key",
        "__notice_type"
    ]
    for first_index, second_index in pairs:
        first_helper = work.loc[first_index, helper_columns]
        second_helper = work.loc[second_index, helper_columns]
        comparison = compare_duplicate_pair(first_helper, second_helper, fuzzy_threshold)
        if not comparison["Automatic_duplicate"] and not comparison["Manual_review"]:
            continue
        first = work.loc[first_index]
        second = work.loc[second_index]
        pair_id = pair_identifier(dataset, first, second)
        report = duplicate_report_row(dataset, pair_id, first, second, comparison)
        decision = decisions.get(pair_id, "")
        if comparison["Automatic_duplicate"] and not decision:
            automatic_edges.append((first_index, second_index, comparison, pair_id))
            continue
        if comparison["Automatic_duplicate"]:
            report["Automatic_duplicate"] = False
            report["Manual_review"] = True
            report["Removal_reason"] = ""
            report["Suggested_review"] = "MANUAL_OVERRIDE_OF_AUTOMATIC_DUPLICATE"
        report["Decision"] = decision
        manual_reports[pair_id] = report
        if decision == "REMOVE_RECORD_1":
            manual_edges.append((first_index, second_index, pair_id, first_index, second_index))
        elif decision == "REMOVE_RECORD_2":
            manual_edges.append((first_index, second_index, pair_id, second_index, first_index))
        elif decision == "KEEP_BOTH":
            keep_both_adjacency[first_index].add(second_index)
            keep_both_adjacency[second_index].add(first_index)
    for first_index, second_index, pair_id, removed_index, target_index in manual_edges:
        if component_separation_conflict(disjoint, first_index, second_index, keep_both_adjacency):
            manual_reports[pair_id]["Decision_status"] = "NOT_APPLIED_CONTRADICTORY_KEEP_BOTH"
            continue
        disjoint.union(first_index, second_index)
        manual_disjoint.union(first_index, second_index)
        forced_removals[removed_index].append(pair_id)
        preferred_targets[target_index] += 1
        manual_reports[pair_id]["Decision_status"] = "APPLIED_TO_FINAL_COMPONENT"
    automatic_edges.sort(
        key=lambda item: (
            0 if item[2]["Same_DOI"] else 1,
            0 if item[2]["Exact_title"] else 1,
            -item[2]["Title_similarity"],
            work.at[item[0], "__row_order"],
            work.at[item[1], "__row_order"]
        )
    )
    automatic_edge_pairs = {}
    for first_index, second_index, comparison, pair_id in automatic_edges:
        automatic_edge_pairs[frozenset((first_index, second_index))] = (comparison, pair_id)
        if disjoint.find(first_index) == disjoint.find(second_index):
            continue
        if component_separation_conflict(disjoint, first_index, second_index, keep_both_adjacency):
            continue
        first_dois = component_doi_values(work, disjoint, first_index)
        second_dois = component_doi_values(work, disjoint, second_index)
        if component_doi_compatible(first_dois, second_dois):
            if automatic_components_match_selected_representative(
                work,
                disjoint,
                manual_disjoint,
                first_index,
                second_index,
                helper_columns,
                fuzzy_threshold,
                forced_removals,
                preferred_targets
            ):
                disjoint.union(first_index, second_index)
            continue
        first_rep = component_conflict_representative(work, disjoint, first_index, first_dois - second_dois)
        second_rep = component_conflict_representative(work, disjoint, second_index, second_dois - first_dois)
        conflict_pair_id = pair_identifier(dataset, work.loc[first_rep], work.loc[second_rep])
        conflict_comparison = compare_duplicate_pair(
            work.loc[first_rep, helper_columns],
            work.loc[second_rep, helper_columns],
            fuzzy_threshold
        )
        conflict_comparison["Automatic_duplicate"] = False
        conflict_comparison["Manual_review"] = True
        conflict_comparison["Removal_reason"] = ""
        conflict_comparison["Suggested_review"] = "REVIEW_COMPONENT_CONFLICTING_DOI"
        conflict_report = duplicate_report_row(
            dataset,
            conflict_pair_id,
            work.loc[first_rep],
            work.loc[second_rep],
            conflict_comparison
        )
        conflict_decision = decisions.get(conflict_pair_id, "")
        conflict_report["Decision"] = conflict_decision
        manual_reports[conflict_pair_id] = conflict_report
        if conflict_decision == "REMOVE_RECORD_1":
            if not component_separation_conflict(disjoint, first_rep, second_rep, keep_both_adjacency):
                forced_removals[first_rep].append(conflict_pair_id)
                preferred_targets[second_rep] += 1
                disjoint.union(first_rep, second_rep)
                manual_disjoint.union(first_rep, second_rep)
                manual_reports[conflict_pair_id]["Decision_status"] = "APPLIED_TO_FINAL_COMPONENT"
        elif conflict_decision == "REMOVE_RECORD_2":
            if not component_separation_conflict(disjoint, first_rep, second_rep, keep_both_adjacency):
                forced_removals[second_rep].append(conflict_pair_id)
                preferred_targets[first_rep] += 1
                disjoint.union(first_rep, second_rep)
                manual_disjoint.union(first_rep, second_rep)
                manual_reports[conflict_pair_id]["Decision_status"] = "APPLIED_TO_FINAL_COMPONENT"
        elif conflict_decision == "KEEP_BOTH":
            keep_both_adjacency[first_rep].add(second_rep)
            keep_both_adjacency[second_rep].add(first_rep)
    groups = defaultdict(list)
    for index in work.index:
        groups[disjoint.find(index)].append(index)
    kept_indices = []
    removal_rows = []
    for indices in groups.values():
        ranked = sorted(
            indices,
            key=lambda index: (
                index not in forced_removals,
                preferred_targets[index],
                work.at[index, "__completeness_score"],
                -work.at[index, "__row_order"]
            ),
            reverse=True
        )
        kept = ranked[0]
        kept_indices.append(kept)
        cluster_removed = [index for index in indices if index != kept]
        for removed in cluster_removed:
            comparison = compare_duplicate_pair(work.loc[kept, helper_columns], work.loc[removed, helper_columns], fuzzy_threshold)
            pair_id = pair_identifier(dataset, work.loc[kept], work.loc[removed])
            row = duplicate_report_row(dataset, pair_id, work.loc[kept], work.loc[removed], comparison)
            direct_edge = automatic_edge_pairs.get(frozenset((kept, removed)))
            if removed in forced_removals:
                row["Removal_reason"] = "manual_duplicate_decision"
                row["Manual_decision_pair_ID"] = "; ".join(forced_removals[removed])
            elif direct_edge:
                row["Removal_reason"] = direct_edge[0]["Removal_reason"]
            else:
                row["Removal_reason"] = "resolved_duplicate_component"
            row["Kept_source_row"] = work.at[kept, "__source_row"]
            row["Removed_source_row"] = work.at[removed, "__source_row"]
            row["Kept_title"] = work.at[kept, "Title"]
            row["Removed_title"] = work.at[removed, "Title"]
            row["Kept_DOI"] = work.at[kept, "DOI"]
            row["Removed_DOI"] = work.at[removed, "DOI"]
            removal_rows.append(row)
        if cluster_removed:
            merge_duplicate_cluster(work, indices, kept, mapping)
    result = work.loc[kept_indices].sort_values("__row_order").copy()
    helper_columns_to_drop = [column for column in result.columns if str(column).startswith("__")]
    result = result.drop(columns=helper_columns_to_drop)
    return result, removal_rows, list(manual_reports.values())

def fetch_crossref_authors(doi, mailto, timeout, retries, sleep_seconds):
    doi = normalize_doi(doi)
    if not doi:
        return None, "missing_doi"
    headers = {"User-Agent": "ScopusBibliometricPreprocessing/1.0"}
    if mailto:
        headers["User-Agent"] = f"ScopusBibliometricPreprocessing/1.0 (mailto:{mailto})"
    url = "https://api.crossref.org/works/" + requests.utils.quote(doi, safe="")
    status = "request_error"
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                raw_authors = data.get("message", {}).get("author", [])
                authors = []
                for author in raw_authors:
                    given = clean_author_name(author.get("given", ""))
                    family = clean_author_name(author.get("family", ""))
                    if family and given:
                        authors.append(f"{family}, {given}")
                    elif family:
                        authors.append(family)
                    elif given:
                        authors.append(given)
                authors = [value for value in authors if value]
                time.sleep(sleep_seconds)
                return authors if authors else None, "found_no_authors"
            status = f"http_{response.status_code}"
            if response.status_code in {404, 410}:
                time.sleep(sleep_seconds)
                return None, status
        except requests.Timeout:
            status = "timeout"
        except requests.ConnectionError:
            status = "connection_error"
        except requests.RequestException:
            status = "request_error"
        except (ValueError, TypeError, KeyError):
            status = "invalid_response"
        except Exception:
            status = "request_error"
        time.sleep(sleep_seconds * (attempt + 1))
    return None, status


def load_crossref_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
        reusable = {}
        for doi, entry in cache.items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status", "")
            if (isinstance(entry.get("authors"), list) and entry.get("authors")) or status in {"http_404", "http_410"}:
                reusable[doi] = entry
        return reusable
    return {}


def save_crossref_cache(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)


def author_surname_key(name):
    name = clean_author_name(name)
    if not name:
        return ""
    if "," in name:
        surname = name.split(",", 1)[0]
    else:
        tokens = normalize_for_similarity(name).split()
        surname = tokens[-1] if tokens else ""
    return normalize_for_similarity(surname)


def align_scopus_ids_to_recovered(scopus_authors, scopus_author_ids, recovered_authors):
    scopus_author_ids = align_author_ids(scopus_authors, scopus_author_ids)
    recovered_ids = [""] * len(recovered_authors)
    unmatched = set(range(len(recovered_authors)))
    for scopus_index, (scopus_author, scopus_id) in enumerate(zip(scopus_authors, scopus_author_ids)):
        if not scopus_id:
            continue
        normalized = normalize_for_similarity(scopus_author)
        ordered = author_order_key(scopus_author)
        exact = [index for index in unmatched if normalize_for_similarity(recovered_authors[index]) == normalized]
        candidates = exact if exact else [index for index in unmatched if author_order_key(recovered_authors[index]) == ordered]
        if len(candidates) == 1:
            selected = candidates[0]
            recovered_ids[selected] = scopus_id
            unmatched.remove(selected)
            continue
        if scopus_index in unmatched and author_surname_key(scopus_author) == author_surname_key(recovered_authors[scopus_index]):
            recovered_ids[scopus_index] = scopus_id
            unmatched.remove(scopus_index)
            continue
        surname_candidates = [
            index for index in unmatched
            if author_surname_key(scopus_author) and author_surname_key(scopus_author) == author_surname_key(recovered_authors[index])
        ]
        if surname_candidates:
            ranked = sorted(
                (
                    fuzz.token_sort_ratio(normalized, normalize_for_similarity(recovered_authors[index])),
                    -abs(index - scopus_index),
                    index
                )
                for index in surname_candidates
            )
            best_score, _, selected = ranked[-1]
            if best_score >= 70:
                recovered_ids[selected] = scopus_id
                unmatched.remove(selected)
    return recovered_ids


def recover_authors(row, cache, mailto, timeout, retries, sleep_seconds, skip_crossref):
    scopus_authors = parse_scopus_authors(row.get("Author full names", ""))
    scopus_author_ids = align_author_ids(scopus_authors, parse_scopus_author_ids(row.get("Author(s) ID", "")))
    doi = normalize_doi(row.get("DOI", ""))
    if skip_crossref or len(scopus_authors) < 10 or not doi:
        return scopus_authors, scopus_author_ids, "scopus_not_queried", len(scopus_authors)
    if doi in cache:
        cached = cache[doi]
        crossref_authors = cached.get("authors")
        status = cached.get("status", "cached")
    else:
        crossref_authors, status = fetch_crossref_authors(doi, mailto, timeout, retries, sleep_seconds)
        cache[doi] = {"authors": crossref_authors, "status": status}
    if isinstance(crossref_authors, list) and len(crossref_authors) > len(scopus_authors):
        authors = [clean_author_name(value) for value in crossref_authors if clean_author_name(value)]
        author_ids = align_scopus_ids_to_recovered(scopus_authors, scopus_author_ids, authors)
        return authors, author_ids, "crossref", len(scopus_authors)
    return scopus_authors, scopus_author_ids, status, len(scopus_authors)


def write_list_columns(df, lists, prefix):
    existing = [column for column in df.columns if re.fullmatch(fr"{prefix}_\d+", str(column))]
    if existing:
        df = df.drop(columns=existing)
    max_len = max([len(values) for values in lists], default=0)
    columns = {
        f"{prefix}_{index + 1}": [
            values[index] if index < len(values) else ""
            for values in lists
        ]
        for index in range(max_len)
    }
    if columns:
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    return df


def preferred_canonical_identity(authors, frequencies):
    def rank(author):
        name, author_id = author
        normalized = normalize_for_similarity(name)
        tokens = normalized.split()
        initials = sum(1 for token in tokens if len(token) == 1)
        return bool(author_id), frequencies.get(author, 0), len(tokens), -initials, len(name), name.casefold(), author_id
    return max(authors, key=rank)


def generate_similarity_candidates(author_lists, author_id_lists, output_path, threshold, mapping):
    frequencies = Counter()
    for authors, author_ids in zip(author_lists, author_id_lists):
        for author, author_id in zip(authors, align_author_ids(authors, author_ids)):
            if author:
                frequencies[(author, clean_author_id(author_id))] += 1
    authors = sorted(frequencies)
    unresolved = [author for author in authors if not author_is_standardized(author[0], author[1], mapping)]
    known_authors = sorted(mapping["reference_authors"])
    rows = []
    handled_same_id = set()
    unresolved_by_id = defaultdict(list)
    for author in unresolved:
        if author[1]:
            unresolved_by_id[author[1]].append(author)
    for author_id, values in sorted(unresolved_by_id.items()):
        unique_values = sorted(set(values))
        if len({normalize_for_similarity(value[0]) for value in unique_values}) < 2:
            continue
        canonical = preferred_canonical_identity(unique_values, frequencies)
        for variant in unique_values:
            if variant == canonical:
                continue
            rows.append({
                "Variant": variant[0],
                "Variant ID": variant[1],
                "Canonical": canonical[0],
                "Canonical ID": canonical[1],
                "Similarity": round(fuzz.token_sort_ratio(normalize_for_similarity(variant[0]), normalize_for_similarity(canonical[0])), 2)
            })
            handled_same_id.add(variant)
    known_by_order = defaultdict(set)
    for name, author_id in known_authors:
        canonical_name, canonical_id = canonicalize_author_identity(name, author_id, mapping)
        known_by_order[author_order_key(name)].add((canonical_name, canonical_id))
    unmatched = []
    for author in unresolved:
        if author in handled_same_id:
            continue
        name, author_id = author
        ordered = author_order_key(name)
        canonical_values = {
            value for value in known_by_order.get(ordered, set())
            if value[0] and value != author
        }
        if len(canonical_values) == 1:
            canonical_name, canonical_id = next(iter(canonical_values))
            rows.append({
                "Variant": name,
                "Variant ID": author_id,
                "Canonical": canonical_name,
                "Canonical ID": canonical_id,
                "Similarity": 100.0
            })
        else:
            unmatched.append(author)
    blocks = defaultdict(list)
    for author in unmatched:
        tokens = normalize_for_similarity(author[0]).split()
        keys = set(tokens[:1] + tokens[-1:])
        for key in keys:
            if key:
                blocks[key].append(author)
    pair_scores = {}
    for values in blocks.values():
        unique_values = sorted(set(values))
        for first, second in combinations(unique_values, 2):
            key = tuple(sorted((first, second)))
            if key in pair_scores:
                continue
            score = fuzz.token_sort_ratio(normalize_for_similarity(first[0]), normalize_for_similarity(second[0]))
            if score >= threshold:
                pair_scores[key] = score
    adjacency = defaultdict(set)
    for first, second in pair_scores:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = set()
    for author in unmatched:
        if author in visited or not adjacency[author]:
            continue
        stack = [author]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) < 2:
            continue
        canonical = preferred_canonical_identity(component, frequencies)
        for variant in sorted(component):
            if variant == canonical:
                continue
            score = pair_scores.get(tuple(sorted((variant, canonical))))
            if score is None:
                score = max(
                    pair_scores.get(tuple(sorted((variant, other))), 0)
                    for other in component
                    if other != variant
                )
            rows.append({
                "Variant": variant[0],
                "Variant ID": variant[1],
                "Canonical": canonical[0],
                "Canonical ID": canonical[1],
                "Similarity": round(score, 2)
            })
    columns = ["Variant", "Variant ID", "Canonical", "Canonical ID", "Similarity"]
    report = pd.DataFrame(rows, columns=columns)
    if not report.empty:
        report = report.drop_duplicates(subset=["Variant", "Variant ID"], keep="first")
        report = report.sort_values(["Similarity", "Variant", "Variant ID"], ascending=[False, True, True])
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(report)


def mapping_row_count(path):
    if not path or not os.path.exists(path):
        return -1
    try:
        mapping_df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return -1
    if not {"Variant", "Canonical"}.issubset(mapping_df.columns):
        return -1
    valid = (mapping_df["Variant"].str.strip() != "") & (mapping_df["Canonical"].str.strip() != "")
    return int(valid.sum())


def find_reference_mapping(output_dir, explicit_path):
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else ""
    candidates = [
        os.path.join(output_dir, "author_name_manual_mapping_template.csv"),
        "author_name_manual_mapping_template.csv",
        "author_name_manual_mapping.csv"
    ]
    available = []
    seen = set()
    for position, path in enumerate(candidates):
        absolute = os.path.abspath(path)
        if absolute in seen or not os.path.exists(path):
            continue
        seen.add(absolute)
        available.append((mapping_row_count(path), position, path))
    if not available:
        return ""
    available.sort(key=lambda value: (-value[0], value[1]))
    return available[0][2]


def create_mapping_template(path, seed_path):
    columns = ["Variant", "Variant ID", "Canonical", "Canonical ID"]
    existing_count = mapping_row_count(path)
    seed_count = mapping_row_count(seed_path)
    if os.path.exists(path):
        frame = pd.read_csv(path, dtype=str).fillna("")
        changed = False
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
                changed = True
        ordered = columns + [column for column in frame.columns if column not in columns]
        if changed or list(frame.columns) != ordered:
            frame[ordered].to_csv(path, index=False, encoding="utf-8-sig")
        if existing_count > 0:
            return
    if seed_count > existing_count and seed_path and os.path.abspath(seed_path) != os.path.abspath(path):
        seed = pd.read_csv(seed_path, dtype=str).fillna("")
        for column in columns:
            if column not in seed.columns:
                seed[column] = ""
        ordered = columns + [column for column in seed.columns if column not in columns]
        seed[ordered].to_csv(path, index=False, encoding="utf-8-sig")
        return
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def enrich_author_mapping_template(path, input_paths):
    if not path or not os.path.exists(path):
        return
    frame = pd.read_csv(path, dtype=str).fillna("")
    columns = ["Variant", "Variant ID", "Canonical", "Canonical ID"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    name_ids = defaultdict(set)
    for input_path in input_paths:
        if not input_path or not os.path.exists(input_path):
            continue
        source = pd.read_csv(input_path, dtype=str, usecols=lambda column: column in {"Author full names", "Author(s) ID"}).fillna("")
        if "Author full names" not in source.columns:
            continue
        if "Author(s) ID" not in source.columns:
            source["Author(s) ID"] = ""
        for names_value, ids_value in zip(source["Author full names"], source["Author(s) ID"]):
            authors = parse_scopus_authors(names_value)
            author_ids = align_author_ids(authors, parse_scopus_author_ids(ids_value))
            for author, author_id in zip(authors, author_ids):
                author_id = clean_author_id(author_id)
                if author and author_id:
                    name_ids[author].add(author_id)
    for index, row in frame.iterrows():
        if not clean_author_id(row.get("Variant ID", "")):
            values = name_ids.get(clean_author_name(row.get("Variant", "")), set())
            if len(values) == 1:
                frame.at[index, "Variant ID"] = next(iter(values))
    canonical_group_ids = defaultdict(set)
    for _, row in frame.iterrows():
        canonical = clean_author_name(row.get("Canonical", ""))
        variant_id = clean_author_id(row.get("Variant ID", ""))
        canonical_id = clean_author_id(row.get("Canonical ID", ""))
        if canonical and variant_id:
            canonical_group_ids[canonical].add(variant_id)
        if canonical and canonical_id:
            canonical_group_ids[canonical].add(canonical_id)
    for index, row in frame.iterrows():
        if clean_author_id(row.get("Canonical ID", "")):
            continue
        canonical = clean_author_name(row.get("Canonical", ""))
        exact_values = name_ids.get(canonical, set())
        if len(exact_values) == 1:
            frame.at[index, "Canonical ID"] = next(iter(exact_values))
            continue
        grouped_values = canonical_group_ids.get(canonical, set())
        if len(grouped_values) == 1:
            frame.at[index, "Canonical ID"] = next(iter(grouped_values))
    ordered = columns + [column for column in frame.columns if column not in columns]
    frame[ordered].to_csv(path, index=False, encoding="utf-8-sig")





def component_key(value):
    return normalize_lookup_text(clean_affiliation_component(value))


def empty_country_mapping_data():
    return {
        "exact": {},
        "rows": [],
        "reference_components": set(),
        "loaded": 0
    }


def affiliation_component_mapping_row_count(path):
    if not path or not os.path.exists(path):
        return -1
    try:
        frame = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return -1
    required = {"Variant", "Canonical", "Component_Type", "Decision"}
    if not required.issubset(frame.columns):
        return -1
    valid = (
        frame["Decision"].apply(decision_value).eq("accept")
        & (frame["Variant"].str.strip() != "")
        & (frame["Canonical"].str.strip() != "")
        & frame["Component_Type"].apply(component_type_value).ne("")
    )
    return int(valid.sum())


def find_reference_affiliation_component_mapping(output_dir, explicit_path):
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else ""
    candidates = [
        os.path.join(output_dir, "affiliation_component_manual_mapping_template.csv"),
        "affiliation_component_manual_mapping_template.csv",
        "affiliation_component_manual_mapping.csv"
    ]
    available = []
    seen = set()
    for position, path in enumerate(candidates):
        absolute = os.path.abspath(path)
        if absolute in seen or not os.path.exists(path):
            continue
        seen.add(absolute)
        available.append((affiliation_component_mapping_row_count(path), position, path))
    if not available:
        return ""
    available.sort(key=lambda value: (-value[0], value[1]))
    return available[0][2]


def affiliation_component_template_columns():
    return [
        "Variant", "Canonical", "Component_Type", "Decision", "Mentions",
        "Canonical Mentions", "Candidate Reason", "Token Sort Similarity",
        "Character Similarity", "Token Jaccard", "Token Containment",
        "Variant Full Affiliation"
    ]


def create_affiliation_component_mapping_template(path, seed_path=""):
    columns = affiliation_component_template_columns()
    existing_count = affiliation_component_mapping_row_count(path)
    seed_count = affiliation_component_mapping_row_count(seed_path)
    source_path = ""
    if os.path.exists(path):
        source_path = path
    if seed_path and os.path.exists(seed_path) and seed_count > existing_count:
        source_path = seed_path
    if source_path:
        frame = pd.read_csv(source_path, dtype=str).fillna("")
    else:
        frame = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[columns].copy()
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def resolve_country_mapping_chain(key, replacements):
    seen = []
    current = key
    while current in replacements:
        if current in seen:
            raise ValueError("Circular country mapping detected")
        seen.append(current)
        canonical = replacements[current]
        next_key = component_key(canonical)
        if not canonical or next_key == current or next_key not in replacements:
            return canonical
        current = next_key
    return replacements.get(current, "")


def load_country_mapping(mapping_path):
    if not mapping_path or not os.path.exists(mapping_path):
        return empty_country_mapping_data()
    mapping_df = pd.read_csv(mapping_path, dtype=str).fillna("")
    required = {"Variant", "Canonical", "Component_Type", "Decision"}
    if not required.issubset(mapping_df.columns):
        raise ValueError("Manual affiliation component mapping must contain Variant, Canonical, Component_Type, and Decision columns")
    invalid_types = sorted({
        normalize_space(value)
        for value in mapping_df["Component_Type"]
        if normalize_space(value) and not component_type_value(value)
    })
    if invalid_types:
        raise ValueError("Component_Type accepts only institution, city, country, and exclude: " + "; ".join(invalid_types[:20]))
    grouped = defaultdict(set)
    display_variants = {}
    reference_components = set()
    for _, row in mapping_df.iterrows():
        if decision_value(row.get("Decision", "")) != "accept":
            continue
        variant = clean_affiliation_component(row.get("Variant", ""))
        canonical = clean_affiliation_component(row.get("Canonical", ""))
        component_type = component_type_value(row.get("Component_Type", ""))
        if not variant or not canonical or not component_type:
            continue
        reference_components.add(variant)
        reference_components.add(canonical)
        if component_type != "country":
            continue
        key = component_key(variant)
        grouped[key].add(canonical)
        display_variants[key] = variant
    conflicts = {key: values for key, values in grouped.items() if len(values) > 1}
    if conflicts:
        details = "; ".join(f"{display_variants.get(key, key)}: {sorted(values)}" for key, values in sorted(conflicts.items())[:20])
        raise ValueError(f"Conflicting country mappings detected: {details}")
    replacements = {key: next(iter(values)) for key, values in grouped.items()}
    exact = {key: resolve_country_mapping_chain(key, replacements) for key in replacements}
    rows = [
        {"Variant": display_variants[key], "Canonical": exact[key]}
        for key in sorted(exact)
    ]
    return {
        "exact": exact,
        "rows": rows,
        "reference_components": reference_components,
        "loaded": len(rows)
    }


def observed_affiliation_components(entry):
    seen = set()
    output = []
    for part in entry.get("components", []):
        cleaned = clean_affiliation_component(part)
        key = component_key(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def affiliation_component_record_rank(record):
    component = clean_affiliation_component(record.get("component", ""))
    key = component_key(component)
    tokens = key.split()
    return (
        int(record.get("mentions", 0)),
        len(tokens),
        len(component),
        component.casefold()
    )


def strict_affiliation_component_similarity_metrics(first_part, second_part, threshold):
    first_key = component_key(first_part)
    second_key = component_key(second_part)
    if not first_key or not second_key or first_key == second_key:
        return None
    first_tokens = set(first_key.split())
    second_tokens = set(second_key.split())
    overlap = first_tokens & second_tokens
    union = first_tokens | second_tokens
    jaccard = len(overlap) / max(1, len(union))
    containment = max(len(overlap) / max(1, len(first_tokens)), len(overlap) / max(1, len(second_tokens)))
    token_sort = float(fuzz.token_sort_ratio(first_key, second_key))
    character = float(fuzz.ratio(first_key, second_key))
    if token_sort < threshold and character < 94:
        return None
    reasons = []
    if token_sort >= threshold:
        reasons.append("TOKEN_SORT")
    if character >= 94:
        reasons.append("CHARACTER")
    return {
        "Candidate Reason": "STRICT_" + "_AND_".join(reasons),
        "Token Sort Similarity": round(token_sort, 2),
        "Character Similarity": round(character, 2),
        "Token Jaccard": round(jaccard, 4),
        "Token Containment": round(containment, 4)
    }


def build_affiliation_component_records(entry_lists, mapping):
    records = {}
    mapped = {component_key(value) for value in mapping.get("reference_components", set())}
    for values in entry_lists:
        for entry in values:
            raw_affiliation = clean_affiliation_text(entry.get("raw_affiliation", ""))
            for component in observed_affiliation_components(entry):
                key = component_key(component)
                if not key:
                    continue
                record = records.setdefault(key, {
                    "component": component,
                    "mentions": 0,
                    "raw_affiliations": Counter(),
                    "mapped": key in mapped
                })
                record["mentions"] += 1
                if raw_affiliation:
                    record["raw_affiliations"][raw_affiliation] += 1
    return records


def affiliation_component_raw_example(record):
    values = record.get("raw_affiliations", Counter())
    return values.most_common(1)[0][0] if values else ""


def affiliation_component_similarity_blocks(records):
    blocks = defaultdict(list)
    generic_tokens = {"a", "an", "and", "at", "by", "de", "del", "department", "division", "for", "hospital", "in", "institute", "laboratory", "of", "on", "research", "the", "to", "university", "with"}
    for key in records:
        tokens = key.split()
        block_tokens = []
        if tokens:
            block_tokens.extend(tokens[:2])
            block_tokens.extend(tokens[-2:])
        compact = re.sub(r"[^a-z0-9]", "", key)
        if compact:
            block_tokens.append(compact[:8])
            block_tokens.append(compact[-8:])
            block_tokens.append(f"{compact[:1]}:{len(compact) // 5}")
        for token in set(block_tokens):
            if token and token not in generic_tokens:
                blocks[token].append(key)
    return blocks


def update_affiliation_component_mapping_template(template_path, entry_lists, threshold, mapping):
    columns = affiliation_component_template_columns()
    records = build_affiliation_component_records(entry_lists, mapping)
    if os.path.exists(template_path):
        template = pd.read_csv(template_path, dtype=str).fillna("")
    else:
        template = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in template.columns:
            template[column] = ""
    template = template[columns].copy()
    for index, row in template.iterrows():
        variant = clean_affiliation_component(row.get("Variant", ""))
        if variant and not clean_affiliation_component(row.get("Canonical", "")):
            template.at[index, "Canonical"] = variant
    existing = set()
    for index, row in template.iterrows():
        variant = clean_affiliation_component(row.get("Variant", ""))
        canonical = clean_affiliation_component(row.get("Canonical", ""))
        if variant and canonical:
            if not normalize_space(row.get("Candidate Reason", "")) and component_key(variant) == component_key(canonical):
                template.at[index, "Candidate Reason"] = "OBSERVED_COMPONENT"
            existing.add((component_key(variant), component_key(canonical)))
    rows = []
    for key, record in records.items():
        component = record["component"]
        example = affiliation_component_raw_example(record)
        row_key = (key, key)
        if row_key not in existing:
            existing.add(row_key)
            rows.append({
                "Variant": component,
                "Canonical": component,
                "Component_Type": "",
                "Decision": "unresolved",
                "Mentions": record["mentions"],
                "Canonical Mentions": record["mentions"],
                "Candidate Reason": "OBSERVED_COMPONENT",
                "Token Sort Similarity": "",
                "Character Similarity": "",
                "Token Jaccard": "",
                "Token Containment": "",
                "Variant Full Affiliation": example
            })
    blocks = affiliation_component_similarity_blocks({
        key: record for key, record in records.items()
        if not record.get("mapped") and len(key) > 1
    })
    compared = set()
    candidate_count = 0
    for values in blocks.values():
        unique_values = sorted(set(values))
        for first_key, second_key in combinations(unique_values, 2):
            pair = tuple(sorted((first_key, second_key)))
            if pair in compared:
                continue
            compared.add(pair)
            first = records[first_key]
            second = records[second_key]
            metrics = strict_affiliation_component_similarity_metrics(first["component"], second["component"], threshold)
            if metrics is None:
                continue
            if affiliation_component_record_rank(first) >= affiliation_component_record_rank(second):
                canonical_record = first
                variant_record = second
            else:
                canonical_record = second
                variant_record = first
            variant_key = component_key(variant_record["component"])
            canonical_key = component_key(canonical_record["component"])
            row_key = (variant_key, canonical_key)
            if row_key in existing:
                continue
            existing.add(row_key)
            candidate_count += 1
            rows.append({
                "Variant": variant_record["component"],
                "Canonical": canonical_record["component"],
                "Component_Type": "",
                "Decision": "unresolved",
                "Mentions": variant_record["mentions"],
                "Canonical Mentions": canonical_record["mentions"],
                "Candidate Reason": metrics["Candidate Reason"],
                "Token Sort Similarity": metrics["Token Sort Similarity"],
                "Character Similarity": metrics["Character Similarity"],
                "Token Jaccard": metrics["Token Jaccard"],
                "Token Containment": metrics["Token Containment"],
                "Variant Full Affiliation": affiliation_component_raw_example(variant_record)
            })
    if rows:
        template = pd.concat([template, pd.DataFrame(rows)], ignore_index=True)
    template = template[columns].copy()
    sort_frame = template.assign(
        __canonical_sort=template["Canonical"].apply(component_key),
        __variant_sort=template["Variant"].apply(component_key),
        __reason_sort=template["Candidate Reason"].astype(str),
        __token_sort=pd.to_numeric(template["Token Sort Similarity"], errors="coerce").fillna(-1),
        __character_sort=pd.to_numeric(template["Character Similarity"], errors="coerce").fillna(-1)
    )
    sort_frame = sort_frame.sort_values(
        ["__canonical_sort", "__variant_sort", "__reason_sort", "__token_sort", "__character_sort"],
        ascending=[True, True, True, False, False]
    )
    sort_frame = sort_frame.drop(columns=["__canonical_sort", "__variant_sort", "__reason_sort", "__token_sort", "__character_sort"])
    sort_frame.to_csv(template_path, index=False, encoding="utf-8-sig")
    return candidate_count


def extract_countries_from_entries(entries, country_mapping):
    countries = []
    seen = set()
    for entry in entries:
        for component in entry.get("components", []):
            canonical = country_mapping.get("exact", {}).get(component_key(component), "")
            key = component_key(canonical)
            if canonical and key and key not in seen:
                seen.add(key)
                countries.append(canonical)
    return countries


def country_mapping_validation_rows(entry_lists, country_lists, mapping):
    raw_mentions = Counter()
    for entries in entry_lists:
        for entry in entries:
            for component in entry.get("components", []):
                key = component_key(component)
                if key:
                    raw_mentions[key] += 1
    output_mentions = Counter(
        component_key(country)
        for countries in country_lists
        for country in countries
        if component_key(country)
    )
    rows = []
    for row in mapping.get("rows", []):
        variant = row["Variant"]
        canonical = row["Canonical"]
        variant_count = raw_mentions[component_key(variant)]
        canonical_count = output_mentions[component_key(canonical)]
        if component_key(variant) == component_key(canonical):
            status = "SELF_MAPPING"
        elif variant_count == 0:
            status = "NOT_PRESENT_IN_INPUT"
        elif canonical_count > 0:
            status = "PASS"
        else:
            status = "FAIL"
        rows.append({
            "Variant": variant,
            "Canonical": canonical,
            "Raw_variant_mentions": variant_count,
            "Canonical_output_mentions": canonical_count,
            "Status": status
        })
    return rows


def remove_obsolete_affiliation_outputs(output_dir):
    names = [
        "affiliation_components_observed.csv",
        "affiliation_component_similarity_candidates.csv",
        "affiliation_sites_observed_after_component_mapping.csv",
        "site_similarity_candidates.csv",
        "site_manual_mapping_template.csv",
        "affiliation_component_mapping_validation.csv",
        "site_mapping_validation.csv",
        "primary_immunodeficiency_diseases_affiliation_mapping_audit.csv",
        "inborn_errors_of_immunity_affiliation_mapping_audit.csv"
    ]
    for name in names:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def process_dataset(input_path, dataset_key, output_dir, cache, args, applied_mapping, country_mapping, duplicate_decisions):
    raw = pd.read_csv(input_path, dtype=str, low_memory=False).fillna("")
    raw_count = len(raw)
    raw = ensure_columns(raw)
    raw["__source_row"] = range(2, len(raw) + 2)
    for column in TEXT_COLUMNS:
        raw[column] = raw[column].apply(normalize_space)
    raw["Cited by"] = pd.to_numeric(raw["Cited by"], errors="coerce").fillna(0).astype(int)
    raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce")
    eligible = raw[raw.apply(text_contains_filter_terms, axis=1)].copy()
    after_filter = len(eligible)
    recovered_author_lists = []
    recovered_author_id_lists = []
    recovered_author_statuses = []
    recovered_listed_author_counts = []
    for _, row in eligible.iterrows():
        authors, author_ids, status, listed_count = recover_authors(
            row,
            cache,
            args.crossref_mailto,
            args.crossref_timeout,
            args.crossref_retries,
            args.crossref_sleep,
            args.skip_crossref
        )
        recovered_author_lists.append(authors)
        recovered_author_id_lists.append(author_ids)
        recovered_author_statuses.append(status)
        recovered_listed_author_counts.append(listed_count)
    normalized_author_lists, normalized_author_id_lists = apply_manual_author_mapping(
        recovered_author_lists,
        recovered_author_id_lists,
        applied_mapping
    )
    eligible["_Recovered_author_list"] = recovered_author_lists
    eligible["_Recovered_author_id_list"] = recovered_author_id_lists
    eligible["_Normalized_author_list"] = normalized_author_lists
    eligible["_Normalized_author_id_list"] = normalized_author_id_lists
    eligible["_Author_recovery_source"] = recovered_author_statuses
    eligible["_Listed_author_count"] = recovered_listed_author_counts
    deduplicated, duplicate_audit, manual_review = deduplicate_records(
        eligible,
        dataset_key,
        applied_mapping,
        args.duplicate_title_threshold,
        duplicate_decisions
    )
    after_dedup = len(deduplicated)
    raw_author_lists = [list(values) for values in deduplicated["_Recovered_author_list"]]
    raw_author_id_lists = [list(values) for values in deduplicated["_Recovered_author_id_list"]]
    author_lists = [list(values) for values in deduplicated["_Normalized_author_list"]]
    author_id_lists = [list(values) for values in deduplicated["_Normalized_author_id_list"]]
    author_statuses = deduplicated["_Author_recovery_source"].tolist()
    listed_author_counts = deduplicated["_Listed_author_count"].tolist()
    affiliation_entry_lists = [
        parse_affiliation_entries(row.get("Affiliations", ""))
        for _, row in deduplicated.iterrows()
    ]
    country_lists = [
        extract_countries_from_entries(entries, country_mapping)
        for entries in affiliation_entry_lists
    ]
    deduplicated = deduplicated.drop(columns=[
        "__source_row", "Original CSV Row", "_Recovered_author_list", "_Recovered_author_id_list",
        "_Normalized_author_list", "_Normalized_author_id_list", "_Author_recovery_source", "_Listed_author_count"
    ], errors="ignore")
    deduplicated["Listed_author_count_before_crossref"] = listed_author_counts
    deduplicated["Author_recovery_source"] = author_statuses
    deduplicated["Final_author_count"] = [len(values) for values in author_lists]
    deduplicated = write_list_columns(deduplicated, author_lists, "Author")
    deduplicated = write_list_columns(deduplicated, author_id_lists, "Author_ID")
    deduplicated = write_list_columns(deduplicated, country_lists, "Country")
    output_path = os.path.join(output_dir, f"{dataset_key}_preprocessed.csv")
    deduplicated.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit_path = os.path.join(output_dir, f"{dataset_key}_duplicate_removal_audit.csv")
    pd.DataFrame(duplicate_audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
    unique_countries = len({country for values in country_lists for country in values if country})
    summary = {
        "dataset": dataset_key,
        "input_file": input_path,
        "raw_records": raw_count,
        "records_before_duplicate_removal": after_filter,
        "records_after_duplicate_title_removal": after_dedup,
        "records_after_term_filtering": after_filter,
        "duplicate_titles_removed": after_filter - after_dedup,
        "records_excluded_by_term_filter": raw_count - after_filter,
        "manual_author_replacements_loaded": applied_mapping["loaded"],
        "manual_country_replacements_loaded": country_mapping["loaded"],
        "crossref_records_used": author_statuses.count("crossref"),
        "duplicate_manual_review_candidates": len(manual_review),
        "unique_standardized_countries": unique_countries,
        "records_with_standardized_country": sum(bool(values) for values in country_lists),
        "records_without_standardized_country": sum(not values for values in country_lists),
        "duplicate_removal_audit_file": audit_path,
        "output_file": output_path
    }
    return deduplicated, raw_author_lists, raw_author_id_lists, author_lists, author_id_lists, affiliation_entry_lists, country_lists, summary, manual_review



def mapping_validation_rows(raw_author_lists, raw_author_id_lists, mapped_author_lists, mapped_author_id_lists, mapping):
    raw_mentions = [
        name
        for authors in raw_author_lists
        for name in authors
    ]
    mapped_mentions = [
        name
        for authors in mapped_author_lists
        for name in authors
    ]
    rows = []
    for mapping_row in mapping["rows"]:
        variant = mapping_row["Variant"]
        variant_id = mapping_row["Variant ID"]
        canonical = mapping_row["Canonical"]
        canonical_id = mapping_row["Canonical ID"]
        raw_count = sum(name == variant for name in raw_mentions)
        output_variant_count = sum(name == variant for name in mapped_mentions)
        canonical_output_count = sum(name == canonical for name in mapped_mentions)
        if variant == canonical:
            status = "SELF_MAPPING"
        elif raw_count == 0:
            status = "NOT_PRESENT_IN_INPUT"
        elif output_variant_count == 0 and canonical_output_count > 0:
            status = "PASS"
        else:
            status = "FAIL"
        rows.append({
            "Variant": variant,
            "Variant ID": variant_id,
            "Canonical": canonical,
            "Canonical ID": canonical_id,
            "Raw_variant_mentions": raw_count,
            "Output_variant_mentions": output_variant_count,
            "Canonical_output_mentions": canonical_output_count,
            "Status": status
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", default="Primary Immunodeficiency Diseases.csv")
    parser.add_argument("--iei", default="Inborn errors of immunity.csv")
    parser.add_argument("--out", default="preprocessed_bibliometric_data")
    parser.add_argument("--manual-author-mapping", default="")
    parser.add_argument("--manual-affiliation-component-mapping", default="")
    parser.add_argument("--similarity-threshold", type=int, default=90)
    parser.add_argument("--component-similarity-threshold", "--institution-similarity-threshold", dest="component_similarity_threshold", type=int, default=90)
    parser.add_argument("--duplicate-title-threshold", type=float, default=95.0)
    parser.add_argument("--skip-crossref", action="store_true")
    parser.add_argument("--crossref-mailto", default=os.environ.get("CROSSREF_MAILTO", ""))
    parser.add_argument("--crossref-timeout", type=int, default=20)
    parser.add_argument("--crossref-retries", type=int, default=2)
    parser.add_argument("--crossref-sleep", type=float, default=0.2)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    remove_obsolete_affiliation_outputs(args.out)
    if args.manual_author_mapping and not os.path.exists(args.manual_author_mapping):
        raise FileNotFoundError(f"Manual author mapping file not found: {args.manual_author_mapping}")
    if args.manual_affiliation_component_mapping and not os.path.exists(args.manual_affiliation_component_mapping):
        raise FileNotFoundError(f"Manual affiliation component mapping file not found: {args.manual_affiliation_component_mapping}")
    affiliation_component_mapping_template = os.path.join(args.out, "affiliation_component_manual_mapping_template.csv")
    reference_component_mapping_path = find_reference_affiliation_component_mapping(args.out, args.manual_affiliation_component_mapping)
    create_affiliation_component_mapping_template(affiliation_component_mapping_template, reference_component_mapping_path)
    country_mapping = load_country_mapping(affiliation_component_mapping_template)
    mapping_template = os.path.join(args.out, "author_name_manual_mapping_template.csv")
    reference_mapping_path = find_reference_mapping(args.out, args.manual_author_mapping)
    create_mapping_template(mapping_template, reference_mapping_path)
    if args.manual_author_mapping and os.path.abspath(args.manual_author_mapping) != os.path.abspath(mapping_template):
        seed = pd.read_csv(args.manual_author_mapping, dtype=str).fillna("")
        mapping_columns = ["Variant", "Variant ID", "Canonical", "Canonical ID"]
        for column in mapping_columns:
            if column not in seed.columns:
                seed[column] = ""
        ordered = mapping_columns + [column for column in seed.columns if column not in mapping_columns]
        seed[ordered].to_csv(mapping_template, index=False, encoding="utf-8-sig")
    enrich_author_mapping_template(mapping_template, [args.pid, args.iei])
    reference_mapping_path = mapping_template
    reference_mapping = load_author_mapping(reference_mapping_path)
    applied_mapping = load_author_mapping(mapping_template) if args.manual_author_mapping else empty_mapping_data()
    manual_review_path = os.path.join(args.out, "possible_duplicate_manual_review.csv")
    duplicate_decisions = load_duplicate_decisions(manual_review_path)
    cache_path = os.path.join(args.out, "crossref_author_cache.json")
    cache = load_crossref_cache(cache_path)
    pid_df, pid_raw_authors, pid_raw_author_ids, pid_authors, pid_author_ids, pid_affiliations, pid_countries, pid_summary, pid_manual = process_dataset(
        args.pid,
        "primary_immunodeficiency_diseases",
        args.out,
        cache,
        args,
        applied_mapping,
        country_mapping,
        duplicate_decisions
    )
    save_crossref_cache(cache_path, cache)
    iei_df, iei_raw_authors, iei_raw_author_ids, iei_authors, iei_author_ids, iei_affiliations, iei_countries, iei_summary, iei_manual = process_dataset(
        args.iei,
        "inborn_errors_of_immunity",
        args.out,
        cache,
        args,
        applied_mapping,
        country_mapping,
        duplicate_decisions
    )
    save_crossref_cache(cache_path, cache)
    all_raw_author_lists = pid_raw_authors + iei_raw_authors
    all_raw_author_id_lists = pid_raw_author_ids + iei_raw_author_ids
    all_mapped_author_lists = pid_authors + iei_authors
    all_mapped_author_id_lists = pid_author_ids + iei_author_ids
    all_affiliation_entry_lists = pid_affiliations + iei_affiliations
    all_country_lists = pid_countries + iei_countries
    component_candidate_count = update_affiliation_component_mapping_template(
        affiliation_component_mapping_template,
        all_affiliation_entry_lists,
        args.component_similarity_threshold,
        country_mapping
    )
    similarity_path = os.path.join(args.out, "author_name_similarity_candidates.csv")
    candidate_count = generate_similarity_candidates(
        all_mapped_author_lists,
        all_mapped_author_id_lists,
        similarity_path,
        args.similarity_threshold,
        reference_mapping
    )
    validation_path = os.path.join(args.out, "author_mapping_validation.csv")
    validation_rows = mapping_validation_rows(all_raw_author_lists, all_raw_author_id_lists, all_mapped_author_lists, all_mapped_author_id_lists, applied_mapping)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False, encoding="utf-8-sig")
    country_validation_path = os.path.join(args.out, "country_mapping_validation.csv")
    country_validation = country_mapping_validation_rows(all_affiliation_entry_lists, all_country_lists, country_mapping)
    pd.DataFrame(country_validation).to_csv(country_validation_path, index=False, encoding="utf-8-sig")
    combined = pd.concat([
        pid_df.assign(Search_Category="Primary Immunodeficiency Diseases"),
        iei_df.assign(Search_Category="Inborn Errors of Immunity")
    ], ignore_index=True)
    combined["__source_row"] = range(2, len(combined) + 2)
    combined_deduplicated, combined_audit, combined_manual = deduplicate_records(
        combined,
        "combined_pid_iei",
        applied_mapping,
        args.duplicate_title_threshold,
        duplicate_decisions
    )
    combined_deduplicated = combined_deduplicated.drop(columns=["__source_row"], errors="ignore")
    combined_path = os.path.join(args.out, "combined_pid_iei_preprocessed_deduplicated_by_title_year.csv")
    combined_deduplicated.to_csv(combined_path, index=False, encoding="utf-8-sig")
    combined_audit_path = os.path.join(args.out, "combined_pid_iei_duplicate_removal_audit.csv")
    pd.DataFrame(combined_audit).to_csv(combined_audit_path, index=False, encoding="utf-8-sig")
    manual_review = pid_manual + iei_manual + combined_manual
    if manual_review:
        manual_df = pd.DataFrame(manual_review)
        manual_df = manual_df.drop_duplicates(subset=["Pair_ID"], keep="first")
    else:
        manual_df = pd.DataFrame(columns=["Pair_ID", "Dataset", "Decision"])
    manual_df.to_csv(manual_review_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([pid_summary, iei_summary])
    summary["author_similarity_candidates_generated"] = candidate_count
    summary["affiliation_component_manual_template_similarity_rows_added"] = component_candidate_count
    summary["author_mapping_validation_file"] = validation_path
    summary["country_mapping_validation_file"] = country_validation_path
    summary["author_similarity_candidates_file"] = similarity_path
    summary["affiliation_component_manual_mapping_template_file"] = affiliation_component_mapping_template
    summary["possible_duplicate_manual_review_file"] = manual_review_path
    summary["combined_deduplicated_title_year_file"] = combined_path
    summary["combined_duplicate_removal_audit_file"] = combined_audit_path
    summary.to_csv(os.path.join(args.out, "preprocessing_summary.csv"), index=False, encoding="utf-8-sig")
    failed_mappings = sum(1 for row in validation_rows if row["Status"] == "FAIL")
    failed_country_mappings = sum(1 for row in country_validation if row["Status"] == "FAIL")
    print("Preprocessing complete")
    print(f"Author mapping validation failures: {failed_mappings}")
    print(f"Country mapping validation failures: {failed_country_mappings}")
    print(f"Possible duplicate pairs requiring manual review: {len(manual_df)}")
    print(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
