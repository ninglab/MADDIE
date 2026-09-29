from pathlib import Path
import csv
from difflib import get_close_matches
import re

from .errors import LeadOptimizerRequestError


SMILES_COLUMNS = (
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical smiles",
    "canonicalsmiles",
    "isomericsmiles",
    "molecularsmiles",
    "smilesstring",
    "compoundsmiles",
    "structure",
    "molecularstructure",
    "compoundstructure",
    "chemicalstructure",
    "molecule",
    "mol",
    "mols",
)
ID_COLUMNS = (
    "id",
    "name",
    "moleculeid",
    "molecule_id",
    "molecule id",
    "moleculename",
    "molid",
    "mol_id",
    "mol id",
    "compoundid",
    "compound_id",
    "compound id",
    "compoundname",
    "label",
    "identifier",
    "title",
)


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def find_csv_in_prompt(prompt: str) -> str | None:
    """Find the first CSV filename mentioned in the prompt."""
    match = re.search(r"([A-Za-z0-9_\-./]+\.csv)", prompt, flags=re.IGNORECASE)
    return match.group(1) if match else None


def find_num_outputs_in_prompt(prompt: str) -> int | None:
    """Find counts such as '5 valid molecules' without matching thresholds."""
    match = re.search(
        r"\b(\d+)\s+(?:(?:valid|unique|different)\s+)*"
        r"(?:molecules?|candidates?|outputs?|variants?)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def extract_optimization_adjustment(prompt: str) -> str:
    """Remove CSV/count orchestration while preserving the chemistry request."""
    csv_pattern = r"[A-Za-z0-9_\-./]+\.csv"
    orchestration_patterns = [
        rf"^\s*(?:please\s+)?(?:generate|return|produce|create)\s+\d+\s+"
        rf"(?:(?:valid|unique|different)\s+)*(?:molecules|candidates|outputs|variants)\s+"
        rf"(?:from|using)\s+{csv_pattern}\s*(?:by|for|to)?\s*",
        r"^\s*(?:please\s+)?(?:generate|return|produce|create)\s+\d+\s+"
        r"(?:(?:valid|unique|different)\s+)*(?:molecules|candidates|outputs|variants)\s*"
        r"(?:by|for|to)?\s*",
        rf"^\s*(?:please\s+)?optimize\s+(?:the\s+)?molecules?\s+"
        rf"(?:in|from|using)\s+{csv_pattern}\s*(?:by|for|to)?\s*",
        rf"^\s*(?:please\s+)?optimize\s+{csv_pattern}\s*(?:by|for|to)?\s*",
    ]

    adjustment = prompt.strip()
    for pattern in orchestration_patterns:
        cleaned = re.sub(pattern, "", adjustment, count=1, flags=re.IGNORECASE)
        if cleaned != adjustment:
            adjustment = cleaned
            break

    adjustment = re.sub(
        rf"\s+(?:in|from|using)\s+{csv_pattern}\b",
        "",
        adjustment,
        flags=re.IGNORECASE,
    )
    adjustment = adjustment.strip(" \t\n,.;:")
    adjustment = re.sub(r"^increasing\b", "increase", adjustment, flags=re.IGNORECASE)
    adjustment = re.sub(r"^decreasing\b", "decrease", adjustment, flags=re.IGNORECASE)
    adjustment = re.sub(
        r"<SMILES>.*?</SMILES>",
        "",
        adjustment,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    adjustment = re.sub(r"</?SMILES>", "", adjustment, flags=re.IGNORECASE).strip()
    adjustment = tag_thresholds(adjustment)
    return adjustment or prompt.strip()


def tag_thresholds(text: str) -> str:
    """Tag scalar thresholds without rewriting numeric ranges."""
    phrases = (
        "no more than",
        "no less than",
        "no greater than",
        "at or below",
        "at or above",
        "at least",
        "at most",
        "threshold of",
        "threshold",
        "above",
        "below",
        "under",
        "over",
        "up to",
        "at",
        "to",
    )
    phrase_pattern = "|".join(re.escape(phrase) for phrase in phrases)
    pattern = re.compile(
        rf"(\b(?:{phrase_pattern})\s*)({NUMBER_PATTERN})\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        phrase = match.group(1)
        if phrase.strip().lower() == "to":
            before = text[:match.start()]
            if re.search(rf"\b{NUMBER_PATTERN}\s*$", before):
                return match.group(0)
        return f"{phrase}<THRESHOLD> {match.group(2)} </THRESHOLD>"

    return pattern.sub(replace, text)


def resolve_input_csv(
    env: Path,
    requested_file: str | None,
) -> tuple[Path, str | None]:
    """Resolve an input CSV and report when a requested name was repaired."""
    input_dir = env / "input"
    candidates = sorted({
        *input_dir.rglob("*.csv"),
        *env.glob("*.csv"),
    })
    requested = Path(requested_file).name if requested_file else "(none)"
    requested_path_text = str(Path(requested_file).expanduser()) if requested_file else None
    search_paths = [str(env), str(input_dir)]
    if not candidates:
        raise LeadOptimizerRequestError(
            f"No input CSV was found for '{requested}'. Add a CSV file to "
            f"{env} or {input_dir} and try again.",
            status="needs_input",
            metadata={
                "error_category": "input_missing",
                "requested_input": requested_path_text,
                "searched_paths": search_paths,
            },
        )

    if requested_file:
        requested_path = Path(requested_file).expanduser()
        direct_matches = []
        if requested_path.is_absolute() and requested_path.exists():
            direct_matches.append(requested_path)
        direct_matches.extend(
            path
            for path in (env / requested_file, input_dir / requested_file)
            if path.exists()
        )
        direct_matches = list(dict.fromkeys(direct_matches))
        if len(direct_matches) == 1:
            return direct_matches[0], None

        requested_name = requested_path.name
        exact = [
            path for path in candidates
            if path.name.lower() == requested_name.lower()
        ]
        if len(exact) == 1:
            return exact[0], None

        close_names = get_close_matches(
            requested_name.lower(),
            [path.name.lower() for path in candidates],
            n=2,
            cutoff=0.75,
        )
        if len(close_names) == 1:
            match = next(
                path for path in candidates
                if path.name.lower() == close_names[0]
            )
            return (
                match,
                f"Used '{match.name}' because '{requested_name}' did not resolve.",
            )

    if len(candidates) == 1:
        requested = Path(requested_file).name if requested_file else "no filename"
        return (
            candidates[0],
            f"Used '{candidates[0].name}' because {requested!r} did not resolve.",
        )

    names = ", ".join(path.name for path in candidates)
    raise LeadOptimizerRequestError(
        "The input CSV was missing or not found, and multiple CSV files are "
        f"available: {names}. Name the intended file in the request.",
        status="needs_input",
        metadata={
            "error_category": "input_missing",
            "requested_input": requested_path_text,
            "searched_paths": search_paths,
            "available_csvs": [str(path) for path in candidates],
        },
    )


def normalize_column_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").strip().lower())


def looks_like_smiles(value: str | None) -> bool:
    """Best-effort molecule-column detector for loosely formatted CSVs."""
    text = (value or "").strip()
    if not text or len(text) > 500 or any(char.isspace() for char in text):
        return False
    if not re.search(r"[A-Za-z\[\]\(\)=#@+\-0-9]", text):
        return False

    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.error")
        try:
            return Chem.MolFromSmiles(text) is not None
        finally:
            RDLogger.EnableLog("rdApp.error")
    except Exception:
        return bool(
            re.search(r"[BCNOFPSIbcnops][A-Za-z]?", text)
            and (
                len(text) <= 3
                or re.search(r"[\[\]\(\)=#0-9]", text)
            )
        )


def _infer_smiles_column(
    fieldnames: list[str],
    rows: list[dict],
) -> tuple[str | None, str | None]:
    candidates = []
    for fieldname in fieldnames:
        values = [
            (row.get(fieldname) or "").strip()
            for row in rows[:25]
            if (row.get(fieldname) or "").strip()
        ]
        if not values:
            continue
        hits = sum(1 for value in values if looks_like_smiles(value))
        if hits and hits / len(values) >= 0.6:
            candidates.append((hits / len(values), hits, fieldname))

    if len(candidates) != 1:
        return None, None

    fieldname = candidates[0][2]
    return (
        fieldname,
        f"Inferred '{fieldname}' as the SMILES column from molecule-like values.",
    )


def read_smiles_csv(csv_path: Path) -> list[dict]:
    """Read a molecule CSV, preserving the original list-returning API."""
    rows, _warnings, _skipped_rows = normalize_smiles_csv(csv_path)
    return rows


def normalize_smiles_csv(csv_path: Path) -> tuple[list[dict], list[str], int]:
    """Read and repair a molecule CSV into input_id/smiles records."""
    with csv_path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise LeadOptimizerRequestError(
                f"Input CSV has no header row: {csv_path}. Add a recognized "
                "molecule column such as 'smiles' or 'structure'.",
                status="needs_input",
                metadata={
                    "error_category": "input_wrong_type",
                    "requested_input": str(csv_path),
                },
            )
        raw_rows = list(reader)

        columns = {
            normalize_column_name(name): name
            for name in reader.fieldnames
            if name is not None
        }
        smiles_column = next(
            (columns[name] for name in SMILES_COLUMNS if name in columns),
            None,
        )
        inferred_warning = None
        if smiles_column is None:
            header_smiles = [
                index
                for index, name in enumerate(reader.fieldnames)
                if looks_like_smiles(name)
            ]
            if len(header_smiles) == 1:
                handle.seek(0)
                raw_rows = [
                    {
                        "__smiles__": (
                            row[header_smiles[0]]
                            if len(row) > header_smiles[0]
                            else ""
                        )
                    }
                    for row in csv.reader(handle, dialect=dialect)
                ]
                smiles_column = "__smiles__"
                inferred_warning = (
                    "Detected a headerless CSV and used the molecule-like column "
                    f"{header_smiles[0] + 1} as the SMILES column."
                )
            else:
                smiles_column, inferred_warning = _infer_smiles_column(
                    [name for name in reader.fieldnames if name is not None],
                    raw_rows,
                )
        if smiles_column is None:
            available = ", ".join(name.strip() for name in reader.fieldnames if name)
            raise LeadOptimizerRequestError(
                "Input CSV needs a molecule column such as smiles, "
                f"canonical_smiles, or molecule. Found: {available or '(none)'}.",
                status="needs_input",
                metadata={
                    "error_category": "input_wrong_type",
                    "requested_input": str(csv_path),
                    "available_columns": [
                        name.strip() for name in reader.fieldnames if name
                    ],
                },
            )
        id_column = next(
            (columns[name] for name in ID_COLUMNS if name in columns),
            None,
        )

        rows = []
        warnings = []
        if inferred_warning:
            warnings.append(inferred_warning)
        skipped_rows = 0
        used_ids = set()
        for index, row in enumerate(raw_rows, start=1):
            smiles = (row.get(smiles_column) or "").strip()
            if not smiles:
                skipped_rows += 1
                continue

            input_id = (row.get(id_column) or "").strip() if id_column else ""
            input_id = input_id or f"csv_{index:03d}"
            base_id = input_id
            suffix = 2
            while input_id in used_ids:
                input_id = f"{base_id}_{suffix}"
                suffix += 1
            if input_id != base_id:
                warnings.append(
                    f"Renamed duplicate id '{base_id}' to '{input_id}'."
                )
            used_ids.add(input_id)
            rows.append({"input_id": input_id, "smiles": smiles})

    if not rows:
        raise LeadOptimizerRequestError(
            f"Input CSV contains no usable molecule rows: {csv_path}.",
            status="needs_input",
        )
    if smiles_column != "smiles":
        warnings.insert(0, f"Used '{smiles_column}' as the SMILES column.")
    if id_column is None:
        warnings.append("Generated molecule IDs because no ID column was present.")
    if skipped_rows:
        warnings.append(f"Skipped {skipped_rows} row(s) with blank SMILES.")
    return rows, warnings, skipped_rows


def write_optimized_csv(output_path: Path, rows: list[dict]) -> None:
    """Write the standardized optimizer output CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_id",
        "input_smiles",
        "optimized_smiles",
        "output_id",
        "adjustment",
        "tool",
        "property",
        "input_property_value",
        "optimized_property_value",
    ]
    dict_rows = [row.to_dict() if hasattr(row, "to_dict") else row for row in rows]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)
