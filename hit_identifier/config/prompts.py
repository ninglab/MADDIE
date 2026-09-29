import json


PARSER_SYSTEM_PROMPT = "You are a strict JSON parser. Return JSON only."

INTENT_CLASSIFIER_SYSTEM_PROMPT = "You are a strict intent classifier. Return JSON only."

ALLOWED_ACTION_TYPES = {"SBDD_GENERATE", "LBDD_GENERATE", "VIRTUAL_SCREEN"}

ACTION_ALIASES = {
    "SBDD": "SBDD_GENERATE",
    "SBDD_GENERATE": "SBDD_GENERATE",
    "STRUCTURE_BASED": "SBDD_GENERATE",
    "STRUCTURE_BASED_GENERATION": "SBDD_GENERATE",
    "STRUCTURE-BASED GENERATION": "SBDD_GENERATE",
    "STRUCTURE BASED GENERATION": "SBDD_GENERATE",
    "LBDD": "LBDD_GENERATE",
    "LBDD_GENERATE": "LBDD_GENERATE",
    "LIGAND_BASED": "LBDD_GENERATE",
    "LIGAND_BASED_GENERATION": "LBDD_GENERATE",
    "LIGAND-BASED GENERATION": "LBDD_GENERATE",
    "LIGAND BASED GENERATION": "LBDD_GENERATE",
    "VS": "VIRTUAL_SCREEN",
    "VIRTUAL_SCREEN": "VIRTUAL_SCREEN",
    "VIRTUAL SCREEN": "VIRTUAL_SCREEN",
    "VIRTUAL SCREENING": "VIRTUAL_SCREEN",
    "SCREENING": "VIRTUAL_SCREEN",
    "BOLTZ": "VIRTUAL_SCREEN",
    "BOLTZ2": "VIRTUAL_SCREEN",
}


def build_intent_classifier_prompt(user_input: str) -> str:
    return (
        "Classify the user input as either 'generate' or 'chat'.\n\n"
        "'generate': The user wants to start a molecule generation or virtual screening run. "
        "Key signals: mentions a protein target to work on, wants molecules found/generated/screened, "
        "specifies criteria (Vina, QED, SA, number of hits, rounds). "
        "Action-oriented verbs like find, generate, run, screen, identify applied to molecules.\n\n"
        "'chat': The user is asking a question or requesting information. "
        "Key signals: asking about capabilities, configuration, status, results of a past run, "
        "or what files/inputs are needed. Question words (what, how, can you, tell me, show me). "
        "No intent to start a new generation run.\n\n"
        "When in doubt, prefer 'chat'.\n\n"
        "Return exactly one JSON object:\n"
        "{\"intent\": \"generate\"}\n"
        "or\n"
        "{\"intent\": \"chat\"}\n\n"
        f"User input:\n{user_input}"
    )


def build_user_prompt_parser_prompt(user_request: str) -> str:
    return (
        "You are a strict parser for a hit-identification workflow.\n\n"
        "Extract structured parameters from the user request. Return exactly one JSON object and nothing else.\n\n"
        "Available action types:\n"
        "- SBDD_GENERATE: SBDD, structure-based generation, structure-based design\n"
        "- LBDD_GENERATE: LBDD, ligand-based generation, ligand-based design\n"
        "- VIRTUAL_SCREEN: virtual screening, screening, VS, Boltz, Boltz2\n\n"
        "Defaults:\n"
        "- pdb_path: \"\"\n"
        "- sdf_path: \"\"\n"
        "- drug_sheet_path: \"\"\n"
        "- target: \"\"\n"
        "- min_pool_size: 10\n"
        "- max_iter: 10\n"
        "- target_vina: null\n"
        "- target_qed: null\n"
        "- target_sa: null\n"
        "- allowed_action_types: []\n\n"
        "Rules:\n"
        "- If the user provides a file path ending in .pdb, extract it to pdb_path. This is required for any generation or screening run.\n"
        "- If the user provides a file path ending in .sdf, extract it to sdf_path. This is an optional reference ligand.\n"
        "- If the user provides a file path to a drug sheet (ending in .csv, .tsv, or .xlsx, or described as a drug sheet or compound library), extract it to drug_sheet_path.\n"
        "- target is a short human-readable name for the target (e.g. 'EGFR'). If not given explicitly, leave it empty.\n"
        "- If the user specifies at least N molecules/hits/compounds, set min_pool_size to N.\n"
        "- If the user specifies within N rounds/iterations/actions/steps, set max_iter to N.\n"
        "- For Vina, docking score, or binding score thresholds, lower is better.\n"
        "- For QED thresholds, higher is better.\n"
        "- For SA or synthetic accessibility thresholds, lower is better.\n"
        "- If the user restricts methods/actions/strategies, fill allowed_action_types using the internal action type names.\n"
        "- If the user says use any available strategy, any method, or does not restrict methods, keep allowed_action_types as [].\n"
        "- If the user excludes a method, allowed_action_types should contain all other action types.\n"
        "- Do not invent thresholds, paths, or action restrictions.\n"
        "- If a field is not specified, use the default.\n\n"
        "JSON schema:\n"
        "{\n"
        "  \"pdb_path\": \"\",\n"
        "  \"sdf_path\": \"\",\n"
        "  \"drug_sheet_path\": \"\",\n"
        "  \"target\": \"\",\n"
        "  \"min_pool_size\": 10,\n"
        "  \"max_iter\": 10,\n"
        "  \"target_vina\": null,\n"
        "  \"target_qed\": null,\n"
        "  \"target_sa\": null,\n"
        "  \"allowed_action_types\": [],\n"
        "  \"user_task\": \"\"\n"
        "}\n\n"
        f"User request:\n{user_request}"
    )


def build_chat_prompt(context: dict, user_question: str) -> str:
    return (
        "You are the communication interface for Hit Identifier.\n"
        "Answer the user's question using only the provided context.\n"
        "Targets are not pre-configured; the user specifies file paths (pdb_path, sdf_path, drug_sheet_path) directly in each prompt.\n"
        "Use method_requirements when answering what input files or prerequisites are needed.\n"
        "Do not describe LBDD seed requirements as requirements for every molecule-generation method.\n"
        "Do not describe virtual-screening drug-sheet requirements as requirements for molecule generation.\n"
        "When the user asks generally about generating molecules, answer by method: SBDD and LBDD separately.\n"
        "If the context does not contain enough information, say what is unknown and where to check.\n"
        "Do not invent capabilities, metrics, file paths, or results.\n"
        "Keep the answer concise and practical.\n\n"
        f"Context:\n{json.dumps(context, indent=2)}\n\n"
        f"User question:\n{user_question.strip()}"
    )


def build_generation_summary_prompt(context: dict) -> str:
    return (
        "You are the communication interface for Hit Identifier.\n"
        "Write a concise final summary of the completed generation run using only the provided context.\n"
        "Always include the exact run_dir path from last_result, for example log/<run_id>_<target>.\n"
        "Always include the exact molecule_library path and successful_results path when available.\n"
        "Include what was attempted, whether the requested success criteria were met, where evaluated molecules were saved, "
        "where successful molecules were saved, and a practical conclusion.\n"
        "Report the statistics of successful molecules from successful_results.stats: count, vina min/max/median, qed min/max/median, sa min/max/median.\n"
        "If success_criteria.selection_mode is best_binding, describe successful_results as the selected best-binding molecules.\n"
        "Do not report per-round generation metrics. Do not invent molecule counts, metrics, file paths, or actions.\n\n"
        f"Context:\n{json.dumps(context, indent=2)}"
    )


def build_error_conclusion_prompt(diagnosis: dict) -> str:
    if diagnosis.get("error_type") == "unknown":
        return (
            "You are the communication interface for Hit Identifier.\n"
            "A generation run failed with an unrecognized error. Read the exception and the traceback "
            "in evidence.detail and write at most 3 short sentences:\n"
            "1. The most likely cause, inferred from the exception type, message, and traceback.\n"
            "2. The specific code location or step it failed at, if the traceback shows one.\n"
            "3. A concrete next step to investigate or fix it.\n"
            "You MAY interpret the traceback. Be honest about uncertainty (e.g. 'likely'). "
            "When you refer to a file, write its full path. "
            "Do not invent file paths or values that are not in the traceback.\n\n"
            f"Diagnosis:\n{json.dumps(diagnosis, indent=2)}"
        )
    return (
        "You are the communication interface for Hit Identifier.\n"
        "A generation run failed. Using ONLY the diagnosis, write at most 2 short sentences:\n"
        "1. The cause: target (if known) + which file/condition + why, in plain terms.\n"
        "2. The fix: the single most appropriate item from 'suggestions'.\n"
        "Be terse. No preamble, no 'no molecules were generated', no alternatives, no restating the diagnosis. "
        "When you refer to a file, write its full path as it appears in the diagnosis, not just the filename. "
        "Do not invent paths, metrics, or causes.\n\n"
        f"Diagnosis:\n{json.dumps(diagnosis, indent=2)}"
    )
