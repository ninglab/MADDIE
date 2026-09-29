Takes a CSV of molecules and finds synthesis routes or synthesizable analogs for them.
# WHEN TO USE
Use Synthesizer when ANY condition is met:
1. `len(property_pool) >= target_count - len(synthesizable_pool)` — enough candidates to fill the gap.
2. `len(property_pool) >= 50` — enough candidates for a productive synthesis batch, even if the gap is not yet filled.
3. Remaining iterations < 8 — budget is running low.
Run it once on a batch, not after every Evaluator round.
# NOTES
Only works with files, not with raw SMILES.
Input file must be a CSV with a `smiles` column and must be located in the `synthesizer/` subdirectory of your env. ALWAYS run a PROCESS step to copy the file there BEFORE delegating (e.g. `import shutil, os; os.makedirs(os.path.join(ENV_PATH, 'synthesizer'), exist_ok=True); shutil.copy(os.path.join(ENV_PATH, 'property_pool.csv'), os.path.join(ENV_PATH, 'synthesizer', 'property_pool.csv'))`). Do NOT delegate without doing this first.
Output file is named `{input_stem}_meea.csv` (e.g. input `property_pool.csv` → output `property_pool_meea.csv`) written to the `synthesizer/` subdirectory.
Give a time budget of 30 minutes in the prompt (e.g. "within 30 minutes").
Output CSV key columns:
- `smiles`: the original input molecule SMILES (same value as in your input file)
- `synthesizable_decision`: YES if MEEA found a confirmed synthesis route, NO otherwise
- `source`: MEEA (direct synthesis route confirmed), PrexSyn (PrexSyn backup confirmed), or empty string (NO molecules)
- `analog_smiles_1` … `analog_smiles_5`: structurally similar candidates predicted by PrexSyn for NO molecules; empty string if none found
# AFTER RECEIVING RESULTS
Run a PROCESS step to build synthesizable_pool.csv from two streams:
1. YES stream: rows where synthesizable_decision=YES → join with env/property_pool.csv on `smiles` to recover ADMET/docking scores (property_pool uses `generation_source`, not `source`, so no column name conflict) → use the `source` column from the MEEA output (either "MEEA" or "PrexSyn") → append to env/synthesizable_pool.csv (deduplicate by smiles).
2. Analog stream: rows where synthesizable_decision=NO and at least one analog_smiles_* is non-empty → unpivot (one row per non-empty analog: smiles=analog_smiles_N) → deduplicate against env/property_pool.csv and env/synthesizable_pool.csv → write to evaluator/analogs_to_evaluate.csv → delegate to Evaluator → filter analogs passing ALL user constraints → set source="Analog" → append to env/synthesizable_pool.csv (deduplicate by smiles).
After both streams complete: clear env/property_pool.csv (delete or empty it).
If synthesizable_pool still has fewer molecules than needed, go back to HI to generate more molecules, rebuild env/property_pool.csv, and repeat from the trigger check.
FINAL.csv must be written from env/synthesizable_pool.csv (first target_count rows). Include the `source` column alongside all ADMET/docking columns. Deduplicate by smiles.
