Takes one or more hits or early leads and improves them. Use when a starting series exists and the goal is to advance it toward a candidate profile.
# WHEN TO USE
Use LeadOptimizer after Evaluator reveals molecules with acceptable docking scores but failing ADMET thresholds. LeadOptimizer ACTIVELY OPTIMIZES (not just measures) ADMET toxicity scores. Concrete triggers:
- admet_hERG above threshold → delegate with "decrease hERG"
- admet_DILI above threshold → delegate with "decrease DILI"
- admet_ClinTox above threshold → delegate with "decrease ClinTox"
- admet_Mutagenicity above threshold → delegate with "decrease Mutagenicity"
- admet_Carcinogens above threshold → delegate with "decrease Carcinogenicity"
- physchem_QED below threshold → delegate with "increase QED"
Do NOT mark these as "only measured" — LeadOptimizer can improve them.
# NOTES
Properties it can optimize: QED, SAScore, LogP, hERG, DILI, ClinTox, Mutagenicity, Carcinogenicity.
Only works with files, not with raw SMILES.
Can do multi-property optimization, at most three properties at once.
Make sure the file is CSV with column name 'smiles' for SMILES string.
Your file must be located in the environment specific for lead optimizer: the `lead_optimizer/` subdirectory of your env. ALWAYS run a PROCESS step to copy your file there BEFORE delegating (e.g. `os.makedirs('lead_optimizer', exist_ok=True); shutil.copy('your_file.csv', 'lead_optimizer/your_file.csv')`). Do NOT delegate without doing this first — the agent will fail if the file is not already in place.
Output file will likely only contain SMILES, you need to reevaluate the molecules. If you re-use the output as input for another round, rename the `optimized_smiles` column to `smiles` first — LeadOptimizer only accepts `smiles`, `canonical_smiles`, or `molecule` as column names.
When delegating, use explicit directional language for each property: say "decrease hERG", "decrease DILI", "increase QED" — not vague phrases like "minimize toxicity risk" or "reduce risk". ADMET properties (hERG, DILI, ClinTox, Mutagenicity, Carcinogenicity) should always use "decrease".