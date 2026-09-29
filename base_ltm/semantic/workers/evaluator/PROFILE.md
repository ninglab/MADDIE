Critically reviews molecules. 
This is the gate. It is the only source of docking scores (Vina) and ADMET pass/fail
(mutagenicity, carcinogenicity, hERG, etc.). A generated molecule is NOT "accepted" until the
molecules meet every required constraint. Send every new batch here before
counting it toward the goal.
# NOTES: 
Can evaluate physicochemical properties, molecular filters, docking and some ADMET properties. 
RDKit: QED, SAScore, MW, LogP, TPSA, etc. 
Molecular filters: PAINS, Brenk, Lipinski Ro5, Lilly MedChem Rules. 
Molecular docking. 
ADMET: hERG, DILI, ClinTox, Mutagenicity, Carcinogens. 
Only works with files, not with raw SMILES.
The Evaluator only sees its own subdirectory (`evaluator/`) — it cannot find files from other agents' directories. Always provide absolute paths for both the molecule CSV and the protein PDB file in your delegation message. For example: "evaluate /abs/path/to/env/evaluator/hits.csv using /abs/path/to/EGFR.pdb". If the molecule file is not already in the evaluator's directory, use PROCESS to copy it there first.
Output CSV column names (use these exactly when filtering in PROCESS):
- Physicochemical: physchem_QED, physchem_SAScore, physchem_MolecularWeight, physchem_LogP, physchem_TPSA, physchem_HBD, physchem_HBA
- Filters (boolean, True=pass): filter_PAINS, filter_Brenk, filter_RDKIT, filter_LILLY
- ADMET (float 0–1, lower=safer): admet_hERG, admet_DILI, admet_ClinTox, admet_Mutagenicity, admet_Carcinogens
- Docking: docking_kcal_mol (float, more negative=better)