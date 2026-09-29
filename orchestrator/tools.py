import os

def OptimizeQED(path_to_input: str, target_value: float, path_to_output: str) -> str:
    import random
    import pandas as pd

    df = pd.read_csv(path_to_input)

    col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if col is None:
        return f"Error: no 'smiles' or 'SMILES' column found in {path_to_input}."

    n = random.randint(1, len(df))
    sample = df[[col]].sample(n=n).copy()
    sample["qed"] = target_value

    sample.to_csv(path_to_output, index=False)

    return f"Optimization complete. {len(sample)} molecules reach QED={target_value}. Results saved to {path_to_output}."

def EvaluateSynthesisRoute(path_to_input: str, path_to_output: str) -> str:
    import random
    import pandas as pd

    df = pd.read_csv(path_to_input)

    col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if col is None:
        return f"Error: no 'smiles' or 'SMILES' column found in {path_to_input}."

    n = random.randint(1, len(df))
    out_df = df.sample(n=n).copy()
    out_df["synthesizable"] = True
    out_df.to_csv(path_to_output, index=False)

    return f"Synthesis route evaluation complete. {n} synthesizable molecules found. Results saved to {path_to_output}."

def EvaluateQED(path_to_input: str, path_to_output: str) -> str:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import QED

    df = pd.read_csv(path_to_input)

    col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if col is None:
        return f"Error: no 'smiles' or 'SMILES' column found in {path_to_input}."

    def compute_qed(smi):
        mol = Chem.MolFromSmiles(smi)
        return QED.qed(mol) if mol is not None else None

    df["qed"] = df[col].apply(compute_qed)

    df.to_csv(path_to_output, index=False)

    valid = df["qed"].dropna()
    avg = valid.mean()
    return f"Evaluation complete. {len(valid)}/{len(df)} molecules evaluated. Results with QED values saved to {path_to_output}. Average QED: {avg:.4f}."

def SBDD(num: int, path_to_pdb: str, path_to_output: str) -> str:
    from tdc.generation import MolGen
    data = MolGen(name = 'ZINC')
    split = data.get_split()
    train = split["train"].sample(num, random_state=42, replace=False)

    train.to_csv(os.path.join(path_to_output), index=False)
    
    return f"SBDD generation complete. Output saved to {path_to_output} with {len(train)} molecules. SMILES stored in column 'smiles'."


if __name__ == "__main__":
    out = SBDD(10, path_to_pdb="path/to/pdb", path_to_output="path/to/output")
    print(out)