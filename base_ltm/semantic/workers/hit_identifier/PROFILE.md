Generates candidate molecules against a given target. Use to seed a campaign and to re-explore whenever you want scaffolds your current pool doesn't cover or an existing series has stalled. This is a reusable generation source, not a one-time seeding step.
# NOTES:
When delegating, your input must follow the format below.
SBDD and LBDD are different search strategies that sample chemical space differently; using both, or alternating them across rounds, broadens coverage.
# INPUT FORMAT:
Find at least A potential hits for B within C rounds. Action list: D.
You need to specify A, B, C, D respectively.
A is a number of hits; you can specify any number, from 10s to 100s.
B must be a full path to a PDB file.
C is a number of rounds. 3 to 10 is a good range; you can specify more in later steps.
D is one of ('SBDD', 'LBDD', 'SBDD, LBDD').
# INPUT EXAMPLE:
Find at least 100 potential hits for /path/to/EGFR.pdb within 5 rounds. Action list: SBDD, LBDD.
Find at least 50 potential hits for /path/to/CDK5.pdb within 3 rounds. Action list: SBDD.
Find at least 100 potential hits for /path/to/BRAF.pdb within 5 rounds. Action list: SBDD, LBDD.