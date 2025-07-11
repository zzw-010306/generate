import numpy as np
import pandas as pd
from rdkit import rdBase
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

def simi(scaffold,fps):
    sm_score = []
    for i in range(len(fps)):
            cons = DataStructs.FingerprintSimilarity(scaffold,fps[i])
            sm_score.append(cons)
    return sm_score

scaffold = Chem.RDKFingerprint(Chem.MolFromSmiles("CCn1/c(=C/C=C2\\CCCC(/C=C/C3=[N+](CC)c4ccc(C)c5cccc3c45)=C2Cl)c2cccc3c(C)ccc1c32"))

generative_smis = pd.read_csv('/home/wusiwei/project/retry/software/code/structure_generator/210910_test_generated_mols.csv')['SMILES']
mols = [Chem.MolFromSmiles(i) for i in generative_smis]
fps = [Chem.RDKFingerprint(x) for x in mols]

pd.DataFrame(simi(scaffold,fps))[0].to_csv('/home/wusiwei/project/retry/software/code/structure_generator/tanimoto_score.csv',index=False)
# print(pd.DataFrame(simi(scaffold,fps))[0])