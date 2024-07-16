source activate generative_block_vae
python Generative_block_CUI.py "O=C1NC(=O)c2ccccc21" "2,7" "../../data/fragment_list.txt" "CCCCCC C" "C1C2CC1C2" "../../result/210910_test_generated_mols.csv" &> error.log

# argment 1: Scaffold SMILES
# argment 2: Asterisk indices as strings. Separator is ","
# argment 3: Building block list path
# argment 4: Input fragment list as strings. Separator is " "
# argment 5: Priority output fragment SMILES
# argment 6: Output generated molecules SMILES file path
