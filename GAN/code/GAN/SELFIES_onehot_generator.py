#!/usr/bin/env python
# coding: utf-8

# Function one hot encoding for SELFIES

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import selfies as sf

class Onehot_Generator():
    # When instance is defined, compounds expressed in SELFIES list shuld be input.
    #SELFIES symbol label and max SELFIES length are optional.
    def __init__(self, SELFIES_list, symbol_label = None, max_length = 0):
        self.SELFIES_list = SELFIES_list
        self.label = []
        self.supplier = []
        self.max_length = max_length
        self.encoded_data = None

        # Split SELFIES into symbol 
        for selfies in SELFIES_list:
            #单词token化 #['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_2]']
            splitted_selfies = [symbol for symbol in sf.split_selfies(selfies)] 
            self.supplier.append(splitted_selfies)
                
            if symbol_label == None:
                self.label += list(set(splitted_selfies))
            
        if symbol_label == None:
            self.label = sorted(list(set(self.label)))

            self.label = ["[_]"] + self.label + ["[Unknown]"]
            self.label = np.array(self.label)
            
        else:
            self.label = np.array(symbol_label)
            
    # Obtain the number of the longest SELFIES
    def length_search(self):
        if self.max_length == 0:
            for selfies in self.supplier:
                if self.max_length < len(selfies):
                    self.max_length = len(selfies)
        
        return self.max_length
        
    # Convert SELFIES symbols to numbers corresponding to the list
    def encoding(self):
        # Make output csv data. The number of compounds * SELFIES length
        self.encoded_data = np.zeros((len(self.supplier), self.length_search()))
        
        # Single molecule processing
        for mol_idx, mol_symbols in enumerate(self.supplier):
            # Padding by "_" 
            while len(mol_symbols) < self.max_length:
                mol_symbols += ["[_]"]

            for symbol_idx in list(range(len(mol_symbols))):
                # Exception character handling
                if len(np.where(self.label == mol_symbols[symbol_idx])[0]) == 0:
                    self.encoded_data[mol_idx, symbol_idx] = len(self.label) - 1

                # Convert SELFIES to encoded number
                else:
                    self.encoded_data[mol_idx, symbol_idx] = np.where(self.label == mol_symbols[symbol_idx])[0][0]

        # Set data type
        self.encoded_data = self.encoded_data.astype(int)
        
        return self.encoded_data
    
    # Users can select data type.
    def onehot(self, mode = "PyTorch_normal"):
        
        if type(self.encoded_data) == type(None):
            self.encoding()
        
        # "PyTorch_CNN" mode outputs (number of data, feature dimension, symbol length)
        if mode == "PyTorch_CNN":
            self.onehot_data = np.zeros((self.encoded_data.shape[0], len(self.label), self.max_length))

            for mol_idx in list(range(self.encoded_data.shape[0])):
                for length_idx in list(range(self.max_length)):
                    self.onehot_data[mol_idx, self.encoded_data[mol_idx, length_idx], length_idx] = 1
            
        # "PyTorch_RNN"  mode outputs (symbol length, number of data, feature dimension)
        elif mode == "PyTorch_RNN":
            self.onehot_data = np.zeros((self.max_length, self.encoded_data.shape[0], len(self.label)))

            for mol_idx in list(range(self.encoded_data.shape[0])):
                for length_idx in list(range(self.max_length)):
                    self.onehot_data[length_idx, mol_idx, self.encoded_data[mol_idx, length_idx]] = 1
        
        # "PyTorch_normal" mode outputs (number of data, symbol length, feature dimension)
        elif mode == "PyTorch_normal":
            self.onehot_data = np.zeros((self.encoded_data.shape[0], self.max_length, len(self.label)), dtype = np.float16)

            for mol_idx in list(range(self.encoded_data.shape[0])):
                for length_idx in list(range(self.max_length)):
                    self.onehot_data[mol_idx, length_idx, self.encoded_data[mol_idx, length_idx]] = 1
                    
        # Error message
        else:
            return(print("Please input true word!"))
        
        return self.onehot_data

# For test
if __name__ == "__main__":
    X = pd.read_csv("../data/250k_rndm_zinc_drugs_clean_3.csv")
    X_sel = X.iloc[:20000, :]
    generator = Onehot_Generator(X_sel["smiles"])
    generator.onehot(mode = "PyTorch_normal")

