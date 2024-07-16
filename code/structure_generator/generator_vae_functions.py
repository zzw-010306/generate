#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import EditableMol, BondType, ReplaceCore, ReplaceSidechains, rdFMCS
from rdkit.Chem.rdFMCS import FindMCS, BondCompare, AtomCompare
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.rdFMCS import FindMCS, BondCompare, AtomCompare
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D, MolDraw2DSVG, MolsToGridImage
from rdkit.Chem.MolStandardize.charge import Uncharger
from itertools import product

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import selfies as sf
from scipy.stats import norm
import sys

# Obtain module path
# Import VAE
sys.path.append("/home/wusiwei/project/retry/software/code/VAE")
from fragment_vae import VAE, Encoder, Decoder, Decoder_input
from SELFIES_onehot_generator import Onehot_Generator


class VAE_predict():
    def __init__(self, vae_path, label_path):
        
        self.vae = VAE(Encoder(), Decoder(), Decoder_input())
        self.vae.load_state_dict(torch.load(vae_path, map_location = torch.device("cpu")))
        self.vae.eval()

        self.label = list(np.array(pd.read_csv(label_path, index_col = 0)["Symbol"]))
        self.max_length = 30
        
    # SMILES to one hot
    def onehot_encode_from_smiles(self, smiles_list):

        generator = Onehot_Generator([sf.encoder(smiles) for smiles in smiles_list], symbol_label = self.label, max_length = self.max_length)

        return generator.onehot(mode = "PyTorch_normal")
    
    # Onehot to SELFIES
    def onehot_decode_to_selfies(self, encoded_data):
        restoration_selfies = []
        for i in list(range( encoded_data.shape[0])):
            restoration_selfies.append("".join(np.array(self.label)[encoded_data[i, :]].tolist()).replace("[_]", ""))

        return np.array(restoration_selfies)

    # Latent space coordinates to strings
    def VAE_decoder(self, latent_vec):

        # Convert data type
        latent_vec = latent_vec.detach().float()
        decoder_input = torch.reshape(latent_vec, (-1, 1, 256))

        recursive_num = 29

        # String length-times iteration
        for sequence_idx in list(range(recursive_num)):

            # Next character prediction
            pred_char = self.vae.decoder(decoder_input)[:, -1, :]
            pred_char = torch.reshape(pred_char, (pred_char.size(0), 1, pred_char.size(1)))

            # Convert probability to the character index
            pred_tag = torch.argmax(pred_char, 2)

            # Convert to one-hot vector
            new_onehot = torch.zeros(pred_char.size(0), 1, pred_char.size(2))

            for i in list(range(pred_char.size(0))):
                new_onehot[i, 0, pred_tag[i].item()] = 1

            # Convert one-hot vector to decoder-inputtable dimension data
            new_input = self.vae.dim_converter(new_onehot)

            # Add predict results
            decoder_input = torch.cat([decoder_input, new_input], dim = 1)

        # Output results
        decoder_output = self.vae.decoder(decoder_input)

        # Convert to character indices
        X_pred = torch.argmax(decoder_output, 2).detach().numpy()

        return X_pred
    
    def part_generator(self, input_smiles, generate_num = 10, random_state = 0, var_weight = 3):
        
        # Execute one hot encoding
        generator = Onehot_Generator([sf.encoder(input_smiles)], symbol_label = self.label, max_length = self.max_length)
        X_onehot = generator.onehot(mode = "PyTorch_normal")

        # Convert data to latent space coordinates
        latent, logvar = self.vae.encoder(torch.from_numpy(X_onehot).float())

        latent = latent.detach().numpy()
        logvar = logvar.detach().numpy()

        var = np.exp(logvar)

        np.random.seed(random_state)
        random_latent_vectors = torch.from_numpy(np.random.normal(latent, var * var_weight, (generate_num, 256)))

        # Convert latent space coordicaties to predicted one hot data
        generated_encoded = self.VAE_decoder(random_latent_vectors)

        # One hot data to SELFIES
        selfies_matrix = self.onehot_decode_to_selfies(generated_encoded)

        # SELFIES to SMILES
        generated_smiles = []
        for generated_selfies in selfies_matrix:
            generated_mol = Chem.MolFromSmiles(sf.decoder(generated_selfies))

            if generated_mol != None:
                generated_smiles.append(Chem.MolToSmiles(generated_mol))

            else:
                generated_smiles.append("None")

        # Select valid SMILES
        generated_structure = list(set([smiles for smiles in generated_smiles if Chem.MolFromSmiles(smiles) != None]))
        generated_structure

        return generated_structure
    
    # SMILES to latent space coordinates
    def smiles_to_latent(self, input_smiles_list):
        input_selfies_list = [sf.encoder(s) for s in input_smiles_list]
        generator = Onehot_Generator(input_selfies_list, symbol_label = self.label, max_length = self.max_length)
        input_onehot = torch.from_numpy(generator.onehot())
        latent_vector, _ = self.vae.encoder(input_onehot)
        return latent_vector