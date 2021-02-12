import os
import json
import pickle
import concurrent.futures
import tqdm

import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset_utils import *
from utils.parser_utils import get_args


class SMILES_FASTA_Dataset(Dataset):

    def __init__(self, main_df, train_df, args):

        self.main_df = main_df

        if train_df is None:
            self.is_train = True
            self.train_df = main_df
            self.based_on = args.based_on
        else:
            self.is_train = False
            self.train_df = train_df
            self.based_on = "row"

        self.max_smi_len = 100
        self.max_prot_len = 1000
        self.randomized_smiles = args.augmentation
        self.support_size = args.support_size
        
        self.main_pfam_groups = {}
        self.train_pfam_groups = {}
        
        self.main_prot_families = self.main_df['pfam'].unique()
        self.train_prot_families = self.train_df['pfam'].unique()
        
        for g in self.main_prot_families: 
            self.main_pfam_groups[g] = self.main_df[self.main_df['pfam'] == g]
        
        for g in self.train_prot_families:
            temp_df = self.train_df[self.train_df['pfam'] == g] 
            self.train_pfam_groups[g] =  temp_df.sort_values(by=['affinity_score'], ascending=False)


    def __len__(self):
        return self.data_length


    def __getitem__(self, index):

        if self.based_on == "pfam":
            pfam = self.main_prot_families[index%len(self.main_prot_families)]
        elif self.based_on == "row":
            pfam = self.main_df.iloc[index%len(self.main_df)]["pfam"]
        
        target_index, support_indices = self.get_support_and_target_set_indicies_v2(pfam, self.support_size)

        if self.based_on == "row":
            target_index = index
        
        support_ligands = []
        support_proteins = [] 
        if support_indices is not None:
            support_data = self.train_df.iloc[support_indices]  
            for _, row in support_data.iterrows():
                sup_smiles = row['smiles']
                sup_protein = row['aa_sequence']

                sup_random_smiles = randomize_smiles(sup_smiles, canonical=not self.randomized_smiles)

                if sup_random_smiles is not None:
                    sup_smiles = sup_random_smiles

                sup_smiles, sup_protein = self.encode(sup_smiles, sup_protein)
                
                support_ligands.append(torch.from_numpy(sup_smiles).long())
                support_proteins.append(torch.from_numpy(sup_protein).long())
            
            support_ligands = torch.stack(support_ligands)
            support_proteins = torch.stack(support_proteins)
            support_set_labels = torch.from_numpy(support_data['affinity_score'].values).float()
        
        else:
            assert not self.is_train
            support_ligands = torch.zeros((self.support_size,100)).long()
            support_proteins = torch.zeros((self.support_size,1000)).long()
            support_set_labels = torch.zeros((self.support_size)).float()


        target_set_label = torch.from_numpy(np.array([self.main_df.iloc[target_index]['affinity_score']])).float()
        
        smiles = self.main_df.iloc[target_index]['smiles']
        protein = self.main_df.iloc[target_index]['aa_sequence']

        random_smiles = randomize_smiles(smiles, canonical=not self.randomized_smiles)

        if random_smiles is not None:
            smiles = random_smiles

        smiles, protein = self.encode(smiles, protein)

        return support_ligands, support_proteins, torch.from_numpy(smiles).long(), torch.from_numpy(protein).long(), support_set_labels, target_set_label


    def encode(self, smiles, protein):
        xd = self.label_smiles(smiles.strip(), self.max_smi_len, CHARISOSMISET)
        xt = self.label_sequence(protein.strip(), self.max_prot_len, CHARPROTSET)    
        return xd, xt


    def label_smiles(self, line, MAX_SMI_LEN, smi_ch_ind):
        # TODO: 'X' char is not in our SMILES char set !!!
        x = np.zeros(MAX_SMI_LEN, dtype=int)
        for i, ch in enumerate(line[:MAX_SMI_LEN]):
            try:
                x[i] = smi_ch_ind[ch]
            except KeyError:
                print("unknown char", ch)
        return x 


    def label_sequence(self, line, MAX_SEQ_LEN, smi_ch_ind):
        x = np.zeros(MAX_SEQ_LEN, dtype=int)
        for i, ch in enumerate(line[:MAX_SEQ_LEN]):
            if ch in smi_ch_ind.keys():
                x[i] = smi_ch_ind[ch]
        return x 


    def get_support_and_target_set_indicies_v2(self, pfam, support_size):
        target_index = self.main_pfam_groups[pfam].sample(1).index[0]
        
        if not (pfam in self.train_pfam_groups):
            return target_index, None

        pfam_df = self.train_pfam_groups[pfam]

        if len(pfam_df) < support_size:
            sup_indices = list(pfam_df.index)
            i = 0
            while len(sup_indices) < support_size:
                sup_indices.append(list(pfam_df.index)[i])
                i += 1
                i %= len(list(pfam_df.index))
        else:
            assert (support_size % 2) == 0
            sup_indices = list(pfam_df.iloc[:support_size//2].index)
            sup_indices.extend(list(pfam_df.iloc[-(support_size//2):].index))
        
        return target_index, sup_indices


    def get_support_and_target_set_indicies(self, pfam, support_size): # support size is 6 in our case
        
        support_size_val = 0
        if pfam in self.pos_groups:
            positives = np.asarray(self.pos_groups[pfam])
            support_size_val += len(positives)
        if pfam in self.neg_groups:
            negatives = np.asarray(self.neg_groups[pfam])
            support_size_val += len(negatives)

        target_index = self.main_pfam_groups[pfam].sample(1).index  #np.random.choice(np.append(positives,negatives),1)

        if support_size_val == 0:
            return int(target_index[0]), None
        
        if self.train_df is not None:
            if target_index[0] in positives: positives = positives[positives != target_index[0]]
            elif target_index[0] in negatives: negatives = negatives[negatives != target_index[0]]    
        
        else: # test so give support set not randomly
            if(len(negatives)>=support_size/2 and len(positives)>=int(support_size/2)):#everything is ok
                return int(target_index[0]),np.asarray(negatives[int(-support_size/2):].extend(positives[:int(support_size/2)]))
            
            indicies = []
        
            if(len(positives) < len(negatives)):# at least positives not enough
                indicies.extend(positives)
                if(len(negatives)>= support_size - len(indicies)):#negatives can compensate
                    indicies.extend(negatives[-(support_size - len(indicies)):])
                else: #both are not enough
                    if(len(positives)>0):#put positives more
                        indicies.extend(negatives)
                        indicies.extend(np.random.choice(positives,size = support_size - len(indicies),replace=True))
                    else:#there is no positive
                        indicies.extend(np.random.choice(negatives,size = support_size - len(indicies),replace=True))

            else:# at least negatives are not enough
                indicies.extend(negatives)
                if(len(positives)>= support_size - len(indicies)):#positives can compensate
                    indicies.extend(positives[:support_size - len(indicies)])
                elif len(positives)>0:#both are not enough so put positives more
                    indicies.extend(positives)
                    indicies.extend(np.random.choice(positives,size = support_size - len(indicies),replace=True))

            return int(target_index[0]), np.asarray(indicies)
            
        if(len(negatives)>=support_size/2 and len(positives)>=int(support_size/2)):#everything is ok
            return int(target_index[0]),np.concatenate((np.random.choice(negatives,size=int(support_size/2),replace=False),np.random.choice(positives,size=int(support_size/2),replace=False)))
        
        indicies = []
        
        if(len(positives) < len(negatives)):# at least positives not enough
            indicies.extend(positives)
            if(len(negatives)>= support_size - len(indicies)):#negatives can compensate
                indicies.extend(np.random.choice(negatives,size = support_size - len(indicies),replace=False))
            else:#both are not enough
                if(len(positives)>0):#put positives more
                    indicies.extend(negatives)
                    indicies.extend(np.random.choice(positives,size = support_size - len(indicies),replace=True))
                else:#there is no positive
                    indicies.extend(np.random.choice(negatives,size = support_size - len(indicies),replace=True))

        else:# at least negatives are not enough
            indicies.extend(negatives)
            if(len(positives)>= support_size - len(indicies)):#positives can compensate
                indicies.extend(np.random.choice(positives,size = support_size - len(indicies),replace=False))
            elif len(positives)>0:#both are not enough so put positives more
                indicies.extend(positives)
                indicies.extend(np.random.choice(positives,size = support_size - len(indicies),replace=True))
        
        return int(target_index[0]), np.asarray(indicies)




class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.batch_size = args.batch_size
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.based_on = args.based_on

        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.val_path)
        test_df = pd.read_csv(args.test_path)
        self.dataset_train = SMILES_FASTA_Dataset(main_df=train_df, train_df=None, args=args)
        self.dataset_val = SMILES_FASTA_Dataset(main_df=val_df, train_df=train_df, args=args)
        self.dataset_test = SMILES_FASTA_Dataset(main_df=test_df, train_df=train_df, args=args)
        self.continue_from_iter(current_iter=current_iter)
        self.args = args


    def get_dataloader(self, data_type):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        if data_type == "train":
            return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, drop_last=True)
        elif data_type == "val":
            return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, drop_last=True)
        elif data_type == "test":
            return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, drop_last=True)


    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * self.batch_size)


    def get_train_batches(self, total_batches):
        self.dataset_train.data_length = total_batches * self.batch_size
        self.total_train_iters_produced += self.batch_size
        for sample_id, sample_batched in enumerate(self.get_dataloader("train")):
            yield sample_batched


    def get_val_batches(self):
        self.dataset_val.data_length = len(self.dataset_val.main_df)
        for sample_id, sample_batched in enumerate(self.get_dataloader("val")):
            yield sample_batched


    def get_test_batches(self):
        self.dataset_test.data_length = len(self.dataset_test.main_df)
        for sample_id, sample_batched in enumerate(self.get_dataloader("test")):
            yield sample_batched



if __name__ == "__main__":
    
    class dummy_arg():
        def __init__(self):
            self.augmentation = False
            self.support_size = 6
            self.based_on = "pfam"

    train_df = pd.read_csv("datasets/bdb_pfam_0_train.csv")
    val_df = pd.read_csv("datasets/bdb_pfam_0_test_cold_prot.csv")
    args = dummy_arg()
    dataset = SMILES_FASTA_Dataset(main_df=val_df, train_df=train_df, args=args)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,drop_last=True)

    dataset.data_length = len(dataset.main_df)

    i = 0
    for data in loader:
        i+=1
        for e in data:
            print(e.shape, e.dtype)
            try:
                print(e.mean())
            except:
                pass
            print()
        
        
        