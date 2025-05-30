import torch
import esm
import csv
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO


class Classification_FeatureExtractor:
    def __init__(self, model_file, regression_weights_file, layer=33):
        """
        Initialize the categorical feature extractor

        Parameters:
        model_file: The path to the pretrained model file
        regression_weights_file: Regression weight file path
        layer: The model layer used (33 layers by default) is used to extract the sequence representation
        """
        self.model_file = Path(model_file)
        self.regression_weights_file = Path(regression_weights_file)
        self.layer = layer

        if not self.model_file.exists():
            raise FileNotFoundError(f"Model weights file not found at {self.model_file}")
        if not self.regression_weights_file.exists():
            raise FileNotFoundError(f"Regression weights file not found at {self.regression_weights_file}")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(self.model_file))
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def _split_list(self, lst, chunk_size):
        """
        Split the list into smaller lists by block size

        Parameters:
        lst: original list
        chunk_size: The size of each small list

        return: Split list builder
        """
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def extract(self, data_file, output_file, chunk_size=4, max_sequence_length=1020):
        """
        Extract sequence features from FASTA file and write to CSV format file

        Parameters:
        data_file: FASTA file path entered
        output_file: The path to the CSV file of the feature output
        chunk_size: Number of sequences processed in each batch (default 4)
        max_sequence_length: Maximum sequence length (exceeded will be truncated, default 1020)
        """
        all_data = []

        for idx, record in enumerate(SeqIO.parse(data_file, "fasta")):
            label = record.id
            sequence = str(record.seq)
            if len(sequence) > max_sequence_length:
                sequence = sequence[:max_sequence_length]
            data_tuple = (f"{label}_{idx}_{sequence}", sequence)
            all_data.append(data_tuple)

        small_lists = list(self._split_list(all_data, chunk_size))

        with open(output_file, 'w', encoding='utf-8') as svm_file:
            for one in tqdm(small_lists, desc="Processing batches"):
                batch_labels, batch_strs, batch_tokens = self.batch_converter(one)
                batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[self.layer], return_contacts=True)
                    token_representations = results["representations"][self.layer]

                for i, tokens_len in enumerate(batch_lens):
                    sequence_tensor = token_representations[i, 1:tokens_len - 1]
                    sequence_feature_vector = sequence_tensor.mean(dim=0).cpu().numpy()

                    label = batch_labels[i].split('_')[0]
                    feature_string = ','.join(map(str, sequence_feature_vector))
                    svm_file.write(f"{label},{feature_string}\n")


class Regression_FeatureExtractor:
    def __init__(self, model_file, regression_weights_file, layer=33):
        self.model_file = Path(model_file)
        self.regression_weights_file = Path(regression_weights_file)
        self.layer = layer

        if not self.model_file.exists():
            raise FileNotFoundError(f"Model weights file not found at {self.model_file}")
        if not self.regression_weights_file.exists():
            raise FileNotFoundError(f"Regression weights file not found at {self.regression_weights_file}")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(self.model_file))
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def _split_list(self, lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def extract(self, data_file, output_file, chunk_size=4, max_sequence_length=1020):
        """
        Extract sequence features from CSV file (for regression tasks) and output to CSV format

        Parameters:
        data_file: Input CSV file path. Each row in the file should follow the format: sequence,label
        output_file: The path to the CSV file of the feature output
        chunk_size: Number of sequences processed in each batch (default 4)
        max_sequence_length: Maximum sequence length (exceeded will be truncated, default 1020)
        """
        all_data = []

        with open(data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if len(row) != 2:
                    print(f"Skipping malformed line {idx + 1}: {row}")
                    continue
                sequence, label = row
                sequence = sequence.strip()
                label = label.strip()
                if len(sequence) > max_sequence_length:
                    sequence = sequence[:max_sequence_length]
                data_tuple = (f"{label}_{idx}_{sequence}", sequence)
                all_data.append(data_tuple)

        small_lists = list(self._split_list(all_data, chunk_size))

        with open(output_file, 'w', encoding='utf-8') as svm_file:
            for one in tqdm(small_lists, desc="Processing batches"):
                batch_labels, batch_strs, batch_tokens = self.batch_converter(one)
                batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[self.layer], return_contacts=True)
                    token_representations = results["representations"][self.layer]

                for i, tokens_len in enumerate(batch_lens):
                    sequence_tensor = token_representations[i, 1:tokens_len - 1]
                    sequence_feature_vector = sequence_tensor.mean(dim=0).cpu().numpy()

                    label = batch_labels[i].split('_')[0]
                    feature_string = ','.join(map(str, sequence_feature_vector))
                    svm_file.write(f"{label},{feature_string}\n")
