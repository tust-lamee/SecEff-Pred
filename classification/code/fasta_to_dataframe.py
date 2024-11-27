import pandas as pd

class Fasta_to_Dataframe(object):
    def fasta_to_dataframe(file_path):
        labels = []
        sequences = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('>'):
                labels.append(line[1:])
                sequences.append('')
            else:
                sequences[-1] += line

        data = {'Label': labels, 'Sequence': sequences}
        df = pd.DataFrame(data)

        return df