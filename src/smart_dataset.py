from monai.data import SmartCacheDataset
import pandas as pd
import numpy as np
import monai
import torch
from kipoiseq.transforms.functional import one_hot, fixed_len
from sklearn.model_selection import train_test_split



class OneHotEncoding(monai.transforms.MapTransform):
    def __init__(self, keys):
        monai.transforms.MapTransform.__init__(self, keys)
        print(f"keys to be transformed: {self.keys}")

    def __call__(self, x):
        output = {key: self.one_hot(x[key]) for key in self.keys}
        output['expression'] = x['expression']
        # print('output', output)
        return output

    def one_hot(self, seq):
        seq = np.array(list(seq)).astype(np.int64)
        seq = torch.LongTensor(seq)
        seq = torch.nn.functional.one_hot(seq, num_classes=5)
        return seq


def get_dataset(filename=r"../data/train_sequences.txt", max_sample_bytes=-1, replace_rate=0.2, cache_rate=0.02):
    f = open(filename, "r")
    lines = f.readlines(max_sample_bytes)
    f.close()
    tr_data = pd.DataFrame(lines)[0].str.split('\t', 1, expand=True)
    tr_data[1] = tr_data[1].str.replace('\n', "")
    tr_data[1] = tr_data[1].astype(np.float32)
    tr_data[0] = tr_data[0].str[17:-13]
    # below is the code to pad but it is giving an error
    # tr_data[0] = fixed_len(tr_data[0], 112, "start", "X")
    tr_data[0] = tr_data[0].str.replace("A", "0")
    tr_data[0] = tr_data[0].str.replace("C", "1")
    tr_data[0] = tr_data[0].str.replace("N", "2")
    tr_data[0] = tr_data[0].str.replace("G", "3")
    tr_data[0] = tr_data[0].str.replace("T", "4")
    # pad the data
    # print(tr_data[0])
    tr_data.rename(columns={0: 'sequence', 1: 'expression'}, inplace=True)
    train, val = train_test_split(tr_data, test_size=0.2)

    train = train.to_dict('records')
    val = val.to_dict('records')
    tr_dataset = SmartCacheDataset(train, transform=OneHotEncoding(keys=["sequence"]), replace_rate=replace_rate,
                                   cache_rate=cache_rate)
    val_dataset = SmartCacheDataset(val, transform=OneHotEncoding(keys=["sequence"]), replace_rate=replace_rate,
                                   cache_rate=cache_rate)
    return tr_dataset, val_dataset


if __name__ == "__main__":
    tr_dataset, val_dataset = get_dataset(max_sample_bytes=30000)
    # Next steps: implement batch > 1 (with padding) and flipping as augmentation
    for item in torch.utils.data.DataLoader(tr_dataset, batch_size=1):
        text = item["expression"]#.float()
        print(text)
        exit(0)
