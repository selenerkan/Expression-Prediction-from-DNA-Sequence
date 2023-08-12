from monai.data import SmartCacheDataset
from monai.transforms import RandFlipd, ToTensord, SpatialPadd, Transposed
import pandas as pd
import numpy as np
import monai
import torch
from kipoiseq.transforms.functional import one_hot, fixed_len
from sklearn.model_selection import train_test_split
import time
np.random.seed(0)
torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def get_dataset(filename=r"../data/train_sequences.txt", max_sample_bytes=-1, replace_rate=0.2, cache_rate=0.2):
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
    
    transforms = [ 
        OneHotEncoding(keys=["sequence"]),
        Transposed(keys=["sequence"],indices=[1,0]),
        RandFlipd(keys=["sequence"], prob=0.5, spatial_axis=0),
        SpatialPadd(keys=["sequence"],spatial_size=[120]),
        ToTensord(keys=["sequence","expression"], device=DEVICE)
    ]

    tr_dataset = SmartCacheDataset(train, transform=transforms, replace_rate=replace_rate,
                                   cache_rate=cache_rate, shuffle=False)
    val_dataset = SmartCacheDataset(val, transform=transforms, replace_rate=replace_rate,
                                   cache_rate=cache_rate)
    return tr_dataset, val_dataset


if __name__ == "__main__":
    tr_dataset, val_dataset = get_dataset(max_sample_bytes=-1)
    # Next steps: implement batch > 1 (with padding) and flipping as augmentation
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=10)
    start = time.time()
    for item in tr_loader:
        #text = item["sequence"]#.float()
        #label = item["expression"]#.float()
        #print(text)
        #print(text)
        #exit(0)
        continue
    end = time.time()
    print(end-start)

    start = time.time()
    for item in tr_loader:
        continue
    end = time.time()
    print(end-start)