from src.model import *
from src.dataset import *
from torchsummary import summary
from torch import optim, utils
from smart_dataset import get_dataset
from torch.utils.data import DataLoader
from torch import permute

# specify batch size, number of epochs, and learning rate
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-2
DATA_DIR = r'../data/train_sequences.txt'

# determine the device being used for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

net = PromoterNet()
# summary(net, input_size=(5, 112))
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),
                       lr=LR)

# get dataloaders
tr_dataset = get_dataset(filename=DATA_DIR, max_sample_bytes=30000)
train_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE)


# root_dir = r"../data"
# filename = "train_sequences.txt"
# train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
# loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)
# for each in loader:
#     print(each[0].size())


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        print(i)
        print(data)
        # Every data instance is an sequences + label pair
        sequences, labels = data["sequence"], data['expression']

        # change the order of the input shapes, final: [batch_size, channels, depth, height, width]
        sequences = permute(sequences, (0, 2, 1))
        sequences = sequences.type(torch.float)

        print(sequences.size())

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(sequences)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(' epoch {} batch {} loss: {}'.format(epoch_index, i + 1, last_loss))
            # tb_x = epoch_index * len(train_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


best_vloss = 1_000_000.
for epoch in range(EPOCHS):
    # get the loss
    avg_loss = train_one_epoch(epoch)

    # # get a validation loader and run this part later
    # running_vloss = 0.0
    # for i, vdata in enumerate(validation_loader):
    #     vinputs, vlabels = vdata
    #     voutputs = net(vinputs)
    #     vloss = loss_func(voutputs, vlabels)
    #     running_vloss += vloss
    #
    # avg_vloss = running_vloss / (i + 1)
    #
    # print('EPOCH {} LOSS train {} valid {}'.format(epoch, avg_loss, avg_vloss))
    #
    # # Track best performance, and save the model's state
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = 'model_cnn_epoch{}'.format(epoch)
    #     torch.save(net.state_dict(), model_path)
