import torch
#from torchsummary import summary
from torch.optim import Adam
from torch.utils.data import DataLoader
import time

from model import *
from dataset import *
from smart_dataset import get_dataset

# determine the device being used for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

# root_dir = r"../data"
# filename = "train_sequences.txt"
# train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
# loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)
# for each in loader:
#     print(each[0].size())

def train_one_epoch(epoch_index, data_loader, optimizer, net, loss_func):
    total_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(data_loader):
        #print(i)
        #print(data)
        # Every data instance is an sequences + label pair
        sequences, labels = data["sequence"].float().to(DEVICE), data['expression'].float().to(DEVICE)

        # change the order of the input shapes, final: [batch_size, channels, depth, height, width]
        #sequences = torch.permute(sequences, (0, 2, 1))
        sequences = sequences.type(torch.float)

        #print(sequences.size())

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
        total_loss += loss.item()

        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000  # loss per batch
        #     print(' epoch {} batch {} loss: {}'.format(epoch_index, i + 1, last_loss))
        #     # tb_x = epoch_index * len(train_loader) + i + 1
        #     # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return total_loss

def val_one_epoch(epoch_index, data_loader, net, loss_func):
    total_loss = 0.
    last_loss = 0.
    net.eval() # Some layers should act differently in test time if there is any (e.g. batchnorm, dropout...). Switch to eval mode for the model.
    with torch.no_grad():
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(data_loader):
            #print(i)
            #print(data)
            # Every data instance is an sequences + label pair
            sequences, labels = data["sequence"].float(), data['expression'].float()#.to(DEVICE)

            # change the order of the input shapes, final: [batch_size, channels, depth, height, width]
            #sequences = torch.permute(sequences, (0, 2, 1))
            sequences = sequences.type(torch.float)

            #print(sequences.size())

            # Make predictions for this batch
            outputs = net(sequences)

            # Compute the loss and its gradients
            loss = loss_func(outputs, labels)

            # Gather data and report
            total_loss += loss.item()

            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000  # loss per batch
            #     print(' epoch {} batch {} loss: {}'.format(epoch_index, i + 1, last_loss))
            #     # tb_x = epoch_index * len(train_loader) + i + 1
            #     # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
    net.train() # Switch back to training mode for the model.
    return total_loss

def training_loop(  BATCH_SIZE = 1,
                    EPOCHS = 10,
                    LR = 1e-2,
                    DATA_DIR = r'../data/train_sequences.txt',
                    loss_func = nn.MSELoss()):
    # get dataloaders
    net = PromoterNet().to(DEVICE)
    optimizer = Adam(net.parameters(),lr=LR)

    tr_dataset, val_dataset = get_dataset(filename=DATA_DIR, max_sample_bytes=-1)
    train_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        # get the loss
        start_epoch = time.time()
        avg_tr_loss = train_one_epoch(epoch, train_loader, optimizer, net, loss_func) / len(train_loader)
        avg_val_loss = val_one_epoch(epoch, val_loader, net, loss_func) / len(val_loader)
        end_epoch = time.time()

        print(f"Epoch: {epoch}/{EPOCHS} | Tr.Loss: {avg_tr_loss} | Val.Loss: {avg_val_loss} | Time: {end_epoch-start_epoch}")

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

def inference_loop():
    # Function to create submission.txt from the trained model.
    pass

if __name__ == "__main__":
    training_loop(  BATCH_SIZE = 10,
                    EPOCHS = 100,
                    LR = 1e-2,)