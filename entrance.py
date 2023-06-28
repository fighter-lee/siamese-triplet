import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from common import extract_embeddings, plot_embeddings
from datasets import SiameseMNIST
from resnet import resnet50
from trainer import fit
from datasets import TripletMNIST
from networks import EmbeddingNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss


def getEmbeddingNet():
    # return EmbeddingNet()
    return resnet50(2)

def siameseTest():
    # Step 1
    siamese_train_dataset = SiameseMNIST(train_dataset)  # Returns pairs of images and target same/different
    siamese_test_dataset = SiameseMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    # Set up the network and training parameters
    # Step 2
    embedding_net = getEmbeddingNet()
    # Step 3
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()

    # Step 4
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 2
    log_interval = 500

    fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl)
    val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_cl, val_labels_cl)

def tripletTest():
    triplet_train_dataset = TripletMNIST(train_dataset)  # Returns triplets of images
    triplet_test_dataset = TripletMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    margin = 1.
    embedding_net = getEmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10
    log_interval = 500

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_tl, train_labels_tl)
    val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_tl, val_labels_tl)



if __name__ == '__main__':
    mean, std = 0.28604059698879553, 0.35302424451492237
    batch_size = 256

    train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean,), (std,))
                                 ]))
    test_dataset = FashionMNIST('data/FashionMNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))

    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    n_classes = 10

    # Set up data loaders
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    siameseTest()
    # tripletTest()


