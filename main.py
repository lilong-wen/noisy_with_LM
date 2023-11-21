import logging
import argparse
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
from config import parse_args
from utils import get_file_time, prepad_time
from model.model import Classifier

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from data_preprocess.datasets import (
    input_dataset,
    train_cifar100_transform,
    train_cifar10_transform,
)


def train(train_loader, model):


    feature_list = []
    noisy_label_list = []
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            feature, logits = model(images)
        
        feature_list.append(feature)
        noisy_label_list.append(labels)
        
    feature_torch = torch.vstack(feature_list)
    noisy_labels_torch = torch.vstack(noisy_label_list)
    # temporare
    save_dir = "cache_dir"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    feature_save_path = os.path.join(save_dir, "feature_torch.pt")
    torch.save(feature_torch, feature_save_path)
    label_save_path = os.path.join(save_dir, "noisy_label_torch.pt")
    torch.save(feature_torch, label_save_path)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parse_args(parser=parser)
    t0 = time.time()
    file_time = get_file_time(t0, args)

    # read noise files
    if args.noise_path is None:
        if args.dataset == "cifar10":
            args.noise_path = "data/CIFAR-N/CIFAR-10_human.pt"
            train_transform = train_cifar10_transform
            if args.warmups is None:
                args.warmups = 10
            args.mixmatch_lambda_u = 5
        elif args.dataset == "cifar100":
            args.noise_path = "data/CIFAR-N/CIFAR-100_human.pt"
            train_transform = train_cifar100_transform
            if args.warmups is None:
                args.warmups = 35
            args.mixmatch_lambda_u = 75
        else:
            raise NameError(f"Undefined dataset {args.dataset}")

    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        noise_type=args.noise_type,
        noise_path=args.noise_path,
        is_human=args.is_human,
    )

    # noise_type_map = {
        # "clean": "clean_label",
        # "worst": "worse_label",
        # "aggre": "aggre_label",
        # "rand1": "random_label1",
        # "rand2": "random_label2",
        # "rand3": "random_label3",
        # "clean100": "clean_label",
        # "noisy100": "noisy_label",
    # }
    # args.noise_type = noise_type_map[args.noise_type]

    cifar_n_label = torch.load(args.noise_path)
    clean_labels = cifar_n_label["clean_label"]
    noisy_labels = cifar_n_label[args.noise_type]

    # Run relevant files configuration
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.mkdir(args.tensorboard_dir)

    result_file = open(
        os.path.join(
            args.result_dir,
            "{}_{}_{}_{}_{}_{}.txt".format(
                args.dataset,
                args.noise_type,
                args.sample_split,
                args.ssl,
                args.cos_up_bound,
                args.cos_low_bound,
            ),
        ),
        "a+",
    )
    result_file.writelines(str(args) + "\n")

    log_file_name = os.path.join(args.log_dir, "{}.txt".format(file_time))
    handler = logging.FileHandler(log_file_name)
    logger.addHandler(handler)

    tb_writer = SummaryWriter(
        "{}/{}".format(
            args.tensorboard_dir,
            "{}_{}_{}_{}_{}_{}".format(
                args.dataset,
                args.noise_type,
                args.sample_split,
                args.ssl,
                args.cos_up_bound,
                args.cos_low_bound,
            ),
        )
    )

    # Start training
    logger.info(args)
    logger.info("Preparing data...")
    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        noise_type=args.noise_type,
        noise_path=args.noise_path,
        is_human=args.is_human,
    )
    args.num_classes = num_classes
    logging.info("Train labels examples:{}".format(train_dataset.train_labels[:20]))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = Classifier(num_classes=args.num_classes).cuda()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(optimizer, args.epochs, args.lr / 100)
    elif args.scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


    logging.info("Building model...")

    ## preprocess the dataset
    train(train_loader, model)
