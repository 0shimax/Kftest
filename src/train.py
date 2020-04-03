from pathlib import Path
import argparse
import collections
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
from google.cloud import storage
from model.trans_NFCM import TransNFCM
from optimizer.radam import RAdam
from feature.metric_data_loader import WBCDataset, loader
from feature.utils import GcsIO


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

idx2label = {0:"Lymphocyte",
             1:"Monocyte",
             2:"Nuetrophil",
             3:"Eosinophil",
             4:"Basophil"}


def main(args):
    n_relational_embeddings = args.n_class**2
    n_tag_embeddings = args.n_class
    in_ch, out_ch, emb_dim = 3, 128, 128
    model = TransNFCM(in_ch, out_ch,
                      n_relational_embeddings, n_tag_embeddings,
                      embedding_dim=emb_dim).to(device)

    optimizer = RAdam(model.parameters(), weight_decay=1e-3)

    image_label = pandas.read_csv(
        Path("gs://",
             args.bucket_name,
             args.data_root, 
             args.metadata_file_name.format(args.subset))
    )
    image_label = image_label.sample(frac=1, random_state=551)
    image_label["class"] = image_label["class"] - 1
    image_label = image_label.values

    train_dataset = WBCDataset(args.n_class, image_label[:250], args.data_root, 
                               project=args.project, bucket_name=args.bucket_name,
                               subset=args.subset, train=True)
    train_loader = loader(train_dataset, args.batch_size)
    train(args, model, optimizer, train_loader)


def train(args, model, optimizer, data_loader, model_name="NFCM_model.pth"):
    model.train()
    for epoch in range(args.epochs):
        for i, (image, cat, near_image, near_cat, far_image, far_cat, near_relation, far_relation) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)
            near_image = near_image.to(device)
            near_cat = near_cat.to(device)
            far_image = far_image.to(device)
            far_cat = far_cat.to(device)
            near_relation = near_relation.to(device)
            far_relation = far_relation.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            loss = model(image, near_image, image, far_image,
                         cat, near_cat, cat, far_cat,
                         near_relation, far_relation).sum()
            loss.backward()
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

    model_path = args.out_dir+"/"+model_name
    torch.save(model.state_dict(), model_path)
    data_loader.dataset.gcs_io.upload_file(model_path, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--project', default="<your project id>")
    parser.add_argument('--bucket-name', default="kf-test1234")
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--out-dir', default='export/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)