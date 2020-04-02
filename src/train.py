import torch
from pathlib import Path
import argparse
import collections
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas
from model.trans_NFCM import TransNFCM
from optimizer.radam import RAdam
from feature.metric_data_loader import WBCDataset, loader
from metric.utils import print_result, cossim, val


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

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = RAdam(model.parameters(), weight_decay=1e-3)

    image_label = pandas.read_csv(
        Path(args.data_root, 
             args.metadata_file_name.format(args.subset))
    )
    image_label = image_label.sample(frac=1, random_state=551)
    image_label["class"] = image_label["class"] - 1
    image_label = image_label.values

    train_dataset = WBCDataset(args.n_class, image_label[:250], args.data_root, 
                               subset=args.subset, train=True)
    val_dataset = WBCDataset(args.n_class, image_label[:250], args.data_root, 
                              subset=args.subset, train=False)
    test_dataset = WBCDataset(args.n_class, image_label[250:], args.data_root, 
                              subset=args.subset, train=False)
    train_loader = loader(train_dataset, args.batch_size)
    val_loader = loader(val_dataset, 1, shuffle=False)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    center_vec = val(args, model, val_loader, emb_dim=emb_dim)
    test(args, model, test_loader, center_vec,
         show_image_on_board=args.show_image_on_board,
         show_all_embedding=args.show_all_embedding)


def train(args, model, optimizer, data_loader):
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

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                       '{}/NFCM_model.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
               '{}/NFCM_model.pth'.format(args.out_dir))


def test(args, model, data_loader, center_vec,
         show_image_on_board=True, show_all_embedding=False):
    model.eval()
    writer = SummaryWriter()
    weights = []
    images = []
    labels = []
    label_idxs = []
    result_labels = []
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)
            label_idxs.append(cat.item())
            labels.append(idx2label[cat.item()])
            images.append(image.squeeze(0).numpy())

            image_embedded_vec = model.predict(x=image, category=None)
            vec = F.softmax(image_embedded_vec, dim=1).squeeze(0).numpy()
            weights.append(vec)
            result_labels.append(cossim(vec, center_vec))

    weights = torch.FloatTensor(weights)
    images = torch.FloatTensor(images)
    if show_image_on_board:
        writer.add_embedding(weights, label_img=images)
    else:
        writer.add_embedding(weights, metadata=labels)
    print_result(label_idxs, result_labels)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="gs://kf-test1234/data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--resume-model', default='gs://kf-test1234/export/NFCM_model.pth', help='path to trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--show-image-on-board', action='store_false')
    parser.add_argument('--show-all-embedding', action='store_true')
    parser.add_argument('--out-dir', default='gs://kf-test1234/export', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)