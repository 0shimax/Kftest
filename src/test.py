from pathlib import Path
import argparse
import collections
import json
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
from sklearn.metrics import accuracy_score, confusion_matrix
from google.cloud import storage

from model.trans_NFCM import TransNFCM
from optimizer.radam import RAdam
from feature.metric_data_loader import WBCDataset, loader
from metric.utils import cossim, val


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
    image_label = pandas.read_csv(
        Path("gs://",
             args.bucket_name,
             args.data_root, 
             args.metadata_file_name.format(args.subset))
    )
    image_label = image_label.sample(frac=1, random_state=551)
    image_label["class"] = image_label["class"] - 1
    image_label = image_label.values

    val_dataset = WBCDataset(args.n_class, image_label[:250], args.data_root, 
                             project=args.project, bucket_name=args.bucket_name,
                             subset=args.subset, train=False)
    test_dataset = WBCDataset(args.n_class, image_label[250:], args.data_root, 
                              project=args.project, bucket_name=args.bucket_name,
                              subset=args.subset, train=False)
    val_loader = loader(val_dataset, 1, shuffle=False)
    test_loader = loader(test_dataset, 1, shuffle=False)

    test_loader.dataset.gcs_io.download_file(args.resume_model,
                                             args.out_dir+"/"+args.resume_model.split("/")[-1])
    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    center_vec = val(args, model, val_loader, emb_dim=emb_dim)
    test(args, model, test_loader, center_vec)


def test(args, model, data_loader, center_vec):
    model.eval()
    label_idxs = []
    result_labels = []
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)
            label_idxs.append(cat.item())

            image_embedded_vec = model.predict(x=image, category=None)
            vec = F.softmax(image_embedded_vec, dim=1).squeeze(0).numpy()
            result_labels.append(cossim(vec, center_vec))

    write_metric(args, label_idxs, result_labels, args.n_class, 
                 list(idx2label.values()), data_loader.dataset.gcs_io)
    print("done")


def write_metric(args, target, predicted, n_class, class_names, gcs_io, 
                 cm_file='confusion_matrix.csv'):
    cm = confusion_matrix(target, predicted, labels=list(range(n_class)))
    accuracy = accuracy_score(target, predicted)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((class_names[target_index], class_names[predicted_index], count))

    df_cm = pandas.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file_path = '/'.join([args.out_dir, cm_file])
    with open(cm_file_path, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)
    gcs_io.upload_file('{}/{}'.format(args.out_dir, cm_file), cm_file_path)

    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': 'gs://{}/{}/{}'.format(args.bucket_name, args.out_dir, cm_file),
            'labels': class_names,
            }]
        }
    # meta dataをjsonに書き出し、DSLでfile_outputsに指定することでUIからConfusion Matrixを確認できる
    with open(args.out_dir+'/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    metrics = {
        'metrics': [{
        'name': 'accuracy-score',  # The name of the metric. Visualized as the column name in the runs table.
        'numberValue': accuracy,   # The value of the metric. Must be a numeric value.
        'format': "PERCENTAGE",    # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }

    with open(args.out_dir+'/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--project', default="<your project id>")
    parser.add_argument('--bucket-name', default="kf-test1234")
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--resume-model', default='export/wbc/NFCM_model.pth', help='path to trained model')
    parser.add_argument('--out-dir', default='export/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)