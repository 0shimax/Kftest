from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_zoo.load_url('http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth')


def get_dataset():
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])
    testset = torchvision.datasets.MNIST(root='/tmp',
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=1,
                                                shuffle=False, 
                                                num_workers=2)
    return testloader


def main(args):
    test_loader = get_dataset
    test(args, test_loader)


def test(args, data_loader):
    model.eval()
    label_idxs = []
    result_labels = []
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)
            label_idxs.append(cat.item())

            out = model(image)
            result_labels.append(output.data.max(1)[1])

    write_metric(args, label_idxs, result_labels, args.n_class, 
                 list(idx2label.values()), data_loader.dataset.gcs_io)
    print("done")


def write_metric(args, target, predicted, n_class, class_names,
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

    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file_path,
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
    parser.add_argument('--project', default="<your project id>")
    parser.add_argument('--bucket-name', default="kf-test1234")
    parser.add_argument('--n-class', type=int, default=10, help='number of class')
    parser.add_argument('--out-dir', default='export/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)