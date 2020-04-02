import io
from torchvision import transforms
from google.cloud import storage
from pathlib import Path
from PIL import Image


class ImageTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose(
            # [transforms.Resize(256),
            #  transforms.CenterCrop(224),
            [transforms.Resize(76),
             transforms.CenterCrop(64),
             transforms.ToTensor(),
             normalize,
             ])
        return transform(x)


class GcsIO(object):
    def __init__(self, project, bucket_name):
        self.PROJECT = project
        self.BUCKET_NAME = bucket_name

    def upload_file(self, gcs_path, local_path):
        client = storage.Client(self.PROJECT)
        bucket = client.get_bucket(self.BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def download_file(self, gcs_path, local_path):
        p_path = Path(local_path).parent
        if not p_path.exists():
            p_path.mkdir(parents=True)

        client = storage.Client(self.PROJECT)
        bucket = client.get_bucket(self.BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    def load_image(self, gcs_path):
        client_storage = storage.Client(self.PROJECT)
        bucket = client_storage.get_bucket(self.BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        img = Image.open(io.BytesIO(blob.download_as_string()))
        return img