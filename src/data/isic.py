import os
import pandas as pd
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T


class ISIC(Dataset):
    def __init__(
        self,
        label_map,
        class_map,
        image_dir="data/processed/images",
        filename_col="image_name",
        label_col="benign_malignant",
        transforms=None,
    ):
        """
        Initialize the ISIC dataset.

        :param label_map:       location of the label map csv file; csv should have at least two columns: {filename_col}
                                which stores the image name, and {label_col} which defines the image class
        :param class_map:       dict mapping the class name as defined in {label_col} to an int value
        :param image_dir:       path to image directory
        :param filename_col:    column name where the image names are stored
        :param label_col:       column name where the class names are stored
        :param transforms:      set of PyTorch data augmentation transforms

        Usage example:

        from torch.utils.data import DataLoader
        dataset = ISIC(
                "data/processed/train.csv",
                {"benign": 0, "malignant": 1},
                image_dir="data/processed/images",
                filename_col='image_name',
                label_col='benign_malignant',
                transforms=None,
            )
        dataloader = DataLoader(dataset, batch_size=64)
        print(next(iter(dataloader)))
        """
        # Indexing
        self.label_map = pd.read_csv(label_map)
        self.filename_col = filename_col
        self.label_col = label_col
        self.class_map = class_map

        # IO
        self.image_dir = image_dir
        if len(glob(os.path.join(self.image_dir, "*"))) == 0:
            raise IOError(
                f"Image directory '{self.image_dir}' is empty or does not exist."
            )

        # Transforms
        if isinstance(transforms, list):
            transforms = T.Compose(transforms)
        self.transforms = transforms

    def __len__(self):
        return len(self.label_map)

    def __getitem__(self, idx):
        # Image
        image_path = os.path.join(
            self.image_dir, self.label_map.loc[idx, self.filename_col] + ".jpg"
        )
        image_as_pil = Image.open(image_path)
        image = T.ToTensor()(image_as_pil)
        if self.transforms is not None:
            image = self.transforms(image)

        # Label
        label = self.class_map[self.label_map.loc[idx, self.label_col]]

        return image, label
