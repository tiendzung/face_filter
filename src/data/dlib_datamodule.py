from typing import Any, Dict, Optional

import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

import os
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw


class DlibDataset(Dataset):
    def __init__(self, data_len, data_dir, xml_file):
        super().__init__()
        self.data_len = int(data_len)
        self.data_dir = data_dir
        self.samples = self._load_data(data_dir, xml_file)

    def __len__(self):
        return len(self.samples)

    # return: cropped_image (Tensor), landmarks (np.ndarray) unnormalized
    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['filename']
        box_top: int = sample['box_top']
        box_left: int = sample['box_left']
        box_width: int = sample['box_width']
        box_height: int = sample['box_height']
        landmarks: np.ndarray = sample['landmarks']
        original_image: Image = Image.open(
            os.path.join(self.data_dir, filename)).convert('RGB')
        cropped_image: Image = original_image.crop(
            (box_left, box_top, box_left+box_width, box_top+box_height))

        return cropped_image, landmarks # unnormalized

    def _load_data(self, data_dir: str, xml_file: str):
        """Load data from xml file."""
        xml_path = os.path.join(data_dir, xml_file)
        root = ET.parse(xml_path).getroot()
        samples = root.find('images')
        samples = [self._get_cropped_labeled_sample(
            sample) for sample in samples]
        return samples[0:self.data_len]

    def _get_cropped_labeled_sample(self, sample: ET.Element) -> Dict:
        filename = sample.attrib['file']
        width = int(sample.attrib['width'])
        height = int(sample.attrib['height'])

        box = sample.find('box')
        box_top = int(box.attrib['top'])
        box_left = int(box.attrib['left'])
        box_width = int(box.attrib['width'])
        box_height = int(box.attrib['height'])

        landmarks = np.array([
            [float(part.attrib['x']), float(part.attrib['y'])] for part in box
        ])
        landmarks -= np.array([box_left, box_top])  # crop

        return dict(
            filename=filename, width=width, height=height,
            box_top=box_top, box_left=box_left, box_width=box_width, box_height=box_height,
            landmarks=landmarks,
            # original_image=original_image, cropped_image=cropped_image,
        )

    @staticmethod
    def annotate_image(image: Image, landmarks: np.ndarray) -> Image:
        draw = ImageDraw.Draw(image)
        for i in range(landmarks.shape[0]):
            draw.ellipse((landmarks[i, 0] - 2, landmarks[i, 1] - 2,
                          landmarks[i, 0] + 2, landmarks[i, 1] + 2), fill=(255, 255, 0))
        return image


## transform a DlibDataset
## input (PIL image, np.ndarray landmarks)
## output (Tensor, np.ndarray) image size 224x224, centered and normalized landmarks
class TransformDataset(Dataset):
    def __init__(self, dataset: DlibDataset, transform: Optional[Compose] = None):
        self.dataset = dataset
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image in PIL format, landmarks in image pixel coordinates
        image, landmarks = self.dataset[idx]
        image = np.array(image)
        transformed = self.transform(
            image=image, keypoints=landmarks)
        image, landmarks = transformed["image"], transformed["keypoints"]
        _, height, width = image.shape
        landmarks = landmarks / np.array([width, height]) - 0.5
        return image, landmarks.astype(np.float32) # center and normalize

    # @staticmethod
    # def collate_fn(batch):
    #     images, landmarks = zip(*batch)
    #     return torch.stack(images), np.stack(landmarks)

    ## assume image batch tensor, normalized by imagenet
    @staticmethod
    def annotate_tensor(image: torch.Tensor, landmarks: np.ndarray) -> Image:

        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
        
        images = denormalize(image)
        images_to_save = []
        for lm, img in zip(landmarks, images):
            img = img.permute(1, 2, 0).numpy()*255
            h, w, _ = img.shape
            lm = (lm + 0.5) * np.array([w, h]) # convert to image pixel coordinates
            img = DlibDataset.annotate_image(Image.fromarray(img.astype(np.uint8)), lm)
            images_to_save.append( torchvision.transforms.ToTensor()(img) )

        return torch.stack(images_to_save)


class DlibDataModule(LightningDataModule):
    """Example of LightningDataModule for Dlib Facial Landmarks dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir = '/Users/tiendzung/Downloads/facial_landmarks-wandb/data/ibug_300W_large_face_landmark_dataset',
        train_val_test_split=[5_666, 1_000, 1_008],
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        data_set: DlibDataset = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train : Optional[Dataset] = None
        self.data_val : Optional[Dataset] = None
        self.data_test : Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # self.data_train = self.hparams.data_train
            # self.data_test = self.hparams.data_test
            self.data_train, self.data_val, self.data_test = random_split(
                    dataset=self.hparams.data_set,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42),
                )
            self.data_train = TransformDataset(self.data_train, self.hparams.transform_train)
            self.data_val = TransformDataset(self.data_val, self.hparams.transform_val)
            self.data_test = TransformDataset(self.data_test,self.hparams.transform_val)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    def test_dataset(cfg: DictConfig):
        dataset: DlibDataset = hydra.utils.instantiate(cfg.data_set)
        # dataset = dataset(data_dir=cfg.data_dir)
        print("dataset", len(dataset))
        image, landmarks = dataset.__getitem__(1)
        print("image", image.size, "landmarks", landmarks.shape)
        annotated_image = DlibDataset.annotate_image(image, landmarks)
        annotated_image.save(output_path / "test_dataset_result.png")

    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        annotated_batch = TransformDataset.annotate_tensor(bx, by)
        print("annotated_batch", annotated_batch.shape)
        torchvision.utils.save_image(annotated_batch, output_path / "test_datamodule_result.png")
        
        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")

        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")

        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("test data passed")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        test_dataset(cfg)
        test_datamodule(cfg)

    main()
