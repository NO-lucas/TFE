from .ucf101 import UCF101
from .sun397 import SUN397
from .eurosat import EuroSAT
from .food101 import Food101
from .imagenet import ImageNet
from .sipakmed import SipakMed
from .fgvc import FGVCAircraft
from .hicervix import HiCervix
from .caltech101 import Caltech101
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures
from .stanford_cars import StanfordCars
from .oxford_flowers import OxfordFlowers
from .dataset_kaggle import Dataset_kaggle1, Dataset_kaggle2
from .cyto import Cyto
from .cyto_random import Cyto_random
from .cyto_51_52 import Cyto_51_52


dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "imagenet": ImageNet,
    "sipakmed": SipakMed,
    "kaggle1": Dataset_kaggle1,
    "kaggle2": Dataset_kaggle2,
    "hicervix": HiCervix,
    "cyto":Cyto,
    "cyto_random":Cyto_random,
    "cyto_51_52":Cyto_51_52
}




def build_dataset(
    dataset, root_path, shots, level="level_1", pourcentage=0.0, preprocess=None
):
    if dataset == "imagenet":
        return dataset_list[dataset](root_path, shots, preprocess)
    elif dataset == "hicervix":
        return dataset_list[dataset](root_path, shots, level, pourcentage)
    else:
        return dataset_list[dataset](root_path, shots)
