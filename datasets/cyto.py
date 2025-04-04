import cv2
import glob
import os
from .utils import Datum, DatasetBase

template = ["A pap smear slide showing a {} cervical cells."]


class Cyto(DatasetBase):

    dataset_dir = "cyto_dataset"
    classes = [
        "1",
        "2",
        "3",
        "41",
        "42",
        # "43",
        "51",
        "52",
        "54",
        "57"
    ]

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join("", self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "")

        print(self.dataset_dir, self.image_dir)

        self.template = template

        train = self.create_list_of_datum("train"+f"_{num_shots}")


       

        # n_shots_val = min(num_shots, 4)
      
      
        val = self.create_list_of_datum("val"+f"_{num_shots}")
        test = self.create_list_of_datum("test")
        
        # val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        # train = self.generate_fewshot_dataset(train, num_shots=num_shots)
       


        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, im_files, idx):
        image = cv2.imread(im_files[idx])
        class_name = im_files[idx].replace("\\", "/").split("/")[-1].split("_")[0]
        class_ = self.classes.index(class_name)

        return image, class_, class_name, im_files[idx]

    def create_list_of_datum(self, set):
        """Create a list of Datum objects, each containing the image and label."""
        datum_list = []
        

        im_files = glob.glob(os.path.join(self.image_dir, set, "*.png"))


        for i in range(len(im_files)):
            # Get the image and the class
            image, class_, class_name, impath = self.__getitem__(im_files, i)

            # Create a Datum object
            datum = Datum(impath=impath, label=class_, classname=class_name)

            # Append the datum to the list
            datum_list.append(datum)

        return datum_list