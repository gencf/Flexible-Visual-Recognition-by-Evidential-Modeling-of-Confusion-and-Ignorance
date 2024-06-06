import torch
import torch.nn as nn
import torchvision.models as models
import os


class ResNet18_custom_class_number(nn.Module):
    """
    A custom ResNet18 model with a new fully connected layer for a different number of classes.

    Args:
        num_classes (int): The number of classes in the dataset.
        project_root (str): The project root directory.
        model_cache_path (str): The relative path to save the model.
        use_pretrained (bool): Whether to use the pretrained weights.
        verbose (bool): Whether to print the model information.

    """

    # initialize the model
    def __init__(self,
                 num_classes: int,
                 project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                 use_pretrained: bool = False,
                 model_cache_path: str = os.path.join("weights", "resnet18"),
                 verbose: bool = True,
                 last_activation: nn.Module = nn.Sigmoid()):
        
        super(ResNet18_custom_class_number, self).__init__()

        self.num_classes = num_classes
        self.project_root = project_root
        self.model_cache_path = model_cache_path
        self.use_pretrained = use_pretrained
        self.verbose = verbose
        self.last_activation = last_activation


        # check if the number of classes is valid
        if self.num_classes <= 0:
            raise ValueError("The number of classes must be a positive integer.")
        
        
        if self.use_pretrained and (self.model_cache_path != ''):
            # set the model save path
            model_save_path = os.path.join(self.project_root, self.model_cache_path + "_pretrained.pt")

            # check if the model is already downloaded
            if os.path.exists(model_save_path):
                if verbose:
                    print(f"Using the pretrained resnet18 from: {model_save_path}")
                
                # load the model
                self.resnet18 = torch.load(model_save_path)
            else:
                if verbose:
                        print(f"Downloading pretrained resnet18 to: {model_save_path}")
                
                # check if the model saved in a separate directory (e.g. weights)
                if len(self.model_cache_path.split(os.path.sep)) > 1:
                    # create if the directory does not exist, if not already created, create the directory
                    model_save_dir = os.path.join(self.project_root, os.path.dirname(self.model_cache_path))
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                
                # download and load the model
                self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

                # create a new fully connected layer for the new number of classes
                self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, num_classes)

                # save the model
                torch.save(self.resnet18, model_save_path)

                # print the model save path
                if verbose:
                    print(f"Saved the pretrained resnet18 to: {model_save_path}")
            
        else:
            # basically just load the model
            if self.use_pretrained:
                # print that the resnet18 model is loaded
                self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                if verbose:
                    print("Loaded the resnet18 model with pretrained weights.")
            else:
                self.resnet18 = models.resnet18(weights=None)
                # print that the resnet18 model is loaded
                if verbose:
                    print("Loaded the resnet18 model without pretrained weights.")


            # create a new fully connected layer for the new number of classes
            self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, num_classes)

    # forward pass
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.last_activation(self.resnet18(x))





if __name__ == "__main__":
    

    resnet18_custom = ResNet18_custom_class_number(
        num_classes=10,
        project_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        use_pretrained=True,
        model_cache_path=os.path.join("weights", "resnet18"),
        verbose=True
    )

    print(resnet18_custom.resnet18)