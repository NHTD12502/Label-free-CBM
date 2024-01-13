import os
import torch
from torchvision import datasets, transforms, models

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test",
    "skincon_train":"content/drive/MyDrive/Colab_Notebooks/label_free/skincon/train",
    "skincon_val":"content/drive/MyDrive/Colab_Notebooks/label_free/skincon/val"
}

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt",
              "skincon":"data/skincon_class.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


#==============================================

MODEL_WEB_PATHS = {
'HAM10000_INCEPTION':'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
    'DEEPDERM':'https://drive.google.com/uc?id=1OLt11htu9bMPgsE33vZuDiU5Xe4UqKVJ'
}

# thresholds determined by maximizing F1-score on the test split of the train 
#   dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000_INCEPTION':0.733,
    'DEEPDERM':0.687
}

def load_model(backbone_name, save_dir="./models", download=True):
    # Taken from the DDI repo https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP
    """Load the model and download if necessary. Saves model to provided save 
    directory."""

    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{backbone_name.lower()}.pth")
    if not os.path.exists(model_path):
        if not download:
            raise Exception("Model not downloaded and download option not"\
                            " enabled.")
        else:
            # Requires installation of gdown (pip install gdown)
            import gdown
            gdown.download(MODEL_WEB_PATHS[backbone_name], model_path)
    model = torchvision.models.inception_v3(init_weights=False, pretrained=False, transform_input=True)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model._ddi_name = backbone_name
    model._ddi_threshold = MODEL_THRESHOLDS[backbone_name]
    model._ddi_web_path = MODEL_WEB_PATHS[backbone_name]
    return model


class InceptionBottom(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionBottom, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        self.layer = layer
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.features = nn.Sequential(*all_children[:until_layer])
        self.model = original_model

    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 model.x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 model.x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 model.x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 model.x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 model.x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192model. x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192model. x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256model. x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768model. x 17 x 17
        # N x 768model. x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 128model.0 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 204model.8 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 204model.8 x 8 x 8
        # Adaptivmodel.e average pooling
        x = self.model.avgpool(x)
        # N x 204model.8 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x


class InceptionTop(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionTop, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.layer = layer
        self.features = nn.Sequential(*all_children[until_layer:])
    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_derma_model(args, backbone_name="deepderm"):
    model = load_model(backbone_name.upper(), save_dir=args.out_dir)
    model = model.to("cuda")
    model = model.eval()
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top


#===============================================================================================

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
     
    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets

def get_target_model(target_name, device):
    
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name == 'deepderm':
        print('deepderm')
        model, backbone, model_top = get_derma_model(args, target_name)
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
        target_model = backbone
        
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess
