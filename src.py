import os
import numpy as np
import pandas as pd
import math
import openslide
import h5py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms

from torcheval.metrics import MulticlassAccuracy, BinaryAccuracy
from sklearn.metrics import accuracy_score
from ctran import ctranspath

def init_backbone(name, finetuning, num_classes):
    IN_CHNS = 3
    if name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(IN_CHNS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.fc = nn.Linear(512, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "layer2"
        elif finetuning == "mid":
            freeze_point = "layer3"
        elif finetuning == "shallow":
            freeze_point = "layer4"
        
    elif name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(IN_CHNS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.fc = nn.Linear(2048, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "layer2"
        elif finetuning == "mid":
            freeze_point = "layer3"
        elif finetuning == "shallow":
            freeze_point = "layer4"
        
    elif name == "densenet":
        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        backbone.features.conv0 = nn.Conv2d(IN_CHNS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.classifier = nn.Linear(1024, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "denseblock2"
        elif finetuning == "mid":
            freeze_point = "denseblock3"
        elif finetuning == "shallow":
            freeze_point = "denseblock4"
        
    elif name == "mobilenet":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.features[0][0] = nn.Conv2d(IN_CHNS, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        backbone.classifier[-1] = nn.Linear(1024, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "features.3"
        elif finetuning == "mid":
            freeze_point = "features.6"
        elif finetuning == "shallow":
            freeze_point = "features.9"

    elif name == "vit":
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        backbone.conv_proj = nn.Conv2d(IN_CHNS, 768, kernel_size=(16, 16), stride=(16, 16))
        backbone.heads.head = nn.Linear(768, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "encoder_layer_3"
        elif finetuning == "mid":
            freeze_point = "encoder_layer_6"
        elif finetuning == "shallow":
            freeze_point = "encoder_layer_9"

    elif name == "swin":
        backbone = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
        backbone.features[0][0] = nn.Conv2d(IN_CHNS, 96, kernel_size=(4, 4), stride=(4, 4))
        backbone.head = nn.Linear(768, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "features.2"
        elif finetuning == "mid":
            freeze_point = "features.5.8"
        elif finetuning == "shallow":
            freeze_point = "features.6"
    
    elif name == "ctp":
        backbone = ctranspath()
        backbone.head = nn.Identity()
        backbone.load_state_dict(torch.load(args.ckpt_path)["model"], strict=True)
        backbone.head = nn.Linear(768, num_classes, bias=True)
        if finetuning == "deep":
            freeze_point = "layers.2"
        elif finetuning == "mid":
            freeze_point = "layers.3"
        elif finetuning == "shallow":
            freeze_point = "head"
    
    if finetuning:
        freezing = True
        for name, param in backbone.named_parameters():
            if freeze_point in name:
                freezing = False
            if freezing:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return backbone

class PatchIDCounter:
    def __init__(self, df, shuffle=False, cap=None):
        self.df = df
        self.shuffle = shuffle
        self.cap = cap
        self.mapping = self._compute_mapping()

    def _compute_mapping(self):
        np.random.seed(7)
        mapping = []
        for _, row in self.df.iterrows():
            slide_id = row['slide_id']
            nb_of_patches = row['nb_patches']
            
            # Determine the patch indices we will use
            if self.cap and nb_of_patches > self.cap:
                selected_patch_indices = np.random.choice(nb_of_patches, self.cap, replace=False)
            else:
                selected_patch_indices = range(nb_of_patches)
            
            for patch_id in selected_patch_indices:
                mapping.append((slide_id, patch_id))

        if self.shuffle:
            np.random.shuffle(mapping)
        return mapping

    def next_id(self, idx):
        return self.mapping[idx]


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, h5_dir, df, label_dict, image_size, patch_size, cap=100, augmentation=False, training=False, ext=".tif"):
        if augmentation and training:  
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.data_dir = data_dir
        self.h5_dir = h5_dir
        self.df = df
        self.label_dict = label_dict
        self.ext = ext
        self.patch_size = (patch_size, patch_size)
        
        self.counter = PatchIDCounter(df, shuffle=training, cap=cap)

    def __len__(self):
        return len(self.counter.mapping)
        # return 10
    def __getitem__(self, idx):
        slide_id, patch_id = self.counter.next_id(idx)
        label = self.df["label"][self.df["slide_id"] == slide_id].item()
        
        wsi = openslide.open_slide(os.path.join(self.data_dir, slide_id + self.ext))
        with h5py.File(os.path.join(self.h5_dir, slide_id + ".h5"), "r") as hf:
            c = np.array(hf["coords"])[patch_id]
        patch = wsi.read_region(c, 0, self.patch_size).convert("RGB")
        return self.transform(patch), np.array(self.label_dict[label])[np.newaxis]

def k_fold_split(df, k_fold=5, val_ratio=None):
    splits = {k: {} for k in range(k_fold)}
    val_ratio = 1/k_fold if val_ratio is None else val_ratio
    for k in range(k_fold):
        np.random.seed(k)
        train_cases, val_cases = [], []
        for v, c in enumerate(pd.unique(df["label"])):
            cases = np.unique(df["case_id"][df["label"] == c].values)
            cases = np.random.permutation(cases)
            [val_cases.append(i) for i in cases[:int(len(cases)*val_ratio)]]
            [train_cases.append(i) for i in cases[int(len(cases)*val_ratio):]]

        splits[k]["train"] = df[df["case_id"].isin(train_cases)]
        splits[k]["val"] = df[df["case_id"].isin(val_cases)]
    return splits


class Trainer():
    def __init__(
        self,
        run_dir,
        train_df,
        val_df,
        label_dict,
        cv,
        data_dir, h5_dir,
        training_csv,
        testing_csv=None,
        train_percentage=1.,
        backbone="resnet50",
        finetuning="shallow",
        optimizer="Adam",
        lr=1e-3,
        l2_reg=1e-5,
        earlystopping=20,
        image_size=512,
        ext=".tif", patch_size=224, max_patches=100,
        augmentation=True,
        batch_size=16,
        val_batch_size=16,
        epochs=100, 
        load_from=None,
        is_ddp=False,
        rank=0,
        world_size=1,
        pin_memory=True, 
        num_workers=8,
        **kwargs
    ):
        self.is_ddp = is_ddp
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.run_dir = run_dir
        self.cv = cv
        self.num_classes = len(label_dict)
        self.rev_label_dict = {v: k for k, v in label_dict.items()}

        self.epochs = epochs
        self.patience = earlystopping
        self.device = torch.device(f"cuda:{self.rank}") if torch.cuda.is_available() else "cpu"
        self.model = init_backbone(backbone, finetuning, self.num_classes if self.num_classes>2 else 1)
        self.model.to(self.device)
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
        
        if load_from:
            self.model.load_state_dict(torch.load(load_from))
            print("Model is loaded from", load_from)

        if len(label_dict) == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer == "Adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=l2_reg)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(params, lr=lr, momentum=.9, nesterov=True, weight_decay=l2_reg)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5, verbose=True)

        train_data = PatchDataset(data_dir, h5_dir, train_df, label_dict, image_size, patch_size, cap=max_patches, augmentation=augmentation, training=True, ext=ext)
        val_data = PatchDataset(data_dir, h5_dir, val_df, label_dict, image_size, patch_size, cap=max_patches, ext=ext)
        
        train_sampler = DistributedSampler(train_data, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        val_sampler = DistributedSampler(val_data, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None

        torch.multiprocessing.set_sharing_strategy('file_system')
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=math.ceil(batch_size / self.world_size), shuffle=not self.is_ddp, pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=math.ceil(val_batch_size / self.world_size), shuffle=not self.is_ddp, pin_memory=pin_memory, num_workers=num_workers, sampler=val_sampler)
        print("Training on: ", len(self.train_loader))
        print("Validation on: ", len(self.val_loader))
        if testing_csv:
            test_df = pd.read_csv(testing_csv)
            test_data = PatchDataset(data_dir, h5_dir, test_df, label_dict, image_size, patch_size, cap=max_patches, ext=ext)
            self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
            self.testing = True
            print("Testing on: ", len(self.test_loader))
        else:
            self.testing = False
        
        self.log = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "train_patch_acc": [],
            "val_loss": [],
            "val_patch_acc": [],
        }
        self.best_loss, self.trigger = np.inf, 0

    def one_epoch(self, dataloader, training=True):
        running_loss = 0.
        if self.num_classes > 2:
            acc_meter = MulticlassAccuracy(device=self.device, num_classes=self.num_classes, average="macro")
        else:
            acc_meter = BinaryAccuracy(device=self.device)
        self.model.train(training)
        for idx, (img, label) in enumerate(dataloader):
            if idx % 50 == 0:
                print("\tStep: {}/{}" .format(idx, len(dataloader)))
            img, label = img.to(self.device), label.to(self.device)
            
            with torch.set_grad_enabled(training):
                logits = self.model(img)
                loss = self.criterion(logits.float(), label.float())
            running_loss += loss.detach().cpu().item()

            if self.num_classes > 2:
                pred = F.log_softmax(logits, dim=1).argmax(1)
            else:
                pred = torch.squeeze(torch.sigmoid(logits))
                label = torch.squeeze(label)
            acc_meter.update(pred, label)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return {
            "loss": running_loss / len(dataloader), 
            "patch_acc": acc_meter.compute().detach().cpu().item(),
        }

    def run(self):
        for epoch in range(self.epochs):
            print("\nEpoch: {}/{}" .format(epoch, self.epochs))
            train_log = self.one_epoch(self.train_loader, training=True)
            print("Train loss: {:.4f} | acc: {:.4f}" .format(train_log["loss"], train_log["patch_acc"]))
            val_log = self.one_epoch(self.val_loader, training=False)
            print("Validation loss: {:.4f} | acc: {:.4f}" .format(val_log["loss"], val_log["patch_acc"]))
           
            cont_training = self.log_results(epoch, train_log, val_log)
            if not cont_training:
                return
            self.scheduler.step(val_log["loss"])
        return

    def evaluate(self):
        if self.is_ddp:
            dist.barrier()
        state_dict = torch.load(
            os.path.join(self.run_dir, "best_model_cv{}.pt".format(self.cv)), 
            map_location={'cuda:%d' % 0: 'cuda:%d' % self.rank}
        )
        self.model.load_state_dict(state_dict)
        val_log = self.one_epoch(self.val_loader, training=False)
        print("Validation loss: {:.4f} | acc: {:.4f}" .format(val_log["loss"], val_log["patch_acc"]))
        test_log = self.one_epoch(self.test_loader, training=False)
        print("Test loss: {:.4f} | acc: {:.4f}" .format(test_log["loss"], test_log["patch_acc"]))
        return {"val_loss": val_log["loss"],
                "val_acc": val_log["patch_acc"],
                "test_loss": test_log["loss"],
                "test_acc": test_log["patch_acc"]
                }

    def log_results(self, epoch, train_log, val_log):
        self.log['epoch'].append(epoch)
        self.log['lr'].append(self.optimizer.param_groups[0]['lr'])
        for k, v in self.log.items():
            if "train" in k:
                v.append(train_log[k.replace("train_", "")])
            elif "val" in k:
                v.append(val_log[k.replace("val_", "")])
        pd.DataFrame(self.log).to_csv(os.path.join(self.run_dir, f"log_cv{self.cv}.csv"), index=False)

        if val_log["loss"] < self.best_loss:
            self.best_loss = val_log["loss"]
            self.trigger = 0
            if self.rank == 0:
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, f"best_model_cv{self.cv}.pt"))
        else:
            self.trigger += 1
            if self.patience > 0:
                if self.trigger >= self.patience:
                    print("Early stopping...")
                    return False
                else:
                    print('Early stopping count: {}/{}'.format(self.trigger, self.patience))
        
        return True
        