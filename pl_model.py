import pytorch_lightning as pl
import torch.optim as optim
import torch
from torchvision.models import resnet18
import torch.nn as nn

class cfg:
    img_size=256
    max_epochs=100
    model_name = "resnet18"
    patience = [5,2]
    factor= .1
    folds=5
    min_lr=1e-8

class MTL_Loss(nn.Module):
    def __init__(self,task_num):
        super(MTL_Loss,self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)))
    def forward(self,pred_age,pred_gend,tar_gend,tar_age):
        

        loss0 = nn.functional.binary_cross_entropy_with_logits(pred_gend,tar_gend)
        loss1 = nn.functional.mse_loss(pred_age,tar_age)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        return loss0+loss1


class AgeGendModel(nn.Module):
    def __init__(self):
        super(AgeGendModel,self).__init__()
        if 'resnet18' in cfg.model_name:
            self.model = eval(cfg.model_name)(pretrained=False)
            for params in self.model.parameters():
                params.require_grad = True
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
            self.clf = nn.Linear(512,1)
            self.reg = nn.Linear(512,1)
    
    def forward(self,x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        #print(x.shape)
        #x = self.lin1(x)
        #x = self.lin2(x)

        gend = self.clf(x)
        age = torch.sigmoid(self.reg(x))


        return gend,age


class AgeGendNet(pl.LightningModule):
    def __init__(self):
        super(AgeGendNet,self).__init__()

        self.model = AgeGendModel()
        

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        op = optim.Adam(self.model.parameters(),lr=0.01)
        
        scheduler = {
            'scheduler':optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=op,T_0=10),
            'monitor':'val_loss',
            'interval':'epoch',
            'frequency':1,
            'strict':True
        }

        self.op = op
        self.scheduler = scheduler
        
        return [op],[scheduler]

    def training_step(self,batch,batch_idx):
        y_hat_gend,y_hat_age = self.model(batch['image'])
        loss_tr = mtl(pred_gend=y_hat_gend,pred_age=y_hat_age,tar_gend=batch['gender'],tar_age=batch['age'])
        #f1_tr = torchmetrics.functional.accuracy(y_hat_gend.sigmoid(),batch['gender']) 
        #mae_tr = torchmetrics.functional.mean_absolute_error(y_hat_age.relu(),batch['age'])
        self.log("TrainLoss",loss_tr,prog_bar=True,on_step=False,on_epoch=True)
        return loss_tr
    
    def validation_step(self,batch,batch_idx):
        y_hat_gend,y_hat_age = self.model(batch['image'])
        loss_val = mtl(pred_gend=y_hat_gend,pred_age=y_hat_age,tar_gend=batch['gender'],tar_age=batch['age'])
        #f1_val = torchmetrics.functional.accuracy(y_hat_gend.sigmoid(),batch['gender'])
        #mae_val = torchmetrics.functional.mean_absolute_error(y_hat_age.relu(),batch['age'])
        self.log("val_loss",loss_val,prog_bar=True,on_step=False,on_epoch=True)
        return loss_val