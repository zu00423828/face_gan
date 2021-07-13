import os
import argparse
import cv2
# from torch.nn.modules.loss import L1Loss
# from torch.utils import data
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
from model.face_reenactor_network import Unet, PatchGan, VGGPerceptualLoss,Unet2
from dataset_tool import MyDataset,BaseDataSet


def save_checkpoint(model_G, model_D):
    os.makedirs(args.out_checkpoint, exist_ok=True)
    G_model_path = os.path.join(args.out_checkpoint, "frG.pth")
    D_model_path = os.path.join(args.out_checkpoint, "frD.pth")
    torch.save(model_G.state_dict(), G_model_path)
    torch.save(model_D.state_dict(), D_model_path)


def load_checkpoint(model_G, model_D):
    checkpoint_root = args.load_checkpoint
    G_model_path = os.path.join(checkpoint_root, "frG.pth")
    D_model_path = os.path.join(checkpoint_root, "frD.pth")
    model_G.load_state_dict(torch.load(G_model_path))
    model_D.load_state_dict(torch.load(D_model_path))

def visual_data(epoch,step,x1,x2,pre_y,y):
    logdir=f"log/{args.out_checkpoint}/{epoch}"
    os.makedirs(logdir,exist_ok=True)
    for idx,(input_x1,input_x2,out_y,real_y) in enumerate(zip(x1,x2,pre_y,y)):
        step_idx=step*16+idx
        y_pred_img=(out_y.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        y_img=(real_y.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        result=np.concatenate([input_x1,input_x2,y_pred_img,y_img],axis=1)
        resultpath=f"{logdir}/{step_idx}.png"
        cv2.imwrite(resultpath,result)
def cal_discriminator_loss(x, y, y_pred):
    real = discriminator_out(x, y).squeeze()
    fake = discriminator_out(x, y_pred).squeeze()
    real_d = torch.ones(real.size(), device=device)
    fake_d = torch.zeros(fake.size(), device=device)
    real_loss = BCELoss(real, real_d)
    fake_loss = BCELoss(fake, fake_d)
    return real_loss+fake_loss


def discriminator_out(x, y):
    input = torch.cat([x, y], 1)
    out = discriminator(input)
    return out


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = MyDataset("combined/train", transform=transform)
    train_dataset= BaseDataSet(args.input_dir,args.resize,transform=transform)
    train_len=int(len(train_dataset)*0.8)
    val_len=len(train_dataset)-train_len
    train_dataset,val_dataset=random_split(train_dataset,[train_len,val_len])
    # val_dataset=BaseDataSet("phots",transform)
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, drop_last=True,num_workers=4)
    step_epoch = len(train_dataloader)
    if args.load_checkpoint:
        load_checkpoint(generator, discriminator)
    for epoch in range(args.max_epoch):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        print('Starting Epoch: {}'.format(epoch+1))
        running_l1_loss, running_perceptual_loss, running_d_loss, running_ce_loss = 0., 0., 0., 0.
        prog_bar = tqdm(train_dataloader)
        # for step, data in prog_bar:
        for step,data in enumerate(prog_bar):
            # g_optimizer.zero_grad()
            # d_optimizer.zero_grad()
            x1 = data[0].to(device)
            x2=data[1].to(device)
            y = data[2].to(device)
            x=torch.cat((x1,x2),1)
            y_pred = generator(x)
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_loss = cal_discriminator_loss(x, y, y_pred)
            celoss = CELoss(y_pred, y)
            loss = w1*l1loss+w2*perceptual_loss+w3*d_loss+w4*celoss
            loss.backward()
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_d_loss += d_loss.item()
            running_ce_loss += celoss.item()
            next_step = step+1
            prog_bar.set_description('L1:{:0.4f},PLoss:{:0.4f},DLoss:{:0.4f},CELoss{:0.4f}'.format(
                    running_l1_loss / next_step,running_perceptual_loss / next_step,
                    running_d_loss / next_step,running_ce_loss/next_step,))
            prog_bar.update()
            if step%args.backward_ratio==0:
                g_optimizer.step()
                d_optimizer.step()
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                # eval(epoch,val_dataset) 
        eval(epoch,val_dataset)
        save_checkpoint(generator, discriminator)
def eval(epoch,val_dataset):
    generator.eval()
    discriminator.eval()
    # transform = transforms.Compose([transforms.ToTensor()])
    # test_dataset = MyDataset("combined/val", transform=transform)
    testdataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        running_l1_loss, running_perceptual_loss, running_d_loss, running_ce_loss = 0., 0., 0., 0.
        all_step=len(testdataloader)
        print(all_step)
        for step,data in enumerate(testdataloader):
            x1=data[0].to(device)
            x2=data[1].to(device)
            y=data[2].to(device)
            x=torch.cat([x1,x2],dim=1)
            y_pred=generator(x)
            # print(x1.shape,x2.shape,y.shape,y_pred.shape)
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_loss = cal_discriminator_loss(x, y, y_pred)
            celoss = CELoss(y_pred, y)
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_d_loss += d_loss.item()
            running_ce_loss += celoss.item()
            visual_data(epoch,step,x1,x2,y_pred,y)
        print('EVAL| L1:{:0.4f},PLoss:{:0.4f},DLoss:{:0.4f},CELoss{:0.4f}'.format(
                    running_l1_loss / all_step,running_perceptual_loss / all_step,
                    running_d_loss / all_step,running_ce_loss/all_step))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int)
    parser.add_argument('--resize', dest='resize', type=int)
    parser.add_argument("--backward_ratio",dest="backward_ratio",type=int)
    parser.add_argument("--max_epoch", dest="max_epoch", type=int)
    parser.add_argument("--learning_rate", dest="lr", default=2e-4, type=float)
    parser.add_argument("--out_checkpoint", dest="out_checkpoint", type=str)
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Unet(4,3).to(device=device)#x1=1c,x2=3c
    discriminator = PatchGan(7).to(device=device)#x1=1c,x2=3c,y=3c
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    vgg_perceptual_loss = VGGPerceptualLoss().to(device)
    BCELoss = nn.BCELoss().to(device)
    CELoss = nn.BCEWithLogitsLoss().to(device)
    L1loss = nn.L1Loss().to(device)
    w1, w2, w3, w4 = 1,10,10,10 
    train()
