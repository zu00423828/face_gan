import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import BCELoss
from model.face_reenactor_network import Unet,PatchGan,VGGPerceptualLoss
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
import argparse
from datasettool import MyDataset,BaseDataSet

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


def cal_discriminator_loss(x,y,y_pred):
        real=discriminator_out(x,y)
        fake=discriminator_out(x,y_pred)
        real_loss=BCELoss(real,y)
        fake_loss=BCELoss(fake,y_pred)
        return real_loss+fake_loss
def discriminator_out(x,y):
        input=torch.cat([x,y],1)
        out=discriminator(input)
        return out


def Train():
    transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = MyDataset("combined/train", transform=transform)
    train_dataset= BaseDataSet("photos",transform=transform)
    train_len=int(len(train_dataset)*0.8)
    val_len=len(train_dataset)-train_len
    train_dataset,val_dataset=random_split(train_dataset,[train_len,val_len])
    # val_dataset=BaseDataSet("phots",transform)
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, drop_last=True,num_workers=4)
    step_epoch = len(train_dataloader)
    for epoch in range(args.max_epochs):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        print('Starting Epoch: {}'.format(epoch+1))
        running_l1_loss, running_perceptual_loss, running_d_loss, running_l2_loss = 0., 0., 0., 0.
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
            l2loss=L2loss(y_pred,y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_loss = cal_discriminator_loss(x, y, y_pred)
            loss = L2loss+w3*perceptual_loss+w4*d_loss+w5*l1loss
            loss.backward()
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_d_loss += d_loss.item()
            next_step = step+1
            prog_bar.set_description('L1:{:0.4f},PLoss:{:0.4f},DLoss:{:0.4f},CELoss{:0.4f}'.format(
                    running_l1_loss / next_step,running_perceptual_loss / next_step,
                    running_d_loss / next_step,running_l2_loss/next_step,))
            prog_bar.update()
            if step%args.backward_ratio==0:
                g_optimizer.step()
                d_optimizer.step()
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
        eval(val_dataset)
        save_checkpoint(generator, discriminator)
def eval(val_dataset):
    generator.eval()
    discriminator.eval()
    testdataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        running_l1_loss, running_perceptual_loss, running_d_loss, running_l2loss= 0., 0., 0., 0.
        all_step=len(testdataloader)
        for step,data in enumerate(testdataloader):
            x=data[0].to(device)
            y=data[1].to(device)
            y_pred=generator(x)
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_loss = cal_discriminator_loss(x, y, y_pred)
            l2loss = L2loss(y_pred, y)
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_d_loss += d_loss.item()
            running_l2loss += l2loss.item()
        print('EVAL| L1:{:0.4f},PLoss:{:0.4f},DLoss:{:0.4f},CELoss{:0.4f}'.format(
                    running_l1_loss / all_step,running_perceptual_loss / all_step,
                    running_d_loss / all_step,running_l2loss/all_step))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int)
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
    L2loss=nn.MSELoss().to(device)
    w3, w4,w5 =33,33,33 
    Train()