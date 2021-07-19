import os
import argparse
import cv2
# from torch import random
import random
# from torch.nn.modules.loss import L1Loss
# from torch.utils import data
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split,Subset
from model.face_reenactor_network import Pix2Pix, PatchGan, VGGPerceptualLoss,TVLoss
from model.resunet_network import ResUnet
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief

from dataset_tool import MyDataset, BaseDataSet


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


def visual_data(epoch, step, x1, x2, pre_y,pre_map, y,y_map):
    logdir = f"log/{args.out_checkpoint}/{epoch}"
    os.makedirs(logdir, exist_ok=True)
    for idx, (input_x1, input_x2, out_y, real_y) in enumerate(zip(x1, x2, pre_y, y)):
        step_idx = step#*16+idx
        x1_img = cv2.cvtColor((input_x1.cpu().numpy().transpose(
            1, 2, 0)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        x2_img = (input_x2.cpu().numpy().transpose(
            1, 2, 0)*255).astype(np.uint8)
        y_pred_img = (out_y.cpu().numpy().transpose(
            1, 2, 0)*255).astype(np.uint8)
        y_img = (real_y.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        result = np.concatenate([x1_img, x2_img, y_pred_img, y_img], axis=1)
        resultpath = f"{logdir}/{step_idx}.png"
        cv2.imwrite(resultpath, result)
        return


def cal_discriminator_loss(x, y, y_pred):
    real_d = discriminator_out(x, y).squeeze()
    fake_d = discriminator_out(x, y_pred).squeeze()
    real = torch.ones(real_d.size(), device=device)
    fake = torch.zeros(fake_d.size(), device=device)
    real_loss = BCELoss(real_d, real)
    fake_loss = BCELoss(fake_d, fake)
    return real_loss,fake_loss,real_d,fake_d


def discriminator_out(x, y):
    input = torch.cat([x, y], 1)
    out = discriminator(input)
    return out


def train(train_dataloader,val_dataloader):
    step_epoch = len(train_dataloader)
    if args.load_checkpoint:
        load_checkpoint(generator, discriminator)
    for epoch in range(args.max_epoch):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        print('Starting Epoch: {}'.format(epoch+1))
        running_l1_loss, running_perceptual_loss, running_ad_loss,running_ce_loss,running_tv_loss,running_d_loss = 0.,0., 0., 0., 0.,0.
        prog_bar = tqdm(train_dataloader)
        for step, data in enumerate(prog_bar):
            now_step=(epoch*step_epoch)+(step+1)
            x1 = data[0].to(device)
            x2 = data[1].to(device)
            y = data[2].to(device)
            y_map=data[3].to(device)
            x = torch.cat((x1, x2), 1)
            y_pred,map = generator(x)
            #計算D loss
            real_loss,fake_loss,real_d,fake_d = cal_discriminator_loss(x, y, y_pred.detach())
            d_loss=real_loss+fake_loss
            d_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()
            #計算G loss
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_pred=discriminator_out(x, y_pred).squeeze()
            real_y=torch.ones(d_pred.size(),device=device)
            ad_loss = BCELoss(d_pred,real_y)
            celoss=BCELoss(map,y_map)
            tvloss=TVloss(y_pred)
            g_loss = w1*l1loss+w2*perceptual_loss+w3*ad_loss+w4*celoss+tvloss
            g_loss.backward()

            if step%100==0:
                writer.add_scalar("TrainLoss/Generator/L1Loss",l1loss,now_step)
                writer.add_scalar("TrainLoss/Generator/PerceptualLoss",perceptual_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/AdversarialLoss",ad_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/CrossEntropyLoss",celoss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/real",real_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/fake",fake_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/all",d_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/TVLoss",tvloss,now_step)
                grid_input_x1=make_grid(x1)
                grid_input_x2=make_grid(x2[:,[2,1,0],:,:])
                grid_input=make_grid(x[:,[0,3,2,1],:,:])
                grid_output=make_grid(y_pred[:,[2,1,0],:,:])
                grid_output_map=make_grid(map)
                grid_target=make_grid(y[:,[2,1,0],:,:])
                grid_target_map=make_grid(y_map)
                grid_real_d=make_grid(real_d.unsqueeze(1))
                grid_fake_d=make_grid(fake_d.unsqueeze(1))
                grid_fake_g=make_grid(d_pred.unsqueeze(1))
                writer.add_image("input/x1",grid_input_x1,now_step)
                writer.add_image("input/x2",grid_input_x2,now_step)
                writer.add_image("input/image",grid_input,now_step)
                writer.add_image("output/map",grid_output_map,now_step)
                writer.add_image("output/image",grid_output,now_step)
                writer.add_image("discriminator/real",grid_real_d,now_step)
                writer.add_image("discriminator/fake",grid_fake_d,now_step)
                writer.add_image("discriminator/fake_G",grid_fake_g,now_step)
                writer.add_image("target/map",grid_target_map,now_step)
                writer.add_image("target/img",grid_target,now_step)
                save_checkpoint(generator, discriminator)
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ad_loss+=ad_loss.item()
            running_ce_loss += celoss.item()
            running_tv_loss+=tvloss.item()
            running_d_loss += d_loss.item()
            next_step = step+1
            prog_bar.set_description('L1:{:0.4f},PLoss:{:0.4f},ADLoss:{:0.4f},CELoss:{:0.4f},TVLoss:{:0.4},DLoss:{:0.4f}'.format(
                running_l1_loss / next_step, running_perceptual_loss / next_step,running_ad_loss/next_step,
                running_ce_loss/next_step,running_tv_loss/next_step,running_d_loss / next_step,))
            prog_bar.update()
            if step % args.backward_ratio == 0:
                g_optimizer.step()
                g_optimizer.zero_grad()
        eval(epoch, val_dataloader)
        save_checkpoint(generator, discriminator)
    writer.close()

def eval(epoch, val_dataloader):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        running_l1_loss, running_perceptual_loss, running_ad_loss,running_ce_loss,running_tv_loss,running_d_loss =0., 0., 0., 0., 0.,0.
        all_step = len(val_dataloader)
        print(all_step)
        for step, data in enumerate(val_dataloader):
            now_step=(epoch*all_step)+(step+1)
            x1 = data[0].to(device)
            x2 = data[1].to(device)
            y = data[2].to(device)
            y_map=data[3].to(device)
            x = torch.cat([x1, x2], dim=1)
            y_pred,y_pre_map = generator(x)
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_pred=discriminator_out(x, y_pred).squeeze()
            real_y=torch.ones(d_pred.size(),device=device)
            ad_loss = BCELoss(d_pred,real_y)
            ce_loss = BCELoss(y_pre_map,y_map)
            tvloss=TVloss(y_pred)
            real_loss,fake_loss,real_d,fake_d = cal_discriminator_loss(x, y, y_pred.detach())
            d_loss = real_loss+fake_loss
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ad_loss+=ad_loss.item()
            running_ce_loss += ce_loss.item()
            running_tv_loss+=tvloss.item()
            running_d_loss += d_loss.item()
            visual_data(epoch, step, x1, x2, y_pred,y_pre_map, y,y_map)
            if step%100==0:
                writer.add_scalar("ValLoss/Generator/L1Loss",l1loss,now_step)
                writer.add_scalar("ValLoss/Generator/PerceptualLoss",perceptual_loss,now_step)
                writer.add_scalar("ValLoss/Generator/AdversarialLoss",ad_loss,now_step)
                writer.add_scalar("ValLoss/Generator/CrossEntropyLoss",ce_loss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/real",real_loss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/fake",fake_loss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/all",d_loss,now_step)
                writer.add_scalar("ValLoss/Generator/TVLoss",tvloss,now_step)
        print('EVAL| L1:{:0.4f},PLoss:{:0.4f},ADLoss:{:0.4f},CELoss:{:0.4f},DLoss:{:0.4f}'.format(
            running_l1_loss / all_step, running_perceptual_loss / all_step, running_ad_loss/ all_step,
            running_ce_loss/all_step,running_d_loss / all_step))

def work_init_fn(worker_id):
    works_seed=torch.initial_seed()%2**32
    np.random.seed(works_seed)
    random.seed(works_seed)
    torch.manual_seed(works_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int)
    parser.add_argument('--resize', dest='resize', type=int)
    parser.add_argument("--backward_ratio", dest="backward_ratio", type=int)
    parser.add_argument("--max_epoch", dest="max_epoch", type=int)
    parser.add_argument("--learning_rate", dest="lr", default=2e-4, type=float)
    parser.add_argument("--out_checkpoint", dest="out_checkpoint", type=str)
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generator = Pix2Pix(4, 3).to(device=device)  # x1=1c,x2=3c
   
    generator = ResUnet(4, 3).to(device=device)
    discriminator = PatchGan(7).to(device=device)  # x1=1c,x2=3c,y=3c
    # g_optimizer = optim.Adam(generator.parameters(),
    #                          lr=args.lr, betas=(0.5, 0.999))
    # d_optimizer = optim.Adam(discriminator.parameters(),
    #                          lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = AdaBelief(generator.parameters(),
                             lr=args.lr, eps=1e-16 , betas=(0.5, 0.999), weight_decouple=True, rectify=True,print_change_log=False)
    d_optimizer = AdaBelief(discriminator.parameters(),
                             lr=args.lr, eps=1e-16 , betas=(0.5, 0.999), weight_decouple=True, rectify=True,print_change_log=False)
    vgg_perceptual_loss = VGGPerceptualLoss().to(device)
    BCELoss = nn.BCELoss().to(device)
    L1loss = nn.L1Loss().to(device)
    TVloss=TVLoss(2)
    w1, w2, w3, w4 = 100, 100,1, 1
    transform = transforms.Compose([transforms.ToTensor()])
    origin_dataset=BaseDataSet(
        args.input_dir, args.resize, transform=transform)
    train_len = int(len(origin_dataset)*0.8)
    train_dataset,val_dataset=Subset(origin_dataset,[i for i in range(train_len)]),Subset(origin_dataset,[i for i in range(train_len,len(origin_dataset))])
    print(f"all:{len(origin_dataset)},train:{len(train_dataset)},val:{len(val_dataset)}")
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=4)#,worker_init_fn=work_init_fn)
    val_dataloader=DataLoader(
        val_dataset,args.batch_size, shuffle=True, drop_last=True, num_workers=4)#,worker_init_fn=work_init_fn)
    writer=SummaryWriter()
    train(train_dataloader,val_dataloader)
