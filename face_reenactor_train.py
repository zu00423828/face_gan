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

from torch.utils.data import DataLoader
from model.face_reenactor_network import Pix2Pix, PatchGan, VGGPerceptualLoss,TVLoss
from model.resunet_network import ResUnet
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief
from dataset_tool import  BaseDataSet,TestDataSet
from model.ssim import SSIM

def save_checkpoint(model_G, model_D):
    os.makedirs(f'checkpoint/{args.out_checkpoint}', exist_ok=True)
    G_model_path = os.path.join("checkpoint",args.out_checkpoint, "frG.pth")
    D_model_path = os.path.join("checkpoint",args.out_checkpoint, "frD.pth")
    torch.save(model_G.state_dict(), G_model_path)
    torch.save(model_D.state_dict(), D_model_path)
    G_optim_path=os.path.join("checkpoint",args.out_checkpoint,"opt_G.pth")
    D_optim_path=os.path.join("checkpoint",args.out_checkpoint,"opt_D.pth")
    torch.save(g_optimizer.state_dict(),G_optim_path)
    torch.save(d_optimizer.state_dict(),D_optim_path)


def load_checkpoint(model_G, model_D):
    checkpoint_root = args.load_checkpoint
    G_model_path = os.path.join("checkpoint",checkpoint_root, "frG.pth")
    D_model_path = os.path.join("checkpoint",checkpoint_root, "frD.pth")
    model_G.load_state_dict(torch.load(G_model_path))
    model_D.load_state_dict(torch.load(D_model_path))
    G_optim_path=os.path.join("checkpoint",checkpoint_root,"opt_G.pth")
    D_optim_path=os.path.join("checkpoint",checkpoint_root,"opt_D.pth")
    g_optimizer.load_state_dict(torch.load(G_optim_path))
    d_optimizer.load_state_dict(torch.load(D_optim_path))

def visual_data(epoch, step, x1, x2, pre_y,y):
    logdir = f"log/{args.out_checkpoint}/{epoch}"
    os.makedirs(logdir, exist_ok=True)
    for idx, (input_x1, input_x2, out_y, real_y) in enumerate(zip(x1, x2, pre_y, y)):
        step_idx = step#*16+idx
        # x1_img = cv2.cvtColor((input_x1.cpu().numpy().transpose(
        #     1, 2, 0)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        x1_img = (input_x1.cpu().numpy().transpose(
            1, 2, 0)*255).astype(np.uint8)
        x2_img = (input_x2.cpu().numpy().transpose(
            1, 2, 0)*255).astype(np.uint8)
        y_pred_img = (out_y.cpu().numpy().transpose(
            1, 2, 0)*255 ).astype(np.uint8)
        y_img = (real_y.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        result = np.concatenate([x1_img, x2_img, y_pred_img, y_img], axis=1)
        resultpath = f"{logdir}/{step_idx}.png"
        cv2.imwrite(resultpath, result)
        return


def cal_discriminator_loss(x, y, y_pred):
    real_d = discriminator_out(x, y)#.squeeze()
    fake_d = discriminator_out(x, y_pred)#.squeeze()
    real = torch.ones(real_d.size(), device=device)
    fake = torch.zeros(fake_d.size(), device=device)
    real_loss = BCELoss(real_d, real)
    fake_loss = BCELoss(fake_d, fake)
    return real_loss,fake_loss


def discriminator_out(x, y):
    input = torch.cat([x, y], 1)
    # input=y
    out = discriminator(input)
    return out


def train(train_dataloader,val_dataloader):
    step_epoch = len(train_dataloader)
    if args.load_checkpoint:
        load_checkpoint(generator, discriminator)
    # eval(0,val_dataloader)
    for epoch in range(args.max_epoch):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        print('Starting Epoch: {}'.format(epoch+1))
        running_l1_loss, running_perceptual_loss, running_ad_loss,running_ce_loss,running_d_loss = 0.,0., 0., 0., 0.#,0.,
        running_ssim_loss=0.
        prog_bar = tqdm(train_dataloader,position=0,leave=True)
        for step, data in enumerate(prog_bar):
            now_step=(epoch*step_epoch)+(step+1)
            # x1 = data[0].to(device)
            # x2 = data[1].to(device)
            # y = data[2].to(device)
            # y_map=data[3].to(device)
            # x = torch.cat((x1, x2), 1)
            x = data[0].to(device)
            y = data[1].to(device)
            y_map=data[2].to(device)
            # x2=data[-1].to(device)
   



            y_pred,map = generator(x)#,x2)
            #計算D loss
            real_loss,fake_loss= cal_discriminator_loss(x, y, y_pred.detach())
            d_loss=(real_loss+fake_loss)
            d_loss.backward()
            # d_optimizer.step()
            # d_optimizer.zero_grad()
            #計算G loss
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_pred=discriminator_out(x, y_pred).squeeze()
            real_y=torch.ones(d_pred.size(),device=device)
            ad_loss = BCELoss(d_pred,real_y)
            celoss=BCELoss(map,y_map)
            ssimloss=1-SSIMloss(y_pred,y)
            g_loss = w1*l1loss+w2*perceptual_loss+w3*ad_loss+w4*celoss+w5*ssimloss
            g_loss.backward()

            if step%100==0:
                writer.add_scalar("TrainLoss/Generator/L1Loss",l1loss,now_step)
                writer.add_scalar("TrainLoss/Generator/PerceptualLoss",perceptual_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/AdversarialLoss",ad_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/CrossEntropyLoss",celoss,now_step)
                writer.add_scalar("TrainLoss/Generator/SSIMLoss",ssimloss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/real",real_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/fake",fake_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/all",d_loss,now_step)
                # writer.add_scalar("TrainLoss/Generator/TVLoss",tvloss,now_step)
                grid_input_x1=make_grid(x[:,[2,1,0],:,:])
                grid_input_x2=make_grid(x[:,[5,4,3],:,:])
                # grid_input=make_grid(x[:,[2,1,0],:,:])
                # grid_input=make_grid(x[:,[0,3,2,1],:,:])
                grid_output=make_grid(y_pred[:,[2,1,0],:,:])
                grid_output_map=make_grid(map)
                grid_target=make_grid(y[:,[2,1,0],:,:])
                grid_target_map=make_grid(y_map)
                writer.add_image("input/x1",grid_input_x1,now_step)
                writer.add_image("input/x2",grid_input_x2,now_step)
                # writer.add_image("input/image",grid_input,now_step)
                writer.add_image("output/map",grid_output_map,now_step)
                writer.add_image("output/image",grid_output,now_step)
                writer.add_image("target/map",grid_target_map,now_step)
                writer.add_image("target/img",grid_target,now_step)
                save_checkpoint(generator, discriminator)
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ad_loss+=ad_loss.item()
            running_ce_loss += celoss.item()
            # running_tv_loss+=tvloss.item()
            running_ssim_loss+=ssimloss.item()
            running_d_loss += d_loss.item()
            next_step = step+1
            # prog_bar.set_description('L1:{:0.4f},PLoss:{:0.4f},ADLoss:{:0.4f},CELoss:{:0.4f},TVLoss:{:0.4f},DLoss:{:0.4f}'.format(
            #     running_l1_loss / next_step, running_perceptual_loss / next_step,running_ad_loss/next_step,
            #     running_ce_loss/next_step,running_tv_loss/next_step,running_d_loss / next_step,))
            prog_bar.set_description('L1:{:0.4f},P:{:0.4f},AD:{:0.4f},CE:{:0.4f},SSIM:{:0.4f},D:{:0.4f}'.format(
                running_l1_loss / next_step, running_perceptual_loss / next_step,running_ad_loss/next_step,
                running_ce_loss/next_step,running_ssim_loss/next_step,running_d_loss / next_step))
            # prof.step()
            if step % args.backward_ratio == 0:
                d_optimizer.step()
                d_optimizer.zero_grad()
                g_optimizer.step()
                g_optimizer.zero_grad()
            # g_optimizer.step()
            # g_optimizer.zero_grad()
        eval(epoch, val_dataloader)
        save_checkpoint(generator, discriminator)
    writer.close()

def eval(epoch, val_dataloader):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        running_l1_loss, running_perceptual_loss, running_ad_loss,running_ce_loss,running_d_loss =0., 0., 0., 0., 0.
        running_ssim_loss=0.
        all_step = len(val_dataloader)
        print("EVAL")
        pro_bar=tqdm(val_dataloader)
        # for step, data in enumerate(val_dataloader):
        for step, data in enumerate(pro_bar):
            now_step=(epoch*all_step)+(step+1)
            # x1 = data[0].to(device)
            # x2 = data[1].to(device)
            # y = data[2].to(device)
            # y_map=data[3].to(device)
            # x = torch.cat([x1, x2], dim=1)

            x = data[0].to(device)
            y = data[1].to(device)
            y_map=data[2].to(device)

            y_pred,y_pre_map = generator(x)
            l1loss = L1loss(y_pred, y)
            perceptual_loss = vgg_perceptual_loss(y_pred, y)
            d_pred=discriminator_out(x, y_pred).squeeze()
            real_y=torch.ones(d_pred.size(),device=device)
            ad_loss = BCELoss(d_pred,real_y)
            ce_loss = BCELoss(y_pre_map,y_map)
            # tvloss=TVloss(y_pred)
            ssimloss=SSIMloss(y_pred,y)
            real_loss,fake_loss = cal_discriminator_loss(x, y, y_pred.detach())
            d_loss = (real_loss+fake_loss)
            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ad_loss+=ad_loss.item()
            running_ce_loss += ce_loss.item()
            running_ssim_loss+=ssimloss.item()
            # running_tv_loss+=tvloss.item()
            running_d_loss += d_loss.item()
            if step%100==0:
                writer.add_scalar("ValLoss/Generator/L1Loss",l1loss,now_step)
                writer.add_scalar("ValLoss/Generator/PerceptualLoss",perceptual_loss,now_step)
                writer.add_scalar("ValLoss/Generator/AdversarialLoss",ad_loss,now_step)
                writer.add_scalar("ValLoss/Generator/CrossEntropyLoss",ce_loss,now_step)
                writer.add_scalar("ValLoss/Generator/SSIMLoss",ssimloss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/real",real_loss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/fake",fake_loss,now_step)
                writer.add_scalar("ValLoss/DiscriminatorLoss/all",d_loss,now_step)
                visual_data(epoch, step, x[:,0:3,:,:], x[:,3:,:,:], y_pred,y)
                # writer.add_scalar("ValLoss/Generator/TVLoss",tvloss,now_step)
        print('EVAL| L1:{:0.4f},PLoss:{:0.4f},ADLoss:{:0.4f},CELoss:{:0.4f},SSIM:{:0.4f},DLoss:{:0.4f}'.format(
            running_l1_loss / all_step, running_perceptual_loss / all_step, running_ad_loss/ all_step,
            running_ce_loss/all_step,running_ssim_loss/all_step,running_d_loss / all_step))

def work_init_fn(worker_id):
    # works_seed=torch.initial_seed()%2**32
    seed=torch.utils.data.get_worker_info.seed()%(2**32-1)
    # np.random.seed(works_seed)
    # random.seed(works_seed)
    # torch.manual_seed(works_seed)
    # seed=10
    seed+=worker_id
    # np.random.seed(seed)
    print(seed)

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
    parser.add_argument("--model", dest="model",default="resunet", type=str)
    parser.add_argument("--landmark_gray",dest="landmark_gray",default=False,type=int)
    parser.add_argument("--log",dest="log",default=None,type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.enabled=True
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.deterministic=True
    G_input_channel,D_input_channel=(6,9) if not args.landmark_gray else (4,7)
    if args.model=="resunet":
        # generator = ResUnet(6, 3).to(device=device)
        from model.resunet_network_new import ResUnet
        generator = ResUnet(G_input_channel, 3).to(device=device)
        # generator.apply(weight_init)
    else:
        generator = Pix2Pix(G_input_channel, 3).to(device=device)  # x1=1c,x2=3c
    discriminator = PatchGan(D_input_channel).to(device=device)  # x1=1c,x2=3c,y=3c
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
    # CELoss=nn.CrossEntropyLoss().to(device)
    L1loss = nn.L1Loss().to(device)
    # TVloss=TVLoss(2)
    SSIMloss=SSIM().to(device)
    w1, w2, w3, w4,w5 = 100, 100,1, 1,5
    # train_dir=os.path.join(args.input_dir,"train")
    # val_dir=os.path.join(args.input_dir,"val")
    train_dataset=TestDataSet(args.input_dir,"train",resize=args.resize,landmark_gray=args.landmark_gray)
    val_dataset=TestDataSet(args.input_dir,"val",resize=args.resize)
    print(f"train:{len(train_dataset)},val:{len(val_dataset)}")
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=4,pin_memory=True)#,worker_init_fn=work_init_fn)
    val_dataloader=DataLoader(
        val_dataset,args.batch_size, shuffle=True, drop_last=True, num_workers=4,pin_memory=True)#,worker_init_fn=work_init_fn)
    if args.log is None:
        writer=SummaryWriter()
    else:
        writer=SummaryWriter(f"runs/{args.log}")
    # prof= profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     # activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
    #     record_shapes=True,
    #     with_stack=True)

    train(train_dataloader,val_dataloader)
