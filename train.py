import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
from omegaconf import OmegaConf
from pytorch_msssim import ssim
import wandb
from torchvision.utils import make_grid

from nvc1b_model import NVC1B  # Import your model
from video_dataset import VideoDataset  # Import your dataset

class CombinedL1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super(CombinedL1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        ssim_loss = 1 - ssim(x, y, data_range=1, size_average=True)
        l1_loss = self.l1_loss(x, y)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

class NVC1BTrainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()
        self.model = NVC1B(config.model)
        self.train_dataset = VideoDataset(config.data.train_path)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.training.batch_size, shuffle=True)
        self.criterion = CombinedL1SSIMLoss(alpha=config.training.alpha)
        
        self.model, self.train_dataloader, self.criterion = self.accelerator.prepare(
            self.model, self.train_dataloader, self.criterion)

    def train(self):
        wandb.init(project="NVC1B", config=self.config)

        stages = [
            ("Stage 1", self.train_motion, self.config.training.lambda_m, 
             Adam(list(self.model.motion_estimation.parameters()) + 
                  list(self.model.motion_encoder_decoder.parameters()), lr=self.config.training.lr_motion)),
            ("Stage 2", self.train_motion, self.config.training.lambda_m,
             Adam(list(self.model.motion_estimation.parameters()) + 
                  list(self.model.motion_encoder_decoder.parameters()) + 
                  list(self.model.motion_entropy_model.parameters()), lr=self.config.training.lr_motion)),
            ("Stage 3", self.train_reconstruction, self.config.training.lambda_r,
             Adam(list(self.model.temporal_context_mining.parameters()) + 
                  list(self.model.contextual_encoder_decoder.parameters()), lr=self.config.training.lr_reconstruction)),
            ("Stage 4", self.train_reconstruction, self.config.training.lambda_r,
             Adam(list(self.model.temporal_context_mining.parameters()) + 
                  list(self.model.contextual_encoder_decoder.parameters()) + 
                  list(self.model.contextual_entropy_model.parameters()), lr=self.config.training.lr_reconstruction)),
            ("Stage 5", self.train_all, self.config.training.lambda_t,
             Adam(self.model.parameters(), lr=self.config.training.lr_finetune)),
            ("Stage 6", self.multi_frame_fine_tune, self.config.training.lambda_T,
             Adam(self.model.parameters(), lr=self.config.training.lr_multi_frame))
        ]

        for stage, train_func, lambda_val, optimizer in stages:
            optimizer = self.accelerator.prepare(optimizer)
            for epoch in range(self.config.training.num_epochs_per_stage):
                loss = train_func(optimizer, lambda_val, epoch, stage)
                self.accelerator.print(f"{stage} - Epoch {epoch+1}/{self.config.training.num_epochs_per_stage}, Loss: {loss:.4f}")
                wandb.log({f"{stage}_loss": loss}, step=epoch)

        if self.accelerator.is_main_process:
            self.accelerator.save(self.model.state_dict(), self.config.training.save_path)
            wandb.save(self.config.training.save_path)

    def train_motion(self, optimizer, lambda_m, epoch, stage):
        self.model.train()
        total_loss = 0
        for step, (x_t, x_t_minus_1) in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            
            vs_t, vd_t = self.model.motion_estimation(x_t, x_t_minus_1)
            vs_t_hat, vd_t_hat, m_t_hat = self.model.motion_encoder_decoder(vs_t, vd_t)
            
            x_warp = self.warp_frame(x_t_minus_1, vs_t_hat, vd_t_hat)
            
            D_m_t = self.criterion(x_t, x_warp)
            R_m_t = self.model.motion_entropy_model(m_t_hat)
            
            loss = lambda_m * D_m_t + R_m_t
            
            self.accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                self.log_images(x_t, x_warp, epoch, step, stage, "motion")
        
        return total_loss / len(self.train_dataloader)

    def train_reconstruction(self, optimizer, lambda_r, epoch, stage):
        self.model.train()
        total_loss = 0
        for step, (x_t, x_t_minus_1) in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            
            vs_t, vd_t = self.model.motion_estimation(x_t, x_t_minus_1)
            vs_t_hat, vd_t_hat, _ = self.model.motion_encoder_decoder(vs_t, vd_t)
            C_t = self.model.temporal_context_mining(x_t_minus_1, vs_t_hat, vd_t_hat)
            x_t_hat, y_t_hat = self.model.contextual_encoder_decoder(x_t, C_t)
            
            D_r_t = self.criterion(x_t, x_t_hat)
            R_r_t = self.model.contextual_entropy_model(y_t_hat, C_t)
            
            loss = lambda_r * D_r_t + R_r_t
            
            self.accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                self.log_images(x_t, x_t_hat, epoch, step, stage, "reconstruction")
        
        return total_loss / len(self.train_dataloader)

    def train_all(self, optimizer, lambda_t, epoch, stage):
        self.model.train()
        total_loss = 0
        for step, (x_t, x_t_minus_1) in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            
            x_t_hat, m_t_compressed, y_t_compressed = self.model(x_t, x_t_minus_1)
            
            D_t = self.criterion(x_t, x_t_hat)
            R_t = m_t_compressed + y_t_compressed
            
            loss = lambda_t * D_t + R_t
            
            self.accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                self.log_images(x_t, x_t_hat, epoch, step, stage, "all")
        
        return total_loss / len(self.train_dataloader)

    def multi_frame_fine_tune(self, optimizer, lambda_T, epoch, stage):
        self.model.train()
        total_loss = 0
        for step, frames in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            
            loss = 0
            for t in range(1, self.config.training.num_frames):
                x_t = frames[:, t]
                x_t_minus_1 = frames[:, t-1]
                
                x_t_hat, m_t_compressed, y_t_compressed = self.model(x_t, x_t_minus_1)
                
                D_t = self.criterion(x_t, x_t_hat)
                R_t = m_t_compressed + y_t_compressed
                
                loss += lambda_T * D_t + R_t
            
            loss /= (self.config.training.num_frames - 1)
            
            self.accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                self.log_images(x_t, x_t_hat, epoch, step, stage, "multi_frame")
        
        return total_loss / len(self.train_dataloader)
    
    def log_images(self,original, reconstructed, epoch, step, stage, phase):
        original_grid = make_grid(original[:4], nrow=2, normalize=True)
        reconstructed_grid = make_grid(reconstructed[:4], nrow=2, normalize=True)
        wandb.log({
            f"{stage}_{phase}_original": wandb.Image(original_grid),
            f"{stage}_{phase}_reconstructed": wandb.Image(reconstructed_grid)
        }, step=epoch * len(self.train_dataloader) + step)

    @staticmethod
    def warp_frame(frame, flow_x, flow_y):
        h, w = frame.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((x, y), dim=-1).float().to(frame.device)
        
        flow = torch.stack((flow_x, flow_y), dim=-1)
        grid = grid + flow
        
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / (w - 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / (h - 1) - 1.0
        
        return F.grid_sample(frame, grid, mode='bilinear', padding_mode='border')



def main():
    config = OmegaConf.load('config.yaml')
    trainer = NVC1BTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()