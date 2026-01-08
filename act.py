import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import os
import cv2
import time
from collections import deque
from roarm_sdk.roarm import roarm


STATE_DIM = 4      
ACTION_DIM = 4     
CHUNK_SIZE = 50    
HIDDEN_DIM = 512 
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATS_PATH = "norm_stats.pth"
MODEL_PATH = "roarm_act_model.pth"

class RoArmDataset(Dataset):
    def __init__(self, folder_path, chunk_size=50, augment=False):
        self.chunk_size = chunk_size
        self.augment = augment
        self.samples = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        
        if not files:
            raise FileNotFoundError("no recordings")

        all_poses = []
        print("Processing recordings and calculating statistics...")
        for f in files:
            data = np.load(f)
            all_poses.append(data['poses'])
            images, poses = data['images'], data['poses']
            
            for i in range(len(poses) - chunk_size):
                move_dist = np.linalg.norm(poses[i + chunk_size] - poses[i])
                if move_dist > 0.5:
                    self.samples.append({
                        'img': images[i][-1], 
                        'pos': poses[i], 
                        'chunk': poses[i:i+chunk_size]
                    })
        
        all_poses = np.concatenate(all_poses, axis=0)
        self.stats = {
            'mean': torch.tensor(all_poses.mean(axis=0), dtype=torch.float32),
            'std': torch.tensor(all_poses.std(axis=0), dtype=torch.float32) + 1e-6
        }
        
        self.aug_pipeline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.augment:
            img = self.aug_pipeline(s['img'])
        else:
            img = torch.from_numpy(s['img']).permute(2, 0, 1).float() / 255.0
            
        pos = (torch.tensor(s['pos'], dtype=torch.float32) - self.stats['mean']) / self.stats['std']
        chunk = (torch.tensor(s['chunk'], dtype=torch.float32) - self.stats['mean']) / self.stats['std']
        return img, pos, chunk

class ACTModel(nn.Module):
    def __init__(self, state_dim, action_dim, chunk_size, hidden_dim):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) 
        self.input_proj = nn.Linear(512, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8, num_encoder_layers=4, num_decoder_layers=4, batch_first=True
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, image, state):
        vis_feat = self.backbone(image).flatten(1)
        vis_feat = self.input_proj(vis_feat).unsqueeze(1)
        state_feat = self.state_proj(state).unsqueeze(1)
        enc_input = torch.cat([vis_feat, state_feat], dim=1)
        dec_input = self.query_embed.weight.unsqueeze(0).repeat(image.shape[0], 1, 1)
        out = self.transformer(enc_input, dec_input)
        return self.action_head(out)

def run_train():
    dataset = RoArmDataset("./recordings", CHUNK_SIZE, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    torch.save(dataset.stats, STATS_PATH)
    
    model = ACTModel(STATE_DIM, ACTION_DIM, CHUNK_SIZE, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    print(f"Device:{DEVICE}")
    model.train()
    for epoch in range(EPOCHS):
        loss_val = 0
        for img, pos, chunk in dataloader:
            img, pos, chunk = img.to(DEVICE), pos.to(DEVICE), chunk.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img, pos)
            loss = criterion(pred, chunk)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_val/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved")

def run_play():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return
        
    stats = torch.load(STATS_PATH)
    model = ACTModel(STATE_DIM, ACTION_DIM, CHUNK_SIZE, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    arm = roarm(roarm_type="roarm_m2", port="COM6", baudrate=115200)
    cap = cv2.VideoCapture(0)
    current_pose = [300.0, 0.0, 200.0, 0.0]
    
    all_time_actions = deque(maxlen=CHUNK_SIZE)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            img_t = torch.from_numpy(cv2.resize(frame, (256, 256))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
            pos_t = ((torch.tensor(current_pose, dtype=torch.float32) - stats['mean']) / stats['std']).unsqueeze(0).to(DEVICE)
            

            pred_norm = model(img_t, pos_t).cpu()[0]
            pred_real = (pred_norm * stats['std']) + stats['mean']
            all_time_actions.append(pred_real)

            num_chunks = len(all_time_actions)
            curr_poses = []
            for i in range(num_chunks):
                curr_poses.append(all_time_actions[-(i+1)][i])
            
            target = torch.stack(curr_poses).mean(dim=0).tolist()
            target[3] = max(0.0, min(90.0, target[3]))
            target[2] = max(-100, min(300, target[2]))
            
            
            arm.pose_ctrl(target)
            current_pose = target
            
            cv2.imshow("Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("'train' or 'play':").strip().lower()
    if mode == 'train': run_train()
    elif mode == 'play': run_play()