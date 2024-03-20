import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group('nccl')
    
def cleanup():
    dist.destroy_process_group()
    
class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn([4,10])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        
    def forward(self,x):
        return self.lin1(x)

ckpt_path = 'tmp.pth'

def main():
    #set up ddp
    setup()
    rank = dist.get_rank()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    device_id = rank % torch.cuda.device_count()
    
    #prepare data
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    
    #prepare model
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    #train
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for x in dataloader:
            print(f'epoch {epoch}, rank {rank} data: {x}')
            optimizer.zero_grad()
            y = ddp_model(x)
            loss = loss_fn(y, y)
            loss.backward()
            optimizer.step()
    
    #save model
    if rank == 0:
        torch.save(ddp_model.state_dict(), ckpt_path)
    

    dist.barrier()
    
    #load model
    map_location = {'cuda:0': f'cuda:{device_id}'}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    print(f'rank {rank}: {state_dict}')
    ddp_model.load_state_dict(state_dict)
    
    cleanup()
    
if __name__ == '__main__':
    main()