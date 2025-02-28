import torch
import time
from torcheval.metrics.functional import peak_signal_noise_ratio

def train_epoch(model, optimizer, criterion, train_dataloader, device, scheduler=None, epoch=0, log_interval=30):
    model.train()
    total_psnr, count = 0, 0
    losses = []
    
    for idx, (imgs, labels) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        pred = model(imgs)
        
        loss = criterion(pred, labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        total_psnr += peak_signal_noise_ratio(pred, labels)
        count += 1
        
        if idx % log_interval == 0 and idx > 0:
            print(\
                f"Epoch {epoch:3d} | {idx:5d}/{len(train_dataloader):5d} batches" \
                f"| PSNR: {(total_psnr / count):8.3f} "
                )
            total_psnr, count = 0, 0 
            
    epoch_psnr = total_psnr / count
    epoch_loss = sum(losses) / len(losses)
    if scheduler is not None:
        scheduler.step()
    return epoch_psnr, epoch_loss

def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []
    
    with torch.no_grad():
        for _, (imgs, labels) in enumerate(valid_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            losses.append(loss.item())
            
            total_psnr += peak_signal_noise_ratio(pred, labels)
            total_count += 1
            
    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def train_model(model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, epochs, device, scheduler=None):
    train_psnrs, train_losses = [], []
    val_psnrs, val_losses = [], []
    best_psnr_eval = -10000
    times = []
    
    for epoch in range(1, epochs+1):
        start_time = time.time()
        # Training 
        train_psnr, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, scheduler, epoch)
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)
        
        # Evaluation
        val_psnr, val_loss = evaluate_epoch(model, criterion, valid_dataloader, device)
        val_psnrs.append(val_psnr.cpu())
        val_losses.append(val_loss)
        
        # Save best model
        if best_psnr_eval < val_psnr:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
            inputs_t, targets_t = next(iter(valid_dataloader))
            best_psnr_eval = val_psnr
            
        times.append(time.time() - start_time)
        print("-" * 60)
        print(
            "| End of epoch {:3d} | Time: {:5.2f} | Train PSNR: {:8.3f} | Train Loss {:8.3f} "
            "| Valid PSNR {:8.3f} | Valid Loss {:8.3f}".format(
                epoch, time.time() - start_time, train_psnr, train_loss, val_psnr, val_loss
            )
        )
        print("-" * 60)
        
    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_psnr': train_psnrs,
        'train_loss': train_losses,
        'valid_psnr': val_psnrs,
        'valid_loss': val_psnrs,
        'time': times
    }
    return model, metrics