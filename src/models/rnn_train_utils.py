"""
Training loop for the RNN models.
"""

import time
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0) 
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

    return total_loss / n, correct / n

#Train with Adam + ReduceLROnPlateau + early stopping.
def train_model(model, train_loader, val_loader, class_weights=None,
                epochs=100, lr=1e-3, weight_decay=1e-5, patience=10,
                device=None, verbose=True, callback=None):
    device = device or get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss, best_val_acc = float('inf'), 0.0
    best_state, epochs_no_improve = None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        metrics['train_loss'].append(tr_loss)
        metrics['train_acc'].append(tr_acc)
        metrics['val_loss'].append(va_loss)
        metrics['val_acc'].append(va_acc)

        # keep the weights from the best validation accuracy
        if va_acc > best_val_acc:
            best_val_acc, best_val_loss = va_acc, va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  train {tr_loss:.3f}/{tr_acc:.3f}   "
                  f"val {va_loss:.3f}/{va_acc:.3f}")

        # optional hook: lets a hyperparameter search stop unpromising trials early
        if callback is not None:
            callback(epoch, va_loss, va_acc)

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - start
    if verbose:
        print(f"  best val loss {best_val_loss:.4f} / accuracy {best_val_acc:.4f}  ({train_time:.0f}s)")

    metrics['best_val_acc'] = best_val_acc
    metrics['best_val_loss'] = best_val_loss
    return metrics, train_time


@torch.no_grad()
def predict(model, loader, device=None):
    device = device or get_device()
    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []
    for x, y, lengths in loader:
        logits = model(x.to(device), lengths)
        probs = torch.softmax(logits, dim=1)
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob))
