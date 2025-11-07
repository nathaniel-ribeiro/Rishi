import torch
from dataset import AnnotatedBoardsDataset
import config
from tokenizer import BoardTokenizer
from model import TransformerClassifier
import numpy as np
import time
import argparse
import copy
import torch.nn.functional as F
import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration options")

    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size per iteration")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--board_flip_p", type=float, default=0.5,
                        help="Probability of horizontally flipping the board for data augmentation")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads per transformer layer")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of transformer encoder layers")
    parser.add_argument("--max_seq_len", type=int, default=98,
                        help="Maximum input sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay factor in [0.0, 1.0]")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--save_model", action="store_true",
                        help="If specified, save the trained model at the end of training")
    parser.add_argument("--wandb", action="store_true",
                        help="If specified, log metrics using Weights and Biases (WandB)")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    MAX_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience
    LEARNING_RATE = args.learning_rate
    BOARD_FLIP_P = args.board_flip_p
    D_MODEL = args.d_model
    N_HEADS = args.n_heads
    N_LAYERS = args.n_layers
    MAX_SEQ_LEN = args.max_seq_len
    WEIGHT_DECAY = args.weight_decay
    DROPOUT = args.dropout
    SAVE_MODEL = args.save_model
    WANDB = args.wandb

    run = None
    if WANDB: run = wandb.init(project="rishi", 
                               config={
                                   "learning_rate": LEARNING_RATE,
                                   "d_model": D_MODEL,
                                   "n_heads": N_HEADS,
                                   "n_layers": N_LAYERS,
                                   "weight_decay": WEIGHT_DECAY,
                                   "dropout": DROPOUT
                               })
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BoardTokenizer(MAX_SEQ_LEN)

    train_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/train.csv', tokenizer, BOARD_FLIP_P)
    val_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/val.csv', tokenizer)
    test_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/test.csv', tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    VOCAB_SIZE = tokenizer.vocab_size
    model = TransformerClassifier(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, N_LAYERS, N_HEADS, DROPOUT).to(device)
    if WANDB: wandb.watch(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bce = torch.nn.BCEWithLogitsLoss()
    rmse = torch.nn.MSELoss()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {parameter_count/1e6:.1f} M params")
    old_val_loss = np.inf
    patience = PATIENCE
    scaler = torch.amp.GradScaler(device)
    best_val_rmse = np.inf
    best_model = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        total = 0

        # train
        tick = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                logits = model(inputs)
                # required to prevent PyTorch from shitting itself when encountering a double under AMP
                labels = labels.float()
                # BCEWithLogits needs logits
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        
        avg_train_loss = train_loss / total
        
        # validate
        val_loss = 0.0
        val_rmse = 0.0
        total = 0
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device):
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    outputs = F.sigmoid(logits)
                    # required to prevent PyTorch from shitting itself when encountering a double under AMP
                    labels = labels.float()
                    # RMSE
                    loss = bce(logits, labels)
                    rmse = torch.sqrt(rmse(outputs, labels))
                    val_loss += loss.item() * inputs.size(0)
                    val_rmse += rmse.item() * inputs.size(0)
                    total += inputs.size(0)

        avg_val_loss = val_loss / total
        avg_val_rmse = val_rmse / total
        if WANDB: run.log({"train_loss_bce": avg_train_loss, 'val_loss_bce': avg_val_loss, 'val_rmse': avg_val_rmse, 'epoch': epoch})
        print(f"Epoch {epoch}: \t Train Loss (BCE): {avg_train_loss:.3f} \t Val Loss (BCE): {avg_val_loss:.3f} \t Val Loss (RMSE): {avg_val_loss:.3f}")
        
        # track best RMSE to save best version of model
        if avg_val_rmse < best_val_loss:
            best_val_loss = avg_val_rmse
            best_model = copy.deepcopy(model)
        
        # patience uses BCE to determine early stopping
        if avg_val_loss < old_val_loss:
            patience = PATIENCE
        else:
            patience -= 1
        if patience <= 0:
            break
        old_val_loss = avg_val_loss

    print(f"\nBest val loss: {best_val_loss:.3f}")
    if SAVE_MODEL: torch.save(best_model, f"{config.MODELS_DIR}/rishi.pt")
    if WANDB: run.finish()