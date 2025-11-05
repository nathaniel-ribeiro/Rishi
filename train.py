import torch
from dataset import AnnotatedBoardsDataset
import config
from tokenizer import BoardTokenizer
from model import TransformerClassifier
import numpy as np
import time
import random
import argparse
import copy
import math
from transformers import get_cosine_schedule_with_warmup

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration options")

    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size per iteration")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--board_flip_p", type=float, default=0.5,
                        help="Probability of horizontally flipping the board for data augmentation")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads per transformer layer")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of transformer encoder layers")
    parser.add_argument("--max_seq_len", type=int, default=97,
                        help="Maximum input sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay factor in [0.0, 1.0]")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--save_model", action="store_true",
                        help="If specified, save the trained model at the end of training")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda": 
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    tokenizer = BoardTokenizer(MAX_SEQ_LEN)

    train_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/train.csv', tokenizer, BOARD_FLIP_P)
    val_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/val.csv', tokenizer)
    test_ds = AnnotatedBoardsDataset(f'{config.DATA_DIR}/test.csv', tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    VOCAB_SIZE = tokenizer.vocab_size
    model = TransformerClassifier(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, N_LAYERS, N_HEADS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = MAX_EPOCHS * math.ceil(len(train_ds) / BATCH_SIZE)
    num_warmup_steps = 0.1 * num_training_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )
    criterion = torch.nn.MSELoss()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {parameter_count/1e6:.1f} M params")
    old_val_loss = np.inf
    patience = PATIENCE
    scaler = torch.amp.GradScaler(device)
    best_val_loss = np.inf
    best_model = None
    current_step = 0

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
                outputs = model(inputs)
                # required to prevent PyTorch from shitting itself when encountering a double under AMP
                labels = labels.float()
                # RMSE
                loss = torch.sqrt(criterion(outputs, labels))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            current_step += 1

            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        
        avg_train_loss = train_loss / total
        
        # validate
        val_loss = 0.0
        total = 0
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device):
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    # required to prevent PyTorch from shitting itself when encountering a double under AMP
                    labels = labels.float()
                    # RMSE
                    loss = torch.sqrt(criterion(outputs, labels))
                    val_loss += loss.item() * inputs.size(0)
                    total += inputs.size(0)

        avg_val_loss = val_loss / total
        if device == "cuda": torch.cuda.synchronize()
        tock = time.time()
        elapsed_mins = (tock - tick) / 60

        print(f"Losses for epoch {epoch}: \t Train: {avg_train_loss:.3f} \t Val: {avg_val_loss:.3f} \t in {elapsed_mins:.1f} mins")
        
        # skip early stopping if we're still warming up
        if current_step < num_warmup_steps:
            continue

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
        
        if avg_val_loss < old_val_loss:
            patience = PATIENCE
        else:
            patience -= 1
        if patience <= 0:
            break
        old_val_loss = avg_val_loss

    if SAVE_MODEL: torch.save(best_model, f"{config.MODELS_DIR}/rishi.pt")