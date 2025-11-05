import torch
import pandas as pd
import numpy as np

class AnnotatedBoardsDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv, tokenizer, board_flip_probability=0.0):
        self.df = pd.read_csv(path_to_csv)
        self.tokenizer = tokenizer
        self.board_flip_probability = board_flip_probability
    
    def horizontal_flip(self, fen):
        board, metadata = fen.split(" ", 1)
        rows = board.split("/")
        rows_flipped = [row[::-1] for row in rows]
        board_flipped = "/".join(rows_flipped)
        fen_flipped = board_flipped + " " + metadata
        return fen_flipped

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # ignore game id and cp evaluation
        _, fen, _, win_prob, _, _ = tuple(row)
        evaluation = np.array([win_prob])
        evaluation = torch.from_numpy(evaluation)

        if np.random.rand() <= self.board_flip_probability:
            fen = self.horizontal_flip(fen)

        fen_tokenized_indices = self.tokenizer.encode(fen)
        fen_tokenized_indices = torch.from_numpy(fen_tokenized_indices)
        return fen_tokenized_indices, evaluation
