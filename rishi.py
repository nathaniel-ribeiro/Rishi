from oracle import PikafishEngine
from tokenizer import BoardTokenizer
import config
import numpy as np
import torch
from model import TransformerClassifier
from pprint import pprint

import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    """
    Dummy model to test Rishi.get_best_move() without a trained checkpoint.
    Returns random logits (before sigmoid) to simulate P(side-to-move wins).
    """
    def __init__(self, mode="logit"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x is a tensor of shape [N, seq_len]
        N = x.shape[0]

        if self.mode == "logit":
            # Return a random logit per position
            return torch.randn(N, 1, device=x.device)
        elif self.mode == "wdl":
            # Optional alternate mode: return fake WDL probabilities
            z = torch.randn(N, 3, device=x.device)
            return F.softmax(z, dim=-1)
        else:
            raise ValueError("Unknown mode. Use 'logit' or 'wdl'.")

class Rishi:
    def __init__(self, path_to_model, temperature=0.005, top_p=0.9):
        self.pikafish = PikafishEngine(config.PIKAFISH_THREADS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = BoardTokenizer(98)
        #TODO: fix this later by serializing the whole model in the training script
        if path_to_model == "__DUMMY__":
            self.model = DummyModel("logit")
        else:
            self.model = torch.load(path_to_model, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        self.temperature = temperature
        self.top_p = top_p
    
    def send(self, cmd):
        self.pikafish.send(cmd)
    
    def new_game(self):
        self.pikafish.new_game()
    
    def set_position(self, fen):
        self.pikafish.set_position(fen)

    def setup_game(self, move_history):
        self.pikafish.setup_game(move_history)

    def play_moves(self,fen,moves):
        return self.pikafish.play_moves(fen,moves)

    def is_checkmate(self,fen):
        return self.pikafish.is_checkmate(fen)
    
    def get_fen_after_moves(self, moves):
        return self.pikafish.get_fen_after_moves(moves)
    
    def get_fen_after_fen_and_moves(self, fen, moves):
        return self.pikafish.get_fen_after_fen_and_moves(fen, moves)
    
    def quit(self):
        self.pikafish.send("quit")
        self.pikafish.engine.wait()

    def get_best_move(self, fen, mode = "puzzle"):
        legal_moves = self.pikafish.get_legal_moves(fen)
        if not legal_moves:
            self.pikafish.set_position(fen)
            return self.pikafish.get_best_move(config.PIKAFISH_MOVETIME_MS)

        future_fens = [
            self.pikafish.get_fen_after_fen_and_moves(fen, move)
            for move in legal_moves
        ]

        future_fens_tokenized = np.array(
            [self.tokenizer.encode(future_fen) for future_fen in future_fens],
            dtype=np.int64,
        )
        future_fens_tokenized = torch.from_numpy(future_fens_tokenized).to(self.device)

        with torch.no_grad():
            logits = self.model(future_fens_tokenized).squeeze(-1)
            side_win_probs = torch.sigmoid(logits)
            w_probs = (1.0 - side_win_probs).detach().cpu().numpy().ravel()

        # optional Pikafish fallback if everything is trivially winning
        if (w_probs.size >= 5) and np.all(np.sort(w_probs)[::-1][:5] >= 0.99):
            sorted_idxs = np.argsort(w_probs)[::-1]

            top5_idxs = sorted_idxs[:5]
            top5_moves = [legal_moves[i] for i in top5_idxs]
            successors = self.pikafish.get_all_successors_from_moves(fen, top5_moves, config.PIKAFISH_MOVETIME_MS)

            return successors[0]["move"]

        #Greedy decision for puzzle solving
        if(mode == "puzzle"):
            best_idx = int(np.argmax(w_probs))
            return legal_moves[best_idx]
        
        # Temperature softmax
        cooled_scores = w_probs / self.temperature
        e_x = np.exp(cooled_scores - np.max(cooled_scores))
        move_probabilities = e_x / e_x.sum()

        # top-p sampling
        order = np.argsort(move_probabilities)[::-1]
        probs_sorted = move_probabilities[order]
        cumulative = np.cumsum(probs_sorted)
        cutoff = int(np.searchsorted(cumulative, self.top_p) + 1)
        cutoff = max(cutoff, 1)

        keep_indices = order[:cutoff]
        top_probs = probs_sorted[:cutoff]
        den = top_probs.sum()

        normalized_top_probs = (top_probs / den).astype(float).ravel()
        normalized_top_probs = np.maximum(normalized_top_probs, 0.0)
        s = normalized_top_probs.sum()
        if not np.isfinite(s) or s <= 0:
            normalized_top_probs = np.ones_like(normalized_top_probs) / len(normalized_top_probs)
        else:
            normalized_top_probs /= s

        choice = np.random.choice(len(keep_indices), p=normalized_top_probs)
        selected_idx = int(keep_indices[choice])
        return legal_moves[selected_idx]

    def evaluate(self, move_history):
        fen = self.pikafish.get_fen_after_moves(move_history)
        fen_tokenized = self.tokenizer.encode(fen)
        inputs = np.array(fen_tokenized).unsqueeze(0)

        win, _, lose = self.model(inputs).squeeze(0).tolist()
        # assumes 1 point for winning, 0 points for draw, -1 point for losing (different from get best move!)
        expected_score = win - lose
        # TODO: convert from expected score to CP
        return 0

        