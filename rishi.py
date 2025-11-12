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

        self.tokenizer = BoardTokenizer(97)
        #TODO: fix this later by serializing the whole model in the training script
        self.model = torch.load(path_to_model, map_location=self.device)
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
    
    def get_best_move(self, fen):
        #TODO: update this to account for change from WDL probs -> win prob
        legal_moves = self.pikafish.get_legal_moves(fen)
        future_fens = [self.pikafish.get_fen_after_fen_and_moves(fen, legal_move) for legal_move in legal_moves]
        future_fens_tokenized = np.array([self.tokenizer.encode(future_fen) for future_fen in future_fens])

        future_fens_tokenized = torch.from_numpy(future_fens_tokenized)
        future_fens_tokenized = future_fens_tokenized.to(self.device)

        # pass through model and get expected score from WDL probs
        logits = self.model(future_fens_tokenized)
        side_win_probs = torch.sigmoid(logits)
        w_probs = (1.0 - side_win_probs).detach().cpu().numpy()

        # pprint(dict(zip(legal_moves, expected_scores.tolist())))
        
        # to keep attacking strategy consistent, use Pikafish if all of the top 5 moves have >= 99% chance of winning
        if (w_probs.size >= 5) and np.all(np.sort(w_probs)[::-1][:5] >= 0.99):
            self.pikafish.set_position(fen)
            return self.pikafish.get_best_move(config.PIKAFISH_MOVETIME_MS)

        # convert to sharpened probability distribution with softmax w. temperature
        cooled_scores = w_probs / self.temperature
        e_x = np.exp(cooled_scores - np.max(cooled_scores))
        move_probabilities = e_x / e_x.sum()

        # --- top p sampling (keep it strictly 1-D and safe) ---
        order = np.argsort(move_probabilities)[::-1]
        probs_sorted = move_probabilities[order]
        cumsum = np.cumsum(probs_sorted)
        cutoff = int(np.searchsorted(cumsum, self.top_p) + 1)
        cutoff = max(cutoff, 1)  # ensure at least one kept

        keep_indices = order[:cutoff]                     # shape (K,)
        top_p_probs = probs_sorted[:cutoff]               # shape (K,)
        den = top_p_probs.sum()

        # Fallbacks for degenerate cases
        if cutoff == 1 or not np.isfinite(den) or den <= 0:
            selected_move_index = int(keep_indices[0])
        else:
            normalized_top_p_probs = (top_p_probs / den).astype(float).ravel()
            # clamp tiny negatives & renormalize to avoid “probabilities do not sum to 1”
            normalized_top_p_probs = np.maximum(normalized_top_p_probs, 0.0)
            s = normalized_top_p_probs.sum()
            if (not np.isfinite(s)) or s <= 0:
                normalized_top_p_probs = np.ones_like(normalized_top_p_probs) / len(normalized_top_p_probs)
            else:
                normalized_top_p_probs /= s

        choice_pos = np.random.choice(len(keep_indices), p=normalized_top_p_probs)
        selected_move_index = int(keep_indices[choice_pos])
        selected_move = legal_moves[selected_move_index]
        return selected_move

    def evaluate(self, move_history):
        fen = self.pikafish.get_fen_after_moves(move_history)
        fen_tokenized = self.tokenizer.encode(fen)
        inputs = np.array(fen_tokenized).unsqueeze(0)

        win, _, lose = self.model(inputs).squeeze(0).tolist()
        # assumes 1 point for winning, 0 points for draw, -1 point for losing (different from get best move!)
        expected_score = win - lose
        # TODO: convert from expected score to CP
        return 0

        