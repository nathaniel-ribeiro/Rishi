from oracle import PikafishEngine
from tokenizer import BoardTokenizer
import config
import numpy as np
import torch
from model import TransformerClassifier
from pprint import pprint

class Rishi:
    def __init__(self, path_to_model, temperature=0.005, top_p=0.9):
        self.pikafish = PikafishEngine(config.PIKAFISH_THREADS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = BoardTokenizer(98)
        self.model = torch.load(path_to_model, weights_only=False, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
    
    def send(self, cmd):
        self.pikafish.send(cmd)
    
    def new_game(self):
        self.pikafish.new_game()
    
    def set_position(self, fen):
        self.pikafish.set_position(fen)

    def setup_game(self, move_history):
        self.pikafish.setup_game(move_history)
    
    def get_fen_after_moves(self, moves):
        return self.pikafish.get_fen_after_moves(moves)
    
    def get_fen_after_fen_and_moves(self, fen, moves):
        return self.pikafish.get_fen_after_fen_and_moves(fen, moves)
    
    def get_best_move(self, fen):
        legal_moves = self.pikafish.get_legal_moves(fen)
        future_fens = [self.pikafish.get_fen_after_fen_and_moves(fen, legal_move) for legal_move in legal_moves]
        future_fens_tokenized = np.array([self.tokenizer.encode(future_fen) for future_fen in future_fens])

        future_fens_tokenized = torch.from_numpy(future_fens_tokenized)
        future_fens_tokenized = future_fens_tokenized.to(self.device)

        # pass through model and get expected score from win probs
        win_probs = None
        with torch.no_grad():
            logits = self.model(future_fens_tokenized)
            win_probs = torch.sigmoid(logits)
        expected_scores = win_probs.squeeze()
        expected_scores = expected_scores.detach().cpu().numpy()
        pprint(expected_scores)
        # minimize the OPPONENT's win probability
        selected_move_idx = np.argmin(expected_scores)
        selected_move = legal_moves[selected_move_idx]
        return selected_move

        