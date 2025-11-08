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
        self.pikafish.play_moves(fen,moves)
    
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
        wdl_probs = self.model(future_fens_tokenized)
        expected_scores = torch.sum(wdl_probs * torch.tensor([1.0, 0.0, 0.0], device=self.device), dim=1)
        expected_scores = expected_scores.squeeze()
        expected_scores = expected_scores.detach().cpu().numpy()

        # pprint(dict(zip(legal_moves, expected_scores.tolist())))
        
        # flip expected score to be our winning chances in states s'
        expected_scores = 1 - expected_scores

        # to keep attacking strategy consistent, use Pikafish if all of the top 5 moves have >= 99% chance of winning
        if(np.all(np.sort(expected_scores)[::-1][:5] >= 0.99)):
            self.pikafish.set_position(fen)
            return self.pikafish.get_best_move(config.PIKAFISH_MOVETIME_MS)

        # convert to sharpened probability distribution with softmax w. temperature
        cooled_scores = expected_scores / self.temperature
        e_x = np.exp(cooled_scores - np.max(cooled_scores))
        move_probabilities = e_x / e_x.sum()

        # top p sampling
        sorted_probs_indices = np.argsort(move_probabilities)[::-1]
        sorted_probs = move_probabilities[sorted_probs_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.where(cumulative_probs >= self.top_p)[0][0] + 1
        keep_indices = sorted_probs_indices[:cutoff_index]

        top_p_moves = [legal_moves[idx] for idx in keep_indices]
        top_p_probs = move_probabilities[keep_indices]
        normalized_top_p_probs = top_p_probs / np.sum(top_p_probs)

        # print("Choosing from the following moves with corresponding probabilities:")
        # pprint(dict(zip(top_p_moves, normalized_top_p_probs.tolist())))
        selected_move_index = np.random.choice(
            a=keep_indices,
            p=normalized_top_p_probs
        )

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

        