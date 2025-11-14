import numpy as np
import csv
from scipy.stats import kendalltau

import config
from rishi_dummy import Rishi
from oracle import PikafishEngine

class MoveComparison:
  def __init__(self, oracle, rishi, data):
    self.oracle = oracle
    self.rishi = rishi
    self.data = data

    # stats tracked
    self.move_accuracy_vars = [0, 0] # sum, number of elems
    self.taus = [] # list of kendall's taus

  def get_move_accuracy(self):
    total, n = self.move_accuracy_vars
    if not n: return 0
    return total / n

  def get_taus(self):
    return self.taus

  # takes 2d numpy array of move rankings from both engines and calculates
  # average spearman correlation
  # def spearman(self, pi_ranks, ri_ranks):
  #   n, m = pi_ranks.shape

  #   # encode strings as integers
  #   unique_moves = np.unique(np.concatenate([pi_ranks.flatten(), ri_ranks.flatten()]))
  #   move_to_int = {move: idx for idx, move in enumerate(unique_moves)}
  #   pi_encoded = np.vectorize(move_to_int.get)(pi_ranks)
  #   ri_encoded = np.vectorize(move_to_int.get)(ri_ranks)

  #   # map moves to rank
  #   ri_rank_map = np.zeros((m, len(unique_moves)), dtype=int)
  #   ri_rank_map[np.arange(n)[:, None], ri_encoded] = np.arange(m)

  #   pi_to_ri_ranks = ri_rank_map[np.arange(n)[:, None], pi_encoded]

  #   # spearman formula
  #   d = np.arange(m) - pi_to_ri_ranks
  #   d2_sum = np.sum(d**2, axis=1)
  #   rho_per_position = 1 - (6 * d2_sum) / (m * (m**2 - 1))
    
  #   return np.mean(rho_per_position)

  # rank moves for each fen in dataset and compare results to update stats
  def compare_models(self):
    oracle.new_game()
    for fen in fens:
      oracle.set_position(fen)
      legal_moves = {move: oracle.get_fen_after_fen_and_moves(fen, [move]) for move in self.oracle.get_legal_moves()}

      oracle_ranking = [move for eval, move in [(self.oracle.evaluate(legal_moves[move]), move) for move in legal_moves].sort()]
      rishi_ranking = [move for eval, move in [(self.rishi.evaluate(legal_moves[move]), move) for move in legal_moves].sort()]

      # update stats
      self.move_accuracy_vars[0] += oracle_ranking[0][0] == rishi_ranking[0][0]
      self.move_accuracy_vars[0] += 1
      self.taus.append(kendalltau(oracle_ranking, rishi_ranking))

    return self.get_move_accuracy(), self.get_taus()

def load_fens(file_path):
  fens = []
  with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      fens.append(row['FEN'])
  return fens

def normalize_score(centipawn):
    if isinstance(centipawn, str) and centipawn.startswith("M"):
        n = int(centipawn[1:])
        return 10_000 - n
    if isinstance(centipawn, str) and centipawn.startswith("-M"):
        n = int(centipawn[2:])
        return -10_000 + n
    return centipawn

def main():
  DATA_PATH = './temp.csv'
  data = load_fens(DATA_PATH)
  
  oracle = PikafishEngine(config.PIKAFISH_THREADS)
  rishi = Rishi(None)
  comparison = MoveComparison(oracle, rishi, data)
  accuracy, taus = comparison.compare_models()
  print(f'accuracy: {accuracy}')
  print(f"kendall's taus (first ten): {taus[:10]}")


if __name__ == '__main__':
  main()
