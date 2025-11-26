import numpy as np
import csv
from scipy.stats import kendalltau

import config
from rishi import Rishi
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

  def normalize_score(self, centipawn):
      if isinstance(centipawn, str) and centipawn.startswith("M"):
          n = int(centipawn[1:])
          return 10_000 - n
      if isinstance(centipawn, str) and centipawn.startswith("-M"):
          n = int(centipawn[2:])
          return -10_000 + n
      return centipawn

  # rank moves for each fen in dataset and compare results to update stats
  def compare_models(self):
    self.oracle.new_game()
    for fen in self.data:
      oracle_ranking = []
      rishi_ranking = []
      for move in self.oracle.get_legal_moves(fen):
        new_fen = self.oracle.get_fen_after_fen_and_moves(fen, [move])
        oracle_ranking.append((self.normalize_score(self.oracle.evaluate_pos(new_fen)), move))
        rishi_ranking.append((self.normalize_score(self.rishi.evaluate(new_fen)), move))
      
      oracle_ranking = [move for _, move in sorted(oracle_ranking, reverse=True)]
      rishi_ranking = [move for _, move in sorted(rishi_ranking, reverse=True)]

      # update stats
      self.move_accuracy_vars[0] += (oracle_ranking[0] == rishi_ranking[0])
      self.move_accuracy_vars[1] += 1
      self.taus.append(kendalltau(oracle_ranking, rishi_ranking).statistic)

    return self.get_move_accuracy(), self.get_taus()

def load_fens(file_path):
  fens = []
  with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      fens.append(row['FEN'])
  return fens

def main():
  DATA_PATH = './data/temp.csv'
  data = load_fens(DATA_PATH)
  
  oracle = PikafishEngine(config.PIKAFISH_THREADS)
  rishi = Rishi(None)
  comparison = MoveComparison(oracle, rishi, data)
  accuracy, taus = comparison.compare_models()
  print(f'accuracy: {accuracy}')
  print(f"kendall's taus (first ten): {taus[:10]}")


if __name__ == '__main__':
  main()
