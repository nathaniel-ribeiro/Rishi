import numpy as np
import csv
from scipy.stats import kendalltau
import random
import time

import config
from rishi import Rishi
from oracle import PikafishEngine

class MoveComparison:
  def __init__(self, oracle, rishi, data):
    self.oracle = oracle
    self.rishi = rishi
    self.data = data

    # stats tracked
    self.trials = 0
    self.move_accuracy_sum = 0
    self.tau_sum = 0
    self.tau_count = 0
    self.top_3_sum = 0
    
  def get_move_accuracy(self):
    if not self.trials: return 0
    return self.move_accuracy_sum / self.trials

  def get_tau(self):
    if not self.tau_count: return 0
    return self.tau_sum / self.tau_count

  def get_top_3(self):
    if not self.trials: return 0
    return self.top_3_sum / self.trials

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
        # negate scores because new_fen is opponent's turn
        oracle_score = -self.normalize_score(self.oracle.evaluate_pos(new_fen, think_time=20)[0])
        rishi_score = -self.rishi.evaluate(new_fen)
        oracle_ranking.append((oracle_score, move))
        rishi_ranking.append((rishi_score, move))
      
      oracle_ranking = [move for _, move in sorted(oracle_ranking, reverse=True)]
      rishi_ranking = [move for _, move in sorted(rishi_ranking, reverse=True)]

      # update stats
      self.trials += 1
      self.move_accuracy_sum += (oracle_ranking[0] == rishi_ranking[0])
      tau = kendalltau(oracle_ranking, rishi_ranking).statistic
      if not np.isnan(tau):
        self.tau_sum += tau
        self.tau_count += 1

      top_n = min(3, len(oracle_ranking))
      for i in range(top_n):
        self.top_3_sum += (oracle_ranking[i] == rishi_ranking[0])

      if not self.trials % 100:
        print(f"Trial {self.trials} / {len(self.data)}  complete")

    return self.get_move_accuracy(), self.get_tau(), self.get_top_3()

def load_fens(file_path):
  NUM_TRIALS = 10_000
  fens = []
  with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      fens.append(row['FEN'])

  # sample fens from data
  if len(fens) > NUM_TRIALS:
    fens = random.sample(fens, NUM_TRIALS)

  return fens

def main():
  start = time.time()

  RISHI_PATH = './models/rishi.pt'
  DATA_PATH = './data/val.csv'
  print('Loading data')
  data = load_fens(DATA_PATH)
  
  print('Instantiating engines')
  oracle = PikafishEngine(config.PIKAFISH_THREADS)
  rishi = Rishi(RISHI_PATH)

  print('Comparing evaluations...')
  comparison = MoveComparison(oracle, rishi, data)
  accuracy, taus, top_3 = comparison.compare_models()

  end = time.time()
  print(f'\nProcessed {len(data)} positions in {end - start} seconds')
  print(f'Move accuracy: {accuracy:.2%}')
  print(f"Average Kendall's tau: {taus: .4f}")
  print(f"Top 3 move accuracy: {top_3: .2%}")

if __name__ == '__main__':
  main()
