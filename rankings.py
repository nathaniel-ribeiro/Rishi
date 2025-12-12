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
    
    # store individual results
    self.move_accuracies = []
    self.taus = []
    
  def get_move_accuracy(self):
    if not self.trials: return 0
    return self.move_accuracy_sum / self.trials

  def get_tau(self):
    if not self.tau_count: return 0
    return self.tau_sum / self.tau_count

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
    print_interval = max(10, len(self.data) // 10)

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

      # skip boards with no legal moves
      if not rishi_ranking or not oracle_ranking:
        continue

      # update stats
      self.trials += 1
      is_correct = (oracle_ranking[0] == rishi_ranking[0])
      self.move_accuracy_sum += is_correct
      self.move_accuracies.append(is_correct)
      
      tau_result = kendalltau(oracle_ranking, rishi_ranking)
      tau = tau_result.statistic
      
      if not np.isnan(tau):
        self.tau_sum += tau
        self.tau_count += 1
        self.taus.append(tau)

      # print progress after every 10%
      if self.trials % print_interval == 0:
        accuracy = self.get_move_accuracy()
        taus = self.get_tau()
        print(f"TRIAL {self.trials} / {len(self.data)}:")
        print(f'Move accuracy: {accuracy:.2%}')
        print(f"Average Kendall's tau: {taus: .4f}")

    return self.get_move_accuracy(), self.get_tau()
  
  def get_aggregate_statistics(self, confidence=0.95):
    """Calculate aggregate statistics across all FENs"""
    alpha = 1 - confidence
    
    # calculate statistics for move accuracy
    accuracy_mean = np.mean(self.move_accuracies) if self.move_accuracies else 0
    accuracy_std = np.std(self.move_accuracies, ddof=1) if len(self.move_accuracies) > 1 else 0
    accuracy_se = accuracy_std / np.sqrt(len(self.move_accuracies)) if self.move_accuracies else 0
    
    # calculate statistics for tau
    tau_mean = np.mean(self.taus) if self.taus else 0
    tau_std = np.std(self.taus, ddof=1) if len(self.taus) > 1 else 0
    tau_se = tau_std / np.sqrt(len(self.taus)) if self.taus else 0
    
    # calculate confidence intervals using normal approximation
    from scipy.stats import norm
    z_score = norm.ppf(1 - alpha/2)
    
    accuracy_ci = (
      accuracy_mean - z_score * accuracy_se,
      accuracy_mean + z_score * accuracy_se
    ) if self.move_accuracies else (0, 0)
    
    tau_ci = (
      tau_mean - z_score * tau_se,
      tau_mean + z_score * tau_se
    ) if self.taus else (0, 0)
    
    # overall p-value: test if mean tau is significantly different from 0
    if len(self.taus) > 1:
      t_stat = tau_mean / tau_se if tau_se > 0 else 0
      from scipy.stats import t
      p_value = 2 * (1 - t.cdf(abs(t_stat), len(self.taus) - 1))
    else:
      p_value = 1.0
    
    return {
      'accuracy_ci': accuracy_ci,
      'tau_ci': tau_ci,
      'p_value': p_value,
      'tau_std': tau_std,
      'accuracy_std': accuracy_std
    }

def load_fens(file_path, num_trials):
  fens = []
  with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      fens.append(row['FEN'])

  # sample fens from data
  if len(fens) > num_trials:
    fens = random.sample(fens, num_trials)

  return fens

def main():
  start = time.time()

  RISHI_PATH = './models/rishi.pt'
  DATA_PATH = './data/test.csv'
  NUM_TRIALS = 150_000
  print('Loading data')
  data = load_fens(DATA_PATH, NUM_TRIALS)
  
  print('Instantiating engines')
  oracle = PikafishEngine(config.PIKAFISH_THREADS)
  rishi = Rishi(RISHI_PATH)

  print('Comparing evaluations...\n')
  comparison = MoveComparison(oracle, rishi, data)
  accuracy, taus = comparison.compare_models()
  
  # calculate aggregate statistics
  stats = comparison.get_aggregate_statistics()

  end = time.time()
  duration = int(end - start)
  hours = duration // 3600
  duration %= 3600
  minutes = duration // 60
  seconds = duration % 60
  
  print(f"\n{'='*60}")
  print(f"Final Results ({len(data)} positions):")
  print(f"{'='*60}")
  print(f"Move accuracy: {accuracy:.2%}")
  print(f"  95% CI: [{stats['accuracy_ci'][0]:.2%}, {stats['accuracy_ci'][1]:.2%}]")
  print(f"Average Kendall's tau: {taus:.4f}")
  print(f"  95% CI: [{stats['tau_ci'][0]:.4f}, {stats['tau_ci'][1]:.4f}]")
  print(f"  p-value: {stats['p_value']:.4f}")
  print(f"{'='*60}")
  print(f"Compared {len(data)} positions' move rankings in {hours} hours, {minutes} minutes, {seconds} seconds")

if __name__ == '__main__':
  main()