from scipy.stats import spearmanr
import numpy as np

# takes 1d numpy array of best moves from both engines and scores rishi's moves
# against pikafish's 
def calculate_move_accuracy(pi_bestmoves, ri_bestmoves):
  return np.mean(pi_bestmoves == ri_bestmoves)

# takes 2d numpy array of move rankings from both engines and calculates
# average spearman correlation 
def calculate_spearman_correlation(pi_ranks, ri_ranks):
  n, m = pi_ranks.shape

  # encode strings as integers
  unique_moves = np.unique(np.concatenate([pi_ranks.flatten(), ri_ranks.flatten()]))
  move_to_int = {move: idx for idx, move in enumerate(unique_moves)}
  pi_encoded = np.vectorize(move_to_int.get)(pi_ranks)
  ri_encoded = np.vectorize(move_to_int.get)(ri_ranks)

  # map moves to rank
  ri_rank_map = np.zeros((m, len(unique_moves)), dtype=int)
  ri_rank_map[np.arange(n)[:, None], ri_encoded] = np.arange(m)

  pi_to_ri_ranks = ri_rank_map[np.arange(n)[:, None], pi_encoded]

  # spearman formula
  d = np.arange(m) - pi_to_ri_ranks
  d2_sum = np.sum(d**2, axis=1)
  rho_per_position = 1 - (6 * d2_sum) / (m * (m**2 - 1))
  
  return np.mean(rho_per_position)

def collect_evaluation_data():
  pass

