import numpy as np
import csv
import random

import config
from oracle import PikafishEngine

class Count:
  def __init__(self, oracle, data):
    self.oracle = oracle
    self.data = data
    self.total_legal_moves = 0
    self.trials = 0

    

  def count_moves(self):
    self.oracle.new_game()
    print_interval = max(10, len(self.data) // 10)
    
    for fen in self.data:
      self.total_legal_moves += len(self.oracle.get_legal_moves(fen))
      self.trials += 1

      if self.trials % print_interval == 0:
        print(f'{self.trials} Trials Avg: {self.total_legal_moves / self.trials}')

    return self.total_legal_moves / self.trials

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

  DATA_PATH = './data/test.csv'
  NUM_TRIALS = 10_000
  print('Loading data')
  data = load_fens(DATA_PATH, NUM_TRIALS)
  
  print('Instantiating engine')
  oracle = PikafishEngine(config.PIKAFISH_THREADS)

  print('Counting moves...\n')
  counter = Count(oracle, data)
  print(f'Avg moves: {counter.count_moves()}')

if __name__ == '__main__':
  main()