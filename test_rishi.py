from rishi import Rishi

if __name__ == "__main__":
    engine = Rishi("models/rishi.pt")
    print(engine.get_best_move("4ka3/4a4/4b4/p2rC1R1p/2p6/9/P3P3P/n1C3N2/4A4/1cBAK4 w - - 3 26"))