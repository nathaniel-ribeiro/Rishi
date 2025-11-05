from rishi import Rishi

if __name__ == "__main__":
    rishi = Rishi("models/rishi.pt")
    cur_position_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    while True:
        cmd = input()
        tokens = cmd.split()
        if cmd == "uci":
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            cur_position_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        elif cmd == "quit":
            break
        elif tokens[0] == "position":
            if tokens[1] == "startpos" and len(tokens) == 2:
                cur_position_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
            elif tokens[1] == "startpos" and tokens[2] == "moves":
                cur_position_fen = rishi.get_fen_after_moves(tokens[3:])
            # TODO: fix bug where FEN gets chopped up into multiple tokens
            elif tokens[1] == "fen" and tokens[3] == "moves":
                cur_position_fen = rishi.get_fen_after_fen_and_moves(tokens[2], tokens[4:])
        elif tokens[0] == "go":
            best_move = rishi.get_best_move(cur_position_fen)
            print("best move: \t", best_move)
        elif cmd == "d":
            print(cur_position_fen)