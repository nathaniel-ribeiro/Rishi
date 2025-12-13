from rishi import Rishi

if __name__ == "__main__":
    rishi = Rishi("models/rishi.pt")
    cur_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    engine_is_red = input("Enter 1 if the engine is playing red, 2 if engine is black: \t") == "1"
    red_turn = True
    while True:
        move_played = None
        if red_turn:
            if engine_is_red:
                move_played = rishi.get_best_move(cur_fen)
                cur_fen = rishi.get_fen_after_fen_and_moves(cur_fen, move_played)
            else:
                move_played = input("Enter a move in long algebraic notation: \t")
                cur_fen = rishi.get_fen_after_fen_and_moves(cur_fen, move_played)
        else:
            if not engine_is_red:
                move_played = rishi.get_best_move(cur_fen)
                cur_fen = rishi.get_fen_after_fen_and_moves(cur_fen, move_played)
            else:
                move_played = input("Enter a move in long algebraic notation: \t")
                cur_fen = rishi.get_fen_after_fen_and_moves(cur_fen, move_played)
        
        print(f"Move played: \t {move_played}")
        print(f"FEN: {cur_fen}")
        red_turn = not red_turn