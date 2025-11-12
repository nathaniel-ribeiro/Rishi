from rishi import Rishi, DummyModel

# Fake dependencies
class FakePikafish:
    def get_legal_moves(self, fen):
        return ["a2a3", "a2a4", "b1c3", "g1f3"]
    def get_fen_after_fen_and_moves(self, fen, move):
        return f"{fen} {move}"
    def set_position(self, fen): pass
    def get_best_move(self, t): return "a2a4"
    def new_game(self): pass
    def play_moves(self, fen, moves): return fen
    def is_checkmate(self, fen): return False

class FakeTokenizer:
    def encode(self, fen): return [1]*8  # dummy encoding

# Build dummy Rishi
r = Rishi.__new__(Rishi)  # bypass __init__
r.pikafish = FakePikafish()
r.tokenizer = FakeTokenizer()
r.device = "cpu"
r.temperature = 0.5
r.top_p = 0.9
r.model = DummyModel("logit")  # simulate your new model output

# Run test
move = r.get_best_move("startpos")
print("Selected move:", move)