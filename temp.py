from oracle import PikafishEngine

engine = PikafishEngine(threads=1)
result = engine.get_all_legal_move_successors(fen = "3ak1b2/4a4/1c2bP3/1N7/4C4/9/2n1Pr3/5C3/4A4/2B1KA1Rc w - - 1 2")
for r in result:
    print(r)