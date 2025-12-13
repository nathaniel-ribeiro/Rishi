import pandas as pd
from oracle import PikafishEngine
import config
import time
import os
from argparse import ArgumentParser

SUMMARY_DIR = os.path.expanduser("~/Rishi/data/puzzle_sweeps")
file_dir = "~/Rishi/data/puzzles_data_5000.csv"

#For puzzle test sweep using Slurm
def write_summary_row(thinktime, numThreads, success, alt_success, fail, elapsed):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    total = success + alt_success + fail
    acc = (success + alt_success) / total * 100 if total else 0.0
    # unique file per combo
    out = os.path.join(SUMMARY_DIR, f"summary_tt{thinktime}_th{numThreads}.csv")
    pd.DataFrame([{
        "thinktime_ms": thinktime,
        "threads": numThreads,
        "success": success,
        "alt_success": alt_success,
        "fail": fail,
        "total": total,
        "accuracy_pct": acc,
        "elapsed_sec": elapsed
    }]).to_csv(out, index=False)

#Test all puzzles
def Full_Puzzle_Test(engine, save_results = False, thinktime = 50, numThreads = 8):
    #Read csv
    df = pd.read_csv(file_dir)

    #Start engine
    Success = 0
    Alternate_Success = 0
    Fail = 0
    Puzzle_Count = 1
    results = []
    start_time = time.time()

    #For each puzzle, generate fen
    #Then, calculate answers
    for pid, group in df.groupby("puzzle_id"):
        group = group.sort_values("index", kind="stable")
        engine.new_game()
        preset = group.loc[~group["is_solution_move"], "move"].tolist()
        solution = group.loc[group["is_solution_move"], "move"].tolist()
        #Only applies to our current puzzle set since everything is "Mate in N" puzzles
        category = int(group["category"].iloc[0][-1:])

        fen = engine.get_fen_after_moves(preset)

        puzzle_solved = Solve_Puzzle(engine,fen,solution,category,thinktime)

        result_log = "Puzzle "+ str(Puzzle_Count) + " | pid = " + pid #+ " | category = " + group["category"].iloc[0]

        if(puzzle_solved): 
            Success += 1
            print(result_log + " | Success")
            results.append({"puzzle_index": Puzzle_Count, "pid": pid, "category": group["category"].iloc[0], "result": "Success"})
        elif(Check_Alternate_Answer(engine,fen,category,thinktime)):
            Alternate_Success += 1
            print(result_log + " | Success(Alternate Solution)")
            results.append({"puzzle_index": Puzzle_Count, "pid": pid, "category": group["category"].iloc[0], "result": "Success(Alternate Solution)"})
        else: 
            Fail += 1
            print(result_log + " | Fail")
            results.append({"puzzle_index": Puzzle_Count, "pid": pid, "category": group["category"].iloc[0], "result": "Fail"})
        Puzzle_Count += 1
    
    elapsed = str(time.time()-start_time)

    print("Success : "+ str(Success))
    print("Alternate Success : "+ str(Alternate_Success))
    print("Fail : "+ str(Fail))
    print("Accuracy : " + str((Success+Alternate_Success)/(Success+Alternate_Success+Fail)*100) + "%")
    print("Elapsed Time : " + elapsed + "s")

    if(save_results):
        out_path = os.path.expanduser("~/Rishi/data/pikafish_puzzle_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
    
    write_summary_row(thinktime, numThreads, Success, Alternate_Success, Fail, elapsed)
    engine.quit()

#Test 1 puzzle using given puzzle id
def Puzzle_Test(engine, pid, thinktime = 50, numThreads = 8):
    #Read csv
    df = pd.read_csv(file_dir)

    #Start engine
    start_time = time.time()

    puzzle = df[df["puzzle_id"] == pid].copy()
    if puzzle.empty:
        raise ValueError(f"puzzle_id {pid!r} not found")
    
    puzzle = puzzle.sort_values("index", kind="stable")
    engine.new_game()
    preset = puzzle.loc[~puzzle["is_solution_move"], "move"].tolist()
    solution = puzzle.loc[puzzle["is_solution_move"], "move"].tolist()
    #Only applies to our current puzzle set since everything is "Mate in N" puzzles
    category = int(puzzle["category"].iloc[0][-1:])

    fen = engine.get_fen_after_moves(preset)
    print(fen)
    
    puzzle_solved = Solve_Puzzle(engine,fen,solution,category,thinktime,True)
    print(category)

    if(puzzle_solved): print("Success")
    elif(Check_Alternate_Answer(engine,fen,category,thinktime)): print("Success(Alternate Solution)")
    else:print("Fail")
        
    engine.quit()

#Solve a single puzzle with the fen and solution as parameters
def Solve_Puzzle(engine,fen,solution,category, thinktime, single_test = False):
    engine.new_game()
    engine.set_position(fen)
    puzzle_solved = True
    for i in range(0,len(solution),2):
        best_move = engine.get_best_move(thinktime)
        
        #For Debugging Issues
        if(single_test):
            print("Solution : " + solution[i])
            print("Pikafish Best Move : " + best_move)
            print()
        if best_move != solution[i]:
            puzzle_solved = False
            break
        engine.play_moves(fen,tuple(solution[:i+2]))
     
    return puzzle_solved

#Check if Pikafish's answer is still Mate in N
def Check_Alternate_Answer(engine,fen,category,thinktime):
    engine.new_game()
    engine.set_position(fen)
    for i in range(category):
        best_move = engine.get_best_move(thinktime)
        fen = engine.play_moves(fen,[best_move])
        #Opponent's optimal move
        if i!=category-1:
            best_move = engine.get_best_move(thinktime)
            fen = engine.play_moves(fen,[best_move])
    return engine.is_checkmate(thinktime)

#-m : decide which model to use(Pikafish or Rishi) (required)
#-s : run all and save results in csv
#-t : for running all tests with given thinktime and threads
#-p puzzle_id : run just that puzzle printint each move
def main():
    parser = ArgumentParser(description="Puzzle Solver")
    parser.add_argument("-m", "--model", required=True, choices=["Pikafish","Rishi"], help="select model to use: Pikafish or Rishi")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--save", action="store_true", help="solve all puzzles while saving results as csv")
    group.add_argument("-t", "--test", nargs=2, type=int, metavar=("thinktime", "numThreads"), help="solve all puzzles without saving; requires THINKTIME_MS and THREADS")
    group.add_argument("-p", "--puzzle-id", dest="puzzle_id", help="solve a single puzzle_id")
    args = parser.parse_args()
    model = args.model

    if(model=="Pikafish"):
        threads = args.test[1] if args.test else 8
        engine = PikafishEngine(threads=threads)
    elif(model=="Rishi"):
        pass

    if args.puzzle_id:
        Puzzle_Test(engine, args.puzzle_id)
    elif args.save:
        Full_Puzzle_Test(engine, True)
    elif args.test:
        thinktime, numThreads = args.test  
        if thinktime <= 0 or numThreads <= 0:
            raise ValueError("THINKTIME_MS and THREADS must be positive integers.")
        Full_Puzzle_Test(engine, False, thinktime)
    else:
        Full_Puzzle_Test(engine) 

if __name__ == "__main__":
    main()