import re
from oracle import *
import threading
from config import PIKAFISH_THREADS, PATH_TO_NNUE

option_info = {
  "Threads": {"type": "spin", "default": 1, "min": 1, "max": 1024},
  "MultiPV": {"type": "spin", "default": 1, "min": 1, "max": 256},
  "Skill Level": {"type": "spin", "default": 20, "min": 0, "max": 20}
}
options = { option: option_info[option]["default"] for option in option_info}

def set_option(option, value):
  pass

def uci_handler():
  print("id name Rishi 1.1")
  print("id author the Rishi developers\n")
  for option in option_info:
    attrs = option_info[option]
    attr_string = ""
    for attr in attrs:
      attr_string += f" {attr} {attrs[attr]}"
    print(f"option name {option}{attr_string}")
  print("uciok")

def setoption_handler(cmd):
  cmd_string = ' '.join(cmd)
  option = re.search(r"(?<=name ).*(?= value)", cmd_string).group()
  value = int(cmd[cmd.index("value") + 1])
  if option in options and option_info[option]["min"] <= value <= option_info[option]["max"]:
    options[option] = value
  set_option(option, value)
  print(f"Info string: {option} = {value}")

def live_print_engine_output(engine):
  def printer():
    while True:
      line = engine.output_queue.get()
      print(line)
  
  threading.Thread(target=printer, daemon=True).start()

engine = PikafishEngine(PIKAFISH_THREADS)

engine.send(f"setoption name EvalFile value {PATH_TO_NNUE}")
engine.send("isready")
engine._wait_for("readyok")

# Start live output thread
live_print_engine_output(engine)

quit_flag = False
while not quit_flag:
  try:
    cmd = input().strip()
    if not cmd:
        continue

    # Handle quit
    if cmd.lower() == "quit":
        quit_flag = True
        engine.quit()
        break

    # Send UCI command to engine
    engine.send(cmd)

  except KeyboardInterrupt:
    quit_flag = True
    engine.quit()
    break
  