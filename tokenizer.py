import re
import numpy as np

class BoardTokenizer:
    def __init__(self, expected_seq_len):
        self.expected_seq_len = expected_seq_len
        self.metadata_regex = re.compile(r"^([wb]) - - (\d+) (\d+)$")

        self.vocab = ['K', 'A', 'E', 'R', 'C', 'N', 'P',
                      'k', 'a', 'e', 'r', 'c', 'n', 'p',
                      '0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9', '0', 'w', 'b', '.',
                      '[CLS]']
        self.vocab_size = len(self.vocab)
        self.token_to_idx = dict(zip(self.vocab, range(len(self.vocab))))

    def encode(self, fen):
        board, metadata = fen.split(" ", 1)
        rows = board.split("/")
        # replace digit n with n empty square tokens
        rows = "".join([re.sub(r'\d', lambda m: "." * int(m.group(0)), row) for row in rows])
        # use e and E for black and red elephants to disambiguate from whose turn to move
        rows = rows.replace("b", "e").replace("B", "E")
            
        m = self.metadata_regex.match(metadata)
        whose_move = m.group(1)
        # left pad with zeros to ensure fixed length
        capture_clock = m.group(2).zfill(3)
        halfmove_clock = m.group(3).zfill(3)
            
        # prepend sequence with [CLS] token
        tokenized = ['[CLS]'] + list(rows) + list(whose_move) + list(capture_clock) + list(halfmove_clock)
        tokenized = np.array([self.token_to_idx[token] for token in tokenized])
        assert tokenized.shape[0] == self.expected_seq_len, f"Expected tokenized FEN to be {self.expected_seq_len} chars, got {tokenized.shape[0]}"
        return tokenized