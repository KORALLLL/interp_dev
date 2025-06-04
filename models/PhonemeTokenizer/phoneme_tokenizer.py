import json

class PhonemeTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["<pad>"]

    def encode(self, phoneme_list):
        return [self.vocab[p] for p in phoneme_list]

    def decode(self, id_list):
        return [self.inv_vocab[i] for i in id_list]
