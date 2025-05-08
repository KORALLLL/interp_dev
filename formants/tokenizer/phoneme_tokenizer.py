import json

class PhonemeTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]

    def encode(self, phoneme_list):
        return [self.vocab.get(p, self.unk_token_id) for p in phoneme_list]

    def decode(self, id_list):
        return [self.inv_vocab.get(i, "<unk>") for i in id_list]
