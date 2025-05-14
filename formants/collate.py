import torch

def my_collate_fn(batch):
    phoneme_tokens = [item['phoneme_tokens'] for item in batch]
    formants = [item['formants'] for item in batch]
    speech_embeddings = [item['speech_embedding'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]

    max_len = max([t.shape[0] for t in phoneme_tokens])

    padded_phoneme_tokens = []
    padded_formants = []
    attention_masks = []

    for tokens, frm in zip(phoneme_tokens, formants):
        pad_len = max_len - tokens.shape[0]
        padded_tokens = torch.cat([tokens, torch.full((pad_len,), fill_value=0, dtype=torch.long)])
        padded_formant = torch.cat([frm, torch.zeros((pad_len, 3), dtype=torch.float32)])
        attention_mask = torch.cat([torch.ones(tokens.shape[0], dtype=torch.float32), torch.zeros(pad_len, dtype=torch.float32)])

        padded_phoneme_tokens.append(padded_tokens)
        padded_formants.append(padded_formant)
        attention_masks.append(attention_mask)

    return {
        'phoneme_tokens': torch.stack(padded_phoneme_tokens),
        'formants': torch.stack(padded_formants),
        'attention_mask': torch.stack(attention_masks),
        'speech_embedding': torch.stack(speech_embeddings),
        'audio_path': audio_paths
    }
