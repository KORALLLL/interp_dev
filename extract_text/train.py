import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
import pickle
import chromadb
import wandb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


wandb.login()
wandb.init(project="inrerp", name="t5_text_extraction")



class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=768):
        super(SpeechEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x.unsqueeze(1)

class Speech2TextModel(nn.Module):
    def __init__(self, t5_model_name="sberbank-ai/ruT5-base"):
        super(Speech2TextModel, self).__init__()
        self.speech_encoder = SpeechEncoder(input_dim=256, hidden_dim=768)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    
    def forward(self, speech_embeddings, labels=None):
        encoder_embeds = self.speech_encoder(speech_embeddings)
        attention_mask = torch.ones(encoder_embeds.shape[:2], dtype=torch.long, device=encoder_embeds.device)
        print(attention_mask.shape)
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return outputs

class SpeechDataset(Dataset):
    def __init__(self, path_to_pkl,path_to_db,collection_name, tokenizer, train=True ,max_target_length=128):
        with open(path_to_pkl, "rb") as f:
            self.data = pickle.load(f)
            
        if train:
            self.data = self.data['train']
        else:
            self.data = self.data['dev']
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        
        self.client = chromadb.PersistentClient(path=path_to_db)
        self.collection_name = collection_name
        self.coll = self.client.get_collection(self.collection_name)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        result = self.coll.get(ids=[self.data[idx]['id']], include=["embeddings"])
        embedding = result["embeddings"][0] if result["embeddings"] is not None else None

        tokenized = self.tokenizer(
            text, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        tokenized = {key: val.squeeze(0) for key, val in tokenized.items()}
        labels = tokenized["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "speech_embeddings": torch.Tensor(embedding),
            "labels": labels,
            'text': text
        }
class CustomSpeechDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features):
        speech_embeddings = torch.stack([f["speech_embeddings"] for f in features])
        
        labels = [f["labels"] for f in features]
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]
        
        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100

        return {"speech_embeddings": speech_embeddings, "labels": padded_labels}

class InferenceCallback(TrainerCallback):
    def __init__(self, sample_embedding, original_text):
        self.sample_embedding = sample_embedding
        self.original_text = original_text

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 200 == 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("processing_class")
            device = args.device
            
            sample_emb = self.sample_embedding.to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                encoder_embeds = model.speech_encoder(sample_emb)
                attention_mask = torch.ones(encoder_embeds.shape[:2], dtype=torch.long, device=encoder_embeds.device)
                encoder_outputs = model.t5.encoder(
                    inputs_embeds=encoder_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                generated_ids = model.t5.generate(
                    input_embeds=encoder_outputs, 
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)


            wandb.log({"original": wandb.Html(self.original_text)})
            wandb.log({"generated":  wandb.Html(output_text)})

            model.train()
        return control
    
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

t5_model_name = "sberbank-ai/ruT5-base"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

path_to_pkl = "/path/to/pkl.pkl"
path_to_db = "/path/to/db"
db_collection_name = "collection_name"

train_dataset = SpeechDataset(path_to_pkl,path_to_db,db_collection_name, tokenizer)
val_dataset = SpeechDataset(path_to_pkl,path_to_db,db_collection_name, tokenizer,train=False)

model = Speech2TextModel(t5_model_name=t5_model_name)

print(count_parameters(model))
data_collator = CustomSpeechDataCollator(tokenizer, max_length=128)

training_args = TrainingArguments(
    output_dir="./speech2text_ru_model",
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=10,
)
sample_embedding = train_dataset[0]['speech_embeddings'].to('cuda:0')


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[InferenceCallback(sample_embedding, train_dataset[0]['text'])]
)
trainer.args.save_safetensors = False
trainer.train()

