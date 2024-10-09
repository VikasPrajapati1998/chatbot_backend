# model.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.text[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }



class GPT2TextGenerator:
    def __init__(self, model_path='fine_tuned_gpt2.pth', model_name='gpt2-medium'):
        # Load the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # Load the fine-tuned model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.get_device()))
        self.model.eval()  # Set the model to evaluation mode
        self.device = self.get_device()
        self.model.to(self.device)

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_text(self, prompt, length=200):
        if not prompt:
            return ""  # Handle empty prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_length=length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class GPT2FineTuner:
    def __init__(self, model_name='gpt2-medium', max_len=512, batch_size=16, learning_rate=1e-5):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.get_device())
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_dataset(self, text):
        sequences = []
        seq_len = 200
        for i in range(0, len(text), seq_len):
            sequences.append(text[i:i + seq_len])
        return TextDataset(sequences, self.tokenizer, self.max_len)

    def fine_tune(self, dataset_text, epochs=5):
        dataset = self.create_dataset(dataset_text)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.get_device())
                attention_mask = batch['attention_mask'].to(self.get_device())

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = criterion(outputs.logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

    def save_model(self, path='fine_tuned_gpt2.pth'):
        torch.save(self.model.state_dict(), path)

    def generate_text(self, prompt, length=200):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.get_device())
        output = self.model.generate(input_ids, max_length=length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def fine_tune_gpt2_on_dataset(dataset_text, epochs=5):
    # Initialize GPT2FineTuner
    fine_tuner = GPT2FineTuner(model_name='gpt2-medium', max_len=512, batch_size=16, learning_rate=1e-5)

    # Fine-tune GPT2 model on dataset
    fine_tuner.fine_tune(dataset_text, epochs)

    # Save fine-tuned model
    fine_tuner.save_model('fine_tuned_gpt2.pth')



