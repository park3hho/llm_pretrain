import tiktoken # pip install tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Harry Potter was a wizard."

tokens = tokenizer.encode(text)

print("글자수:", len(text), "토큰수", len(tokens))
print(tokens)
print(tokenizer.decode(tokens))
for t in tokens:
    print(f"{t}\t -> {tokenizer.decode([t])}")

## DataLoader

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # token_ids = tokenizer.encode("<|endoftext|>" + txt, allowed_special={"<|endoftext|>"})
        token_ids = tokenizer.encode(txt)

        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] # Next Token Prediction
            # Input: 입력값, Target: 출력값, 입력값에 따라 출력값이 학습한 가중치에 따라 유사하게 나오게 만든다.
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) # token qunatities

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] # input

# with open("cleaned_한글문서.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
with open("../DataSet/Cleaned_Dataset/cleaned_02 Harry Potter and the Chamber of Secrets.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
    txt = file.read()

dataset = MyDataset(txt, max_length = 32, stride = 4)
print(len(dataset))
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

# 주의: 여기서는 코드를 단순화하기 위해 test, valid는 생략


# Tokenization Check #
# Batch refers to the batch of samples that are entered into the model at a time.
# iterator: traversal object
dataiter = iter(train_loader)
x,y = next(dataiter)
print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))