from tokenizeNdataload import train_loader
from NeuralNetworkModel import GPTModel

import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(123)
model = GPTModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

tokens_seen, global_step = 0, -1
losses = []

start_time = time.time()  # 전체 학습 시간 측정 시작

for epoch in range(20):
    model.train()
    epoch_loss = 0

    epoch_start = time.time()  # epoch 시간 측정 시작

    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()

        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        tokens_seen += input_batch.numel()
        global_step += 1

        if global_step % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {global_step}] Tokens seen: {tokens_seen:,}, Total elapsed: {elapsed:.2f} sec")

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    epoch_time = time.time() - epoch_start  # epoch 소요 시간

    print(f"\nEpoch {epoch + 1} completed.")
    print(f" ➤ Avg Loss: {avg_loss:.4f}")
    print(f" ➤ Epoch Time: {epoch_time:.2f} sec")
    print(f" ➤ Total Time Elapsed: {time.time() - start_time:.2f} sec\n")

    torch.save(model.state_dict(), f"model_{str(epoch + 1).zfill(3)}.pth")

print("\nTraining Finished!")
print(f"Total Training Time: {time.time() - start_time:.2f} sec")
