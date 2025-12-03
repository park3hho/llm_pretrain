이거 왜이리 용량을 많이 잡아먹냐,,,?


## 쓰로틀링 여부 확인
`nvidia-smi -q -d PERFORMANCE`
```
C:\Users\iwill> nvidia-smi -q -d PERFORMANCE

==============NVSMI LOG==============

Timestamp                                 : Wed Dec  3 06:15:36 2025
Driver Version                            : 576.88
CUDA Version                              : 12.9

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Performance State                     : P0 # 최고 성능 모드
    Clocks Event Reasons
        Idle                              : Not Active
        Applications Clocks Setting       : Not Active #ㅅ 수동 오버/언더 클럭
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active 
            HW Thermal Slowdown           : Not Active # 발열 제한 모드
            HW Power Brake Slowdown       : Not Active # 전력 제한 모드
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    Clocks Event Reasons Counters
        SW Power Capping                  : 3737587 us # 소프트웨어가 전력 때문에 제한한 시간
        Sync Boost                        : 0 us
        SW Thermal Slowdown               : 0 us # 소프트웨어가 온도 때문에 제한한 시간
        HW Thermal Slowdown               : 0 us 
        HW Power Braking                  : 0 us
    Sparse Operation Mode                 : N/A 

```

## GPU Clock & Power Draw
`nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv`

```
C:\Users\iwill>nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv

clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W]
2827 MHz, 14001 MHz, 46.83 W
```

## 20 Epochs Result 

```
root@0e8f15392a1f:/app/Tokenize & Dataload# python trainingCode.py 

글자수: 26 토큰수 6
[18308, 14179, 373, 257, 18731, 13]
Harry Potter was a wizard.
18308    -> Harry
14179    ->  Potter
373      ->  was
257      ->  a
18731    ->  wizard
13       -> .

# of tokens in txt: 130520

t you guessed yet, Harry Potter?” said Riddle softly. “Ginny Weasley opened the Chamber of Secrets. She strangled the school
 you guessed yet, Harry Potter?” said Riddle softly. “Ginny Weasley opened the Chamber of Secrets. She strangled the school ro

Device: cuda
[Step 0] Tokens seen: 4,096, Total elapsed: 2.90 sec

Epoch 1 completed.
 ➤ Avg Loss: 4.3906
 ➤ Epoch Time: 1697.83 sec
 ➤ Total Time Elapsed: 1697.83 sec


Epoch 2 completed.
 ➤ Avg Loss: 2.2207
 ➤ Epoch Time: 1699.01 sec
 ➤ Total Time Elapsed: 3400.28 sec


Epoch 3 completed.
 ➤ Avg Loss: 0.7964
 ➤ Epoch Time: 1699.46 sec
 ➤ Total Time Elapsed: 5103.48 sec

[Step 1000] Tokens seen: 4,100,096, Total elapsed: 6704.11 sec

Epoch 4 completed.
 ➤ Avg Loss: 0.3928
 ➤ Epoch Time: 1699.16 sec
 ➤ Total Time Elapsed: 6806.53 sec


Epoch 5 completed.
 ➤ Avg Loss: 0.3051
 ➤ Epoch Time: 1711.41 sec
 ➤ Total Time Elapsed: 8521.63 sec


Epoch 6 completed.
 ➤ Avg Loss: 0.2708
 ➤ Epoch Time: 1710.01 sec
 ➤ Total Time Elapsed: 10235.44 sec


Epoch 7 completed.
 ➤ Avg Loss: 0.2536
 ➤ Epoch Time: 1709.65 sec
 ➤ Total Time Elapsed: 11948.62 sec

[Step 2000] Tokens seen: 8,196,096, Total elapsed: 13451.04 sec

Epoch 8 completed.
 ➤ Avg Loss: 0.2446
 ➤ Epoch Time: 1709.67 sec
 ➤ Total Time Elapsed: 13661.88 sec


Epoch 9 completed.
 ➤ Avg Loss: 0.2370
 ➤ Epoch Time: 1709.81 sec
 ➤ Total Time Elapsed: 15375.46 sec


Epoch 10 completed.
 ➤ Avg Loss: 0.2309
 ➤ Epoch Time: 1709.80 sec
 ➤ Total Time Elapsed: 17089.03 sec


Epoch 11 completed.
 ➤ Avg Loss: 0.2267
 ➤ Epoch Time: 1709.99 sec
 ➤ Total Time Elapsed: 18802.53 sec

[Step 3000] Tokens seen: 12,292,096, Total elapsed: 20197.29 sec

Epoch 12 completed.
 ➤ Avg Loss: 0.2213
 ➤ Epoch Time: 1709.65 sec
 ➤ Total Time Elapsed: 20515.76 sec


Epoch 13 completed.
 ➤ Avg Loss: 0.2173
 ➤ Epoch Time: 1709.68 sec
 ➤ Total Time Elapsed: 22229.71 sec


Epoch 14 completed.
 ➤ Avg Loss: 0.2151
 ➤ Epoch Time: 1710.30 sec
 ➤ Total Time Elapsed: 23943.61 sec


Epoch 15 completed.
 ➤ Avg Loss: 0.2126
 ➤ Epoch Time: 1709.76 sec
 ➤ Total Time Elapsed: 25657.09 sec

[Step 4000] Tokens seen: 16,388,096, Total elapsed: 26944.42 sec

Epoch 16 completed.
 ➤ Avg Loss: 0.2098
 ➤ Epoch Time: 1709.80 sec
 ➤ Total Time Elapsed: 27370.49 sec


Epoch 17 completed.
 ➤ Avg Loss: 0.2072
 ➤ Epoch Time: 1868.57 sec
 ➤ Total Time Elapsed: 29242.58 sec


Epoch 18 completed.
 ➤ Avg Loss: 0.2050
 ➤ Epoch Time: 1710.79 sec
 ➤ Total Time Elapsed: 30957.15 sec


Epoch 19 completed.
 ➤ Avg Loss: 0.2034
 ➤ Epoch Time: 1710.77 sec
 ➤ Total Time Elapsed: 32671.64 sec

[Step 5000] Tokens seen: 20,484,096, Total elapsed: 33859.87 sec

Epoch 20 completed.
 ➤ Avg Loss: 0.2009
 ➤ Epoch Time: 1733.15 sec
 ➤ Total Time Elapsed: 34409.61 sec


Training Finished!
Total Training Time: 34414.43 sec

```

## 1️⃣ 1스텝 총 파라미터 수

모델 구성:

- Embedding: VOCAB_SIZE × EMB_DIM = 50257 × 768 ≈ 38.6M
- Positional Embedding: CONTEXT_LENGTH × EMB_DIM = 128 × 768 ≈ 0.1M
- Transformer 블록 1개:
  - MHA:
    - Q/K/V: 768×768×3 = 1.77M
    - Out_proj: 768×768 ≈ 0.59M
  - FFNN:
    - 768→3072: 768×3072 ≈ 2.36M
    - 3072→768: 3072×768 ≈ 2.36M
  - LayerNorm: 768×2 = 1.5k (무시 가능)
    - 블록 12개 → 12 × (MHA + FFNN) ≈ 12 × (1.77+0.59+2.36+2.36)M ≈ 79.7M
    - 마지막 Linear: EMB_DIM × VOCAB_SIZE = 768×50257 ≈ 38.6M

총 파라미터 ≈ 38.6 + 0.1 + 79.7 + 38.6 ≈ 157M
> 정확히 계산하면 약 1억 5천 7백만 파라미터 정도 됩니다.
 
## 2️⃣ 한 스텝 FLOPs (대략)

- 한 토큰에 대해 Linear 한 번: 2×(input_dim × output_dim) FLOPs (곱셈+덧셈)

- 배치 128, seq_len 128 → 16,384 토큰
MHA 한 블록

1. Q/K/V: 3×(768×768) × tokens = 3 × 768² × 16384 ≈ 30.6G FLOPs

2. Attention (QK^T softmax V):
- QK^T: seq_len² × head_dim × batch_size × num_heads ≈ 128² × 64 × 128 × 12 ≈ 16.2G FLOPs
- V 곱하기 weighted sum: 비슷하게 ≈ 16G FLOPs
3. Out_proj: 768×768 × tokens ≈ 10.2G FLOPs  

MHA 합계 ≈ 30.6 + 16 + 16 + 10.2 ≈ 72.8G FLOPs

FFNN 한 블록
- 768→3072→768, 2×(768×3072 + 3072×768) × tokens ≈ 2 × 4.72M × 16,384 ≈ 154G FLOPs
LayerNorm, Residual: 무시 가능 (수치 작음)

블록 12개
- (72.8 + 154)G × 12 ≈ 2,668G ≈ 2.7 TFLOPs per step

마지막 Linear
- 768×50257 × tokens × 2 ≈ 1,262M × 2 ≈ 2,624G ≈ 2.6 TFLOPs

✅ 정리

- 총 파라미터: 약 157M  
- 한 스텝 FLOPs: 약 5.3 TFLOPs (12 Layer + output)  
- 배치 128, seq_len 128 기준  

> 즉, 배치 하나 돌릴 때 CPU/GPU가 약 5조 3천억 번 연산을 수행하는 셈입니다.


## 3. 1 Epochs 251 Steps
- 총 토큰 수: 130,520
- max_length = 32, stride = 4  
- batch_size = 128  

```
num_sequence = (total_token - max_length) / stride + 1 = (130,520 - 32) / 4 + 1= 32,122  
steps = num_seq / batch_size = 32122 / 128 = 251 
```

> 한 에팍에 251 스텝이 진행된다.
 
## X. 정리
### 1️⃣ 모델 및 학습 세팅
- Vocab Size (토큰 종류): 50,257
- Context Length (시퀀스 길이): 32 (DataLoader 기준)
- Batch Size: 128
- Embedding Dim: 768
- Number of Layers: 12
- Number of Attention Heads: 12
- FFNN Hidden Dim: 4 × 768 = 3,072
- Epochs: 20

### 2️⃣ 파라미터 수 계산
| 구성                         | 파라미터 수 (≈)          |
| -------------------------- | ------------------- |
| Token Embedding            | 50257 × 768 ≈ 38.6M |
| Positional Embedding       | 32 × 768 ≈ 0.025M   |
| Transformer Block ×12      | 79.7M               |
| Final Linear (Output Head) | 768 × 50257 ≈ 38.6M |
| **총합**                     | 약 157M              |

### 3️⃣ FLOPs 계산 (한 스텝 기준, 1 배치)
#### 3-1. Embedding
- Input: [batch, seq_len] = [128, 32]
- Token Embedding: 128 × 32 × 768 ≈ 3.1M FLOPs
- Positional Embedding: negligible

#### 3-2. Multi-Head Attention (MHA) per layer
- Q/K/V projections: 3 × (768×768) × seq_len × batch ≈ 3 × 768² × 32 × 128  
→ 3 × 589,824 × 32 × 128 ≈ 72.3B FLOPs

- Attention score matmul: seq_len² × head_dim × batch × n_heads
  - 32² × (768/12) × 128 × 12 ≈ 12.6M (작은 편)
- Output projection: 768 × 768 × 32 × 128 ≈ 2.4B
> MHA 합계 ≈ 74.7B FLOPs per layer

#### 3-3. Feed-Forward (FFNN) per layer
- Linear1: 768 → 3072: 768 × 3072 × seq_len × batch ≈ 3.1B
- Linear2: 3072 → 768: 3072 × 768 × seq_len × batch ≈ 3.1B
> FFNN 합계 ≈ 6.2B FLOPs per layer

#### 3-4. Transformer Block
- MHA + FFNN ≈ 74.7B + 6.2B ≈ 80.9B FLOPs per layer
- 12 layers → 12 × 80.9B ≈ 970B FLOPs

#### 3-5. Final Linear
- Output: 768 → 50257 × seq_len × batch ≈ 128 × 32 × 768 × 50257 ≈ 157B FLOPs

#### ✅ 한 스텝 총 FLOPs
- Embedding + 12 layers + Final Linear ≈ 3.1M + 970B + 157B ≈ 1.13T FLOPs per step

### 4️⃣ 20 Epoch 기준 총 연산량

- Dataset size: 130,520 tokens
- Chunking: max_len 32, stride 4 → (130,520 - 32) // 4 ≈ 32,122 steps per epoch
- Total steps for 20 epochs ≈ 20 × 32,122 ≈ 642,440 steps
- 총 FLOPs ≈ 1.13T × 642,440 ≈ 7.25 × 10¹⁷ FLOPs 

즉, 20 epoch 동안 7.25e17 floating-point 연산량을 수행한 셈입니다. (70경)