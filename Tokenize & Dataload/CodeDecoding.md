# [CD] 1. Multi-Head Attention
Link: https://velog.io/@park2do/%EC%BD%94%EB%93%9C-%ED%8C%8C%ED%97%A4%EC%B9%98%ED%82%A4-Multi-Head-Attention
gg
Transformer의 핵심 구성요소인 “Multi-Head Attention” (다중 헤드 어텐션) 을 PyTorch로 직접 구현한 클래스.

## 🧩 전체 구조 요약
```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):  # 초기화
        ...
    def forward(self, x):  # 순전파
        ...
```
Transformer에서 입력 x (예: 토큰 임베딩)을 받아
“단어 간의 연관성(Attention)”을 계산하고,
그 결과를 다시 임베딩 형태로 돌려주는 모듈

## 클래스 정의 및 초기화 부문
### 1. 인자
``` 인자
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
```
- `d_in`: 입력 차원 (입력 벡터의 feature 수)
- `d_out`: 출력 차원 (전체 헤드들을 합친 차원)
- `super().__init__()`로 `nn.Module` 초기화.

> nn.Module이 뭔데?
>- nn.Module은 PyTorch에서 신경망(Neural Network)을 만들 때 사용하는 **모든 모델의 기본 클래스(base class)**

### 2. 헤드 개수 할당
``` 
assert d_out % NUM_HEADS == 0, "d_out must be divisible by n_heads"
```
- d_out이 NUM_HEADS로 나누어 떨어지는지 확인. 각 헤드의 차원(head_dim) = d_out // NUM_HEADS.
- d_out은 출력 차원 수 (예: 512)
- NUM_HEADS는 어텐션 헤드의 개수 (예: 8)
- 각 헤드가 처리할 차원을 동일하게 나누기 위해 d_out이 NUM_HEADS로 나누어떨어져야 함
- 컴퓨팅 자원 여유가 있으면 크게.

### 3. Dimension
``` dim
self.d_out = d_out
self.head_dim = d_out // NUM_HEADS
```
- 내부 저장. `head_dim`은 한 헤드가 가지는 feature 수
- 하나의 어텐션 헤드가 담당하는 차원 크기
- 예: d_out=512, NUM_HEADS=8 → head_dim=64

### 4. QKV 가중치 행렬
``` Weight Matrix
self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
```
- 입력 벡터 x를 각각 Query, Key, Value로 변환하는 선형 변환 (가중치 행렬).
- 즉, 입력 문장의 각 단어를 3가지 역할로 매핑하는 과정.
- `QKV_BIAS`는 전역 상수로 bias 사용 여부(참/거짓)

### 5. 선형 
``` out_proj, Dropout
self.out_proj = nn.Linear(d_out, d_out)
self.dropout = nn.Dropout(DROP_RATE)
```
- 여러 헤드의 출력을 다시 합쳐층 최종 출력으로 변환 / 여러 헤드를 합친 결과에
마지막으로 적용되는 출력 투사 
- Dropout으로 일부 연결을 끊어 과적합 방지(regularization)
- 훈련 데이터에 과도하게 맞추어져 있어서(이 문제를 overfitting이라고 일컫음) 일반화 능력이 떨어짐
- 이로 인해 "새로운 문장"에서 성능이 떨어질 수도 있음 그래서 Dropout을 사용.

> `d_model`이 무슨 의미인데?
>- d_model은 **Multi-Head Attention**의 전체 차원을 의미함.

### 6. 마스킹
``` 
self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))
```
- 상삼각행렬(위쪽이 1, 아래는 0)
- “미래 토큰을 보지 못하게” 하는 캐주얼 마스크 (causal mask) (GPT 같은 언어모델에서 중요)

예시 (길이 4):
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]

## 순전파 (foward)
### 1. 입력 형태
```
b, num_tokens, d_in = x.shape
```
- b: batch size [한 번에 처리하는 문장 개수, not 토큰의 개수]
- num_tokens: 한 문장(시퀀스)의 토큰 수
- d_in: 입력 임베딩 차원 > 계산 위도
> 보통 몇개의 feature로 이루어져있는데?

| 모델 종류            | 입력 임베딩 차원(d_in or d_model) | 비고                       |
| ---------------- | -------------------------- | ------------------------ |
| 소형 모델 (toy, 실습용) | 64 ~ 256                   | GPU/CPU에서도 빠르게 학습 가능     |
| 중형 모델 (기초 NLP)   | 512 ~ 1024                 | BERT base, GPT-2 small 등 |
| 대형 모델 (상용 LLM)   | 2048 ~ 12288               | GPT-3, LLaMA, Falcon 등   |
| 비전 트랜스포머(ViT)    | 256 ~ 1024                 | 이미지 patch 임베딩            |

> 임베딩 차원을 입력과 출력 두가지를 다르게 해도 되는가?
>- FFN(Feed Foward Network) 예시
> ```
> self.fc1 = nn.Linear(512, 2048)  # 확장
> self.fc2 = nn.Linear(2048, 512)  # 축소
> ```
>- (1) 비선형 변형 공간을 넓혀서 표현력 증가 (확장)
>- (2) 정보 확장 & 추상화 단계 (응용)
>- (3) 학습 안정화 (축소)

### 2. Q, K, V Mapping
``` QKV Mapping 
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)
```
- 입력 문장의 각 단어를 Query, Key, Value 벡터로 매핑

### 3. 여러 헤드로 분리
```
keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim)
queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim)
values = values.view(b, num_tokens, NUM_HEADS, self.head_dim)
```
- 이제 각 단어의 임베딩을 NUM_HEADS 개로 쪼개기
- 원래: (b, num_tokens, 512)
- 바뀐 후: (b, num_tokens, 8, 64)

> 잠깐 왜 나누어야 하는건데?
>- Multi-Head Attention의 가장 핵심적인 부분,
>- 각 헤드가 각자의 벡터 공간을 사용하여 본다면 각 부분에서 역할이 정해지기 때문에 비선형적인 데이터들을 분석 가능함.

### 4. 차원 재배열 (헤드 기준 계산하기 위해)
```
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
```
(b, num_tokens, NUM_HEADS, self.head_dim) → (b, NUM_HEADS, num_tokens, head_dim)
>- 솔직히 왜 나눠야 하는지에 대한 설명을 이해를 못했다. 
>- 계산이 안된다는데 그냥 진행해도 되는거 아닌가 싶다가도 그냥 그러려니한다.
>- 아무튼 이제 각 헤드별로 어텐션 계산 가능.
>- 아 뒤에 그냥 2,3 이렇게 써야해서 그러네

### 5. 행렬곱(Matrix Duplication) - 어텐션 스코어 계산
```attn_score
attn_scores = queries @ keys.transpose(2, 3)
```

- 이 연산은 Q × Kᵀ (행렬곱) 입니다.
- Q 토큰이 다른 토큰과 얼마나 연관되는지를 나타냄.

`shape: (b, NUM_HEADS, num_tokens, num_tokens)`
> 🔥 Q × Kᵀ 의 의미
Matrix element (i, j) = Q[i] · K[j]  
>
>→ “i번째 토큰이 j번째 토큰에 얼마나 주목하는가?”  
> → Self-attention의 핵심 의미  
여기서 나온 score를 softmax해서 가중치로 사용
 
참고 링크: https://velog.io/@park2do/%ED%96%89%EB%A0%AC%EA%B3%B1-Matrix-Multiplication

### 6. 마스크 적용 (미래 정보 차단)
```
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```
마스크가 1인 위치(=미래)는 -inf로 채워  
softmax 후 0이 되게 함.  
즉, 현재 단어는 미래 단어를 볼 수 없음.

- 보면 고장남, 창의성이 안생김

### 7. Softmax로 가중치 변환
```
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```
- 가중합을 1로 만듦.
- 발생할 수 있는 문제 두가지: Scale Explosion - Saturation

#### Scale Explosion
스케일 값이 적절한 값으로 조절되지 않아 발생하는 문제

- 일어나는 문제
  - softmax가 극단적으로 치우침
  - gradient 흐름이 불안정해짐

#### Saturation
Softmax가 너무 큰 값일 때 거의 0과 1만 나오는 현상

- 일어나는 문제
  - 미분값이 0이 됨 (gradient vanishing)
  - 학습이 안됨
  - 어텐션이 특정 토큰에 **완전 고정**

#### 해결방법
  - √dₖ로 나누기

### 8. Value를 가중합(Weighted Sum)
```
context_vec = (attn_weights @ values).transpose(1, 2)
```
- 각 토큰의 “문맥 정보”를 계산.
- (b, num_tokens, NUM_HEADS, head_dim)
> "QK"로 유사도 계산을 했는데 왜 다시 V(value)와 곱해서 최종 출력을 만드는 거지???
> 
> QKᵀ는 유사도(Attention_Score)을 구하는 단계 -> 각 토큰(단어) 간의 연관성을 파악하는 단계
>> - Q: 내가 뭘 찾고 싶은가
>> - K: 상대는 어떠한 정보를 갖고 있는가
>> - QKᵀ[i, j]는 토큰 i가 j를 얼마나 참고해야 하는가?  
>
> 즉, V는 실제 문맥을 의미하며, QKᵀ를 V와 가중합 하는 것은 **"실제 문맥을 문맥 유사도로
> 분석"**하는 것이라고 해석하면됨.

### 9. 여러 헤드 결과 합치기
```
context_vec = context_vec.reshape(b, num_tokens, self.d_out)
context_vec = self.out_proj(context_vec)
```
- 헤드별 출력을 합쳐 원래 차원으로 되돌림.
- 최종 출력 shape: (b, num_tokens, d_out)

# [CD] 2. Layer Normalization

## Class 설명 및 초기화 부분
### 1. Class 설명
``` Class 설명
class LayerNorm(nn.Module):
```
- nn.Module을 새로 상속받아 새로운 PyTorch 레이어를 정의함.  
- 입력 텐서의 마지막 차원에 대한 정규화를 수행.

### 2. 초기화 부문
```commandline
    def __init__(self, emb_dim):
        super().__init__()
```
- `emb_dim`은 **입력 벡터의 차원**
- super().__init__()는 부모 클래스(nn.Module)의 초기화 함수를 호출. 
- PyTorch 모델 정의에서 필수

>- emb_dim이 위에서 Multi-Head Attention에서 정의한 dim이랑 같은 수인건가
>- 맞아 같은 거임. 

### 3. eps
``` eps
self.eps = 1e-5
``` 

> eps가 무슨 의미인데?
>- epsilon의 약자 
>- "0으로 나누는 것을 방지하기 위한 아주 작은 값"
>- 0과 가까우면 **나누기 연산이 폭발함**
>- 안정적인 계산을 하기 위해 분모에 더하는 작은 수

### 4. scale
``` scale
self.scale = nn.Parameter(torch.ones(emb_dim))
self.shift = nn.Parameter(torch.zeros(emb_dim))
```
- `scale`은 γ (gamma) - 크기 값 (원래 크기와 비교)
- 초기값은 1, 레이어 정규화 후 수정됨.
- nn.Parameter로 정의하면 학습 시 업데이트 됨

- `shift`는 β (beta) - 위치 값 (원래 위치와 비교)
- 초기값은 0, 레이어 정규화 후 수정됨.
- 위와 동일하게 nn.Parameter로 정의하면 학습 시 업데이트 됨.

> 감마-베타와 스케일-쉬프트에 대해서 좀 더 자세히 이해할 것
#### (1) 평균과 분산에 대해서 먼저 이해해야한다...
**평균(mean)**  
평균(mean)은 "가운데 값(대표 값)"  

예시 데이터:
```평균 예시 데이터
[2, 4, 6] 
```
중간값 = 4

**분산(variance)**  
분산(variance)는 "흩어진 정도(퍼져있는 정도)"

예시 데이터 A:
```분산 예시 데이터 A
[3.9, 4.0, 4.1] > 분산이 작다 
```
예시 데이터 B:
```분산 예시 데이터 A
[2, 4, 6] > 분산이 크다 
```

#### (2) 근뎅 이제 Normalization을 하면 원래의 값을 잃어버린다.
[2, 4, 6] -> [-1, 0, 1]

이러한 이유로 이후, 모델이 필요하면 "정규화를 다시 할 수 있게 만들어야 한다."
이것을 γ(scale)·β(shift)로 설정해둔다. 

**γ** (scale = 크기 늘리고 줄이기)  
**β** (shift = 위치 이동)

## 순전파 foward pass
```foward pass
def forward(self, x):
```
- forward 메서드는 순전파(forward pass)
- x는 입력 텐서입니다. 일반적으로 (batch_size, seq_len, emb_dim)

### 1. 평균 계산
```
mean = x.mean(dim=-1, keepdim=True)
```
- 마지막 차원(emb_dim) 기준으로 평균을 계산
- keepdim=True는 결과 텐서의 차원을 유지해서 나중에 브로드캐스트 연산 // 차원 값을 유지하겠단 의미
- 예: (batch, seq_len, emb_dim) → (batch, seq_len, 1) // 마지막 차원 쓰겠단 의미
 
### 2. 분산 계산
``` 분산 계산
var = x.var(dim=-1, keepdim=True, unbiased=False)
```
- 마지막 차원(emb_dim) 기준으로 평균을 계산
- unbased ->  모집단 분석 선택 /

### 3. 정규화
``` Normalization
norm_x = (x - mean) / torch.sqrt(var + self.eps)
```
- 입력을 정규화(normalize)합니다.
- 평균을 빼고, 분산의 제곱근(표준편차)으로 나눕니다.
- eps를 더해서 0으로 나누는 문제를 방지합니다.
- 이렇게 하면 norm_x의 마지막 차원 값들이 평균 0, 분산 1이 됩니다.

### 4. 적용
``` shift와 scale 적용
return self.scale * norm_x + self.shift
```

# [CD] 3. GELU

GELU는 ReLU부도 Tranformer에 더 적합한 함수

입력 x가 클수록 더 통과시키고, 
x가 작을 수록 확률적으로 0 근처에 보내는 함수

## 전체 코드
```GelU
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

# [CD] 4. FeedFoward(FFN)

여기의 FFN은 독립적으로 적용되는 작은 MLP

## Code Overview
```
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMB_DIM, 4 * EMB_DIM),
            GELU(),
            nn.Linear(4 * EMB_DIM, EMB_DIM),
        )
```

## Expansion Embedding
```
nn.Linear(EMB_DIM → 4 × EMB_DIM),
GELU(),
nn.Linear(4×EMB_DIM → EMB_DIM)
```

- 입력 임베딩 크기를 4배로 확장하는 단계
- GELU() 적용
- 임베딩 크기를 다시 원래 상태로 축소 (Projection Back)

# [CD] 5. FeedForward

```
self.layers = nn.Sequential
```
- 여러 레이어(연산)를 순차적으로 묶는 nn.Sequential을 만들어 self.layers에 저장함. 
- forward에서 이 순서대로 입력이 통과합니다.

``` 확장
nn.Linear(EMB_DIM, 4 * EMB_DIM),
```
- 첫번째 레이어: 선형 변환.
- FFN 내부 차원을 4배로 확장함.

``` 함수
GELU(),
```

``` 축소
nn.Linear(4 * EMB_DIM, EMB_DIM),
```

# [CD] 6. TransformerBlock
개요: 트랜스포머를 진행하는 코드 단
- 근데 이 코드는 일반적으로 알려진 방식과 다른 Pre-LayerNorm 구조
- 학습 안정성이 더 좋음
- 원래 구조: `Attention -> Dropout -> Residual -> LayerNorm`
- Pre-Norm: `Residual -> LayerNrom -> MHA -> Dropout -> Residual Add`

## 이니셜라이징 부분
### init
```angular2html
def __init__(self):
    super().__init__()
    self.att = MultiHeadAttention(
        d_in=EMB_DIM,
        d_out=EMB_DIM)

    self.ff = FeedForward()
    self.norm1 = LayerNorm(EMB_DIM)
    self.norm2 = LayerNorm(EMB_DIM)
    self.drop_shortcut = nn.Dropout(DROP_RATE)
```
- MHA 및 임베딩 차원 정의
- FFN 정의
- 정규화 정의 
- Dropout 정의

### Foward
서브 레이어: 서브 레이어는 트랜스포머 블록을 구성하는 단위를 뜻함.

#### Self Attention 서브 레이어 
``` Self Attention
shortcut = x        # (1) Residual 저장
x = self.norm1(x)   # (2) LayerNorm 먼저 적용 (Pre-LN)
x = self.att(x)     # (3) Multi-Head Attention 수행
x = self.drop_shortcut(x)  # (4) Dropout 적용
x = x + shortcut    # (5) Residual Add
```

#### Feed Foward 서브 레이어
``` Feed Foward
shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = self.drop_shortcut(x)
x = x + shortcut
```

### shortcut / Residual 저장
```
shortcut = x        # (1) Residual 저장
```
- Residual(잔차) / Residual Connection 
  - 기존 정보는 그대로 두고 필요한 변화만 더하는 구조
- 여기서 x의 shape은 `3차원 Tensor`형식으로 존재함.
  - x.shape = (batch_size, seq_len, emb_dim)

### LayerNorm 적용
``` Pre-LN
x = self.norm1(x)   # (2) LayerNorm 먼저 적용 (Pre-LN)
```
- LayerNormalization
  - 각 벡터의 평균과 분산을 맞추어준다.

### MHA 실행
``` MHA 실행
x = self.att(x)     # (3) Multi-Head Attention 수행
```
- MHA 실행한다. 

### Dropout 적용
``` Dropout 적용
x = self.drop_shortcut(x) # (4) Dropout 적용
```
- overfitting을 방지하기 위한 regularization 기법
- 학습 데이터에 랜덤 값을 집어 넣어 다양한 답변을 익히게 하는 방식

| 개념         | Dropout             | Gradient Descent      |
| ---------- | ------------------- | --------------------- |
| 목적         | 과적합 방지              | 손실 최소화(학습)            |
| 작동 방식      | 뉴런 일부를 0으로 꺼버림      | 기울기를 사용해 가중치를 업데이트    |
| 학습에 끼치는 영향 | 조금 불안정하게 만들어 일반화 향상 | 실제로 모델을 학습시키는 메인 알고리즘 |
| 적용 위치      | 네트워크 내부 layer       | 모든 parameter 업데이트     |
| 학습 시만 동작   | YES                 | YES (테스트 단계에서도 계산은 함) |


### Residual Add
``` Residual Add
x = x + shortcut    # (5) Residual Add
```
- 마무리 단계

# [CD] 7. GPT Model
Summary: 

## Class Definition
### (1) Constructor (생성자)
``` Constructor
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
```
- `GPTModel`이라는 PyTorch 모듈을 정의.

### (2) Token_EMB Layer (토큰 임베딩 레이어)
```
self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
```
- 토큰 ID를 `EMB_DIM` 길이의 실수 벡터로 변환

> nn.Embedding은 내장 함수인가?
>- "ㅇㅇ" 내장 함수임, 토큰들을 `EMB_DIM`에 따른 차원의 "의미 벡터"로 바꿔주는 역할
>
> 이게 뭔소리 -> 입력 in_idx 가 (batch_size, seq_len)일 때 tok_embeds는 (batch_size, seq_len, EMB_DIM)이 됩니다. 
>- `in_idx`는 토크나이즈 된 문장. `batch_size`는 문장의 개수. `seq_len`은 문장의 길이.
>
> 그러면 문장이 길어지면 중간에서 잘라야하는데 발생하는 문제가 없는가?
>- 없다
>> 이유.
>>- nn.Embedding은 **토큰 자체**를 임베딩 할 뿐 위치 정보는 알 필요 없음.
>>- **문장의 순서 정보**는 Position Embedding이 처리함.
>>> token_embeds: 내용을 임베딩
>>> pos_embeds: 순서/위치 정보를 제공.
>>> 문맥 관계는: Self-Attention
> 
>- GPT 학습 흐름: Embedding → Transformer → Loss → Backprop  
>- 문장을 잘라서 `seq_len`을 맞추는 것은 Transformer의 ***기본 규칙***

### (3) Positional Embedding (위치 임베딩)
``` Positional Embedding
self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
```
- `CONTEXT_LENGTH(최대 문맥  길이)`의 각 위치에 대해 EMB_DIM 크기 벡터를 학습함.
- `seq_len`이 `CONTEXT_LENGTH`보다 커지면 인덱스 오류가 발생.

> `seq_len`과 `CONTEXT_LENGTH`의 관계성?
>- `CONTEXT_LENGTH`는 모델이 한번에 처리할 수 있는 최대 토큰 수
>- 현재 배치 시퀀스의 길이
>- 즉 더 짧아야함. 길면 IndexError이 발생함.

### (4) Dropout Layer (일반화 과정)
``` Dropout
self.drop_emb = nn.Dropout(DROP_RATE)
```
- 임베딩 직후 적용할 Dropout Layer, 학습 시 일부 차원을 랜덤하게 0으로 만들어 일반화.
- 추론(inference)에서는 자동으로 비활성화.

> 추론 과정은 어디서 정확히 진행되는가? Pretrain에서는 안하지 않나? `Reasoning`
>- 한국어로 추론으로 번역되어 혼동 되었음. 딥러닝 내의 추론은 `inference` 일반적인 뜻의 추론 `reasoning`과 다름
>
> 딥러닝 내에서의 inference(추론)은 `예측하는 단계`임을 뜻함.
>- 모델이 `입력 -> 출력(logits, 확률, 토큰)`을 만들어내는 과정
>- forward pass로 해석해도 됨.

### (5) Sequential Execution TransformerBlock
```
self.trf_blocks = nn.Sequential(
    *[TransformerBlock() for _ in range(NUM_LAYERS)])
```
- `NUM_LAYERS`개의 TransformerBlock을 순서대로 담은 nn.Sequential.
- 각 블록은 (LayerNorm → MHA → Residual → LayerNorm → FFN → Residual) 같은 구조로 구성
- I/O Shape은 블록마다 `batch_size, seq_len, EMB_DIM`으로 유지됨.

> 일단 다 이해가 안감.  
>- 여러 레이어를 연속적으로 실행하게 해주는 코드 : 트랜스포머 블록을 여러번 통과하게 만드는 코드 ***`NUM_LAYERS`만큼***
>- 즉, 그냥 반복 실행시킨거라 생각하면 편함.

### (6) LAYER *NORMALIZATION*
``` 
self.final_norm = LayerNorm(EMB_DIM)
```
- 모델의 마지막에 적용하는 LayerNorm
- 토큰별 `EMB_DIM` 축을 정규화해 출력의 스케일을 안정화

### (7) Linear Transformation
``` 선형변환
self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)
```
- `EMB_DIM` -> `VOCAB_SIZE`로 선형 변환 및 각 토큰 위치에 `vocab`에 따른 `logits` 생성
- `bias=False`의 이유는 출력 중 편향을 생략하거나 embedding과  weight-tying(가중치 공유)를 해야하기 때문이다.

> vocab이 뭔데?, logits도 뭔데?
>- `vocab`: 단어사전: 모델이 인식할 수 있는 모든 토큰 집합
>
> logits(로그잇)
>- 모델이 각 토큰이 다음에 올 확률을 얼마나 "좋아하는지" 점수로 나타낸 것 
>- softmax 직전의 값
>- 가중치임 가중치, 근데 가중합을 1로 하지 않은.
>
> bias=False 와 weight-tying
> 가중치 공유가 필요한 이유는 세가지를 크게 말할 수 있다.
>- (1) Semantic Consistency (의미적 연결 유지)
>> 입력 임베딩(단어를 벡터로 변환하는 함수)와 출력(projection, 벡터를 단어로 되돌리는 함수)의 행렬을 공유하면 언어 모델의 품질이 올라감.
>- (2) 학습 안정성
>- (3) 파라미터 절감
>
> 하지만 `bias=False`를 한다고 `weight-tying`이 되는 것이 아니다.
> `weight-tying`을 위한 필수 조건 중 하나가 `bias=False`인 것이다.

## Forward (실제 데이터가 흐르는 경로)
### (1) 인자 부분
``` in_idx
def forward(self, in_idx):
```
- in_idx = Token ID의 정수 텐서

### (2) 입력 텐서 열기
``` 뭐임 이건
batch_size, seq_len = in_idx.shape
``` 
- 입력 텐서의 두 차원을 만듬
- 이후 연산에서 텐서 크기 추적에 쓰임
- `in_idx.shape == (4, 16)` 이면 `batch_size=4`, `seq_len=16`

> 텐서 크기 추적에 쓰인다는 것이 무슨 의미임?
>- 그냥 재사용하기 쉽게 변수를 저장해두었다는 의미임 ㅋㅋ;

### (3) 토큰 임베딩
``` 토큰 임베딩
tok_embeds = self.tok_emb(in_idx)
```
- 제곧내

### (4) 포지션 임베딩
``` 포지션 임베딩
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
```
- 제곧내

### (5) x값 정의
``` x값 정의
x = tok_embeds + pos_embeds
```
- `임베딩된` 토큰과 위치값 정의
> x 값 정의하는게 잔차 연결에서 이걸 써야하기 때문임. 근데 생각해보면 FFNN이랑 MHA를 둘 다 통과시키는데 두번 갱신되는건가?
>- ㅇㅇ 

### (6) OVERFITTING 과적합 방지
``` 과적합방지
x = self.drop_emb(x)
```
- 제곧내

### (7) 트랜스포머 블록 실행 
``` EXECUTE TRANSFORMER-BLCOK
x = self.trf_blocks(x)
```
- 제곧내

### (8) 최종 정규화
```final_norm
x = self.final_norm(x)
```
- 제곧내

### (9) 로그잇 생성
``` 로그잇
logits = self.out_head(x)
```
- 선형 변환된 정보를 logits(로그잇)으로 만들어서 내보냄.