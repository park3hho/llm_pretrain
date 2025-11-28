# 모델을 정의할 때 사용하는 상수들

VOCAB_SIZE = tokenizer.n_vocab # 50257 Tiktoken (틱토크나이저 vocab)
#VOCAB_SIZE = len(tokenizer) # AutoTokenizer
CONTEXT_LENGTH = 128  # Shortened context length (orig: 1024) (최대 해석 길이)
EMB_DIM = 768  # Embedding dimension (임베딩 차원)
NUM_HEADS = 12  # Number of attention heads (헤드 개수 - 분석하는 두뇌 개수)
NUM_LAYERS = 12  # Number of layers (트랜스포머 블록을 몇번 통과 시킬 것인가)
DROP_RATE = 0.1  # Dropout rate (변수 넣는 비율)
QKV_BIAS = False  # Query-key-value bias (weight tying)

import torch.nn as nn


class MultiHeadAttention(nn.Module): # MHA 정의
    def __init__(self, d_in, d_out):
        super().__init__() # 기본 함수

        assert d_out % NUM_HEADS == 0, "d_out must be divisible by n_heads" # 나머지 없게끔 헤드를 나눔

        self.d_out = d_out # 출력 차원 수
        self.head_dim = d_out // NUM_HEADS # 헤드 차원 개수, (총 출력 차원 % 헤드 개수)

        self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS) # Q 선형 변환
        self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS) # K 선형 변환
        self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS) # V 선형 변환
        self.out_proj = nn.Linear(d_out, d_out) # 차원 합치기 = 최종 선형 투사
        self.dropout = nn.Dropout(DROP_RATE) # 과적합 방지
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1)) # 마스킹
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x) # Q
        values = self.W_value(x)  # V

        keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim) # Key 값 head로 나누기
        values = values.view(b, num_tokens, NUM_HEADS, self.head_dim) # Value 값 HEAD로 나누기
        queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim) # Query 값 HEAD로 나누기

        keys = keys.transpose(1, 2) # 서순 변환
        queries = queries.transpose(1, 2) # 서순 변환
        values = values.transpose(1, 2) # 서순 변환 Mapping 해야함

        attn_scores = queries @ keys.transpose(2, 3) # 행렬곱, Q-K 연산

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # 마스킹
        attn_scores.masked_fill_(mask_bool, -torch.inf) # 얘도

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # 소프트 맥스
        attn_weights = self.dropout(attn_weights) # 과적합 방지

        context_vec = (attn_weights @ values).transpose(1, 2) # Weighted sum QK-V 연산

        context_vec = context_vec.reshape(b, num_tokens, self.d_out) # 원래 차원으로 합치기
        context_vec = self.out_proj(context_vec) # 총 투사

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim): # 임베딩 차원
        super().__init__()
        self.eps = 1e-5 # epsilon
        self.scale = nn.Parameter(torch.ones(emb_dim)) # γ 감마 - 크기
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # β 베타 - 좌표

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # 평균치 구하기
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 모집단 분석
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # 정규화
        return self.scale * norm_x + self.shift # 크기와 좌표 기억한 후 되돌리기.

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): # 그냥 활성함수, 공식은 뭐,,,
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module): # FFNN, 히든레이어층이 입력의 퍼셉트론보다 많아진 다음 다시 원래 개수로 돌아옴
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMB_DIM, 4 * EMB_DIM),
            GELU(),
            nn.Linear(4 * EMB_DIM, EMB_DIM),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module): # Transformer BLOCK
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention( # MHA 실행을 "정의하는" 함수
            d_in=EMB_DIM,
            d_out=EMB_DIM)

        self.ff = FeedForward() # FFNN 실행을 "정의하는" 함수
        self.norm1 = LayerNorm(EMB_DIM) # 첫번째 레이어 정규화를 "정의하는" 함수
        self.norm2 = LayerNorm(EMB_DIM) # 두번째 레이어 정규화를 "정의하는" 함수
        self.drop_shortcut = nn.Dropout(DROP_RATE) # 과적합 방지

    def forward(self, x):
        shortcut = x # RESIDUAL DEFINITION
        x = self.norm1(x) # Pre-LN
        x = self.att(x) # MHA
        x = self.drop_shortcut(x) # PREVENTING OVERFITTING
        x = x + shortcut # RESIDUAL ADD in Korean "잔차 계산"

        shortcut = x # RESIDUAL DEFINITION
        x = self.norm2(x) # Pre-LN
        x = self.ff(x) # FFNN
        x = self.drop_shortcut(x) # PREVENTING OVERFTTING
        x = x + shortcut # RESIDUAL ADD in Korean "잔차 계산"

        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM) # 토큰 임베딩 정의
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM) # 위치 임베딩 정의
        self.drop_emb = nn.Dropout(DROP_RATE) # 예외 등장 비율이라 생각하면 됨. - 과적합 방지

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]) # 트랜스포머 블록을 몇번 실행할 것이냐,

        self.final_norm = LayerNorm(EMB_DIM) # 마지막 정규화 / EMB_DIM을 정규화해 출력의 스케일을 안정화
        self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False) # 선형변환, 로그잇 생성.

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape # in_idx 적용
        tok_embeds = self.tok_emb(in_idx) # 토큰 임베딩
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # 위치 정보 임베딩
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size] / x 값 정의 (감마 베타)
        x = self.drop_emb(x) # 과적합 방지g
        x = self.trf_blocks(x) # 트랜스포머 블록
        x = self.final_norm(x) # 마지막 정규화
        logits = self.out_head(x) # 선형변환된 정보를 내보내기 위한 작업
        return logits