# Attention 的數學推導：為什麼 RNN 需要 Attention

> 適合大學一年級程度，從零開始理解 **Seq2Seq with Attention** 的動機、前向傳播、注意力權重、損失函數，以及反向傳播與時間反向傳播（BPTT）。
>
> 本版本已移除 Bidirectional Encoder 相關內容，僅討論單向 RNN Encoder-Decoder 與 Attention。

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [為什麼 RNN 需要 Attention](#2-為什麼-rnn-需要-attention)
3. [沒有 Attention 的 Seq2Seq RNN](#3-沒有-attention-的-seq2seq-rnn)
4. [加入 Attention 的 Seq2Seq RNN](#4-加入-attention-的-seq2seq-rnn)
5. [Encoder 的前向傳播](#5-encoder-的前向傳播)
6. [Attention 的前向傳播](#6-attention-的前向傳播)
7. [Decoder 與輸出層的前向傳播](#7-decoder-與輸出層的前向傳播)
8. [損失函數](#8-損失函數)
9. [反向傳播總覽](#9-反向傳播總覽)
10. [輸出層與 Decoder 的反向傳播](#10-輸出層與-decoder-的反向傳播)
11. [Attention 的反向傳播推導](#11-attention-的反向傳播推導)
12. [Encoder 的時間反向傳播 BPTT](#12-encoder-的時間反向傳播-bptt)
13. [參數更新](#13-參數更新)
14. [完整流程整理](#14-完整流程整理)

---

## 1. 基本符號定義

假設輸入序列長度為 $T_x$，輸出序列長度為 $T_y$。

輸入序列為

$$
x=(x_1,x_2,\ldots,x_{T_x})
$$

目標輸出序列為

$$
y=(y_1,y_2,\ldots,y_{T_y})
$$

其中 $x_i$ 通常是第 $i$ 個輸入 token 的 embedding 向量，$y_t$ 是第 $t$ 個目標輸出 token。

### 1.1 主要符號表

| 符號 | 意義 |
|---|---|
| $T_x$ | 輸入序列長度 |
| $T_y$ | 輸出序列長度 |
| $x_i$ | 第 $i$ 個輸入 token 的 embedding |
| $y_t$ | 第 $t$ 個目標輸出 token |
| $h_i$ | encoder 在第 $i$ 步的 hidden state |
| $s_t$ | decoder 在第 $t$ 步的 hidden state |
| $e_{t,i}$ | decoder 第 $t$ 步對 encoder 第 $i$ 步的 alignment score |
| $\alpha_{t,i}$ | attention weight |
| $c_t$ | decoder 第 $t$ 步的 context vector |
| $\hat{y}_t$ | decoder 第 $t$ 步輸出的預測機率分佈 |
| $z_t$ | 輸出層 softmax 前的 logits |
| $\mathcal{L}$ | 整體 loss |
| $V$ | 字彙表大小 |

---

## 2. 為什麼 RNN 需要 Attention

傳統 Seq2Seq RNN 的 encoder 會將整個輸入序列壓縮成最後一個 hidden state：

$$
c=h_{T_x}
$$

decoder 之後每一步都只能依賴同一個固定向量 $c$ 來產生輸出：

$$
s_t=f(s_{t-1}, y_{t-1}, c)
$$

這個設計有三個主要問題。

第一，**資訊瓶頸**。不論輸入句子有多長，整個序列都必須被壓縮進單一向量 $h_{T_x}$。當 $T_x$ 很大時，早期 token 的資訊容易被後續狀態覆蓋。

第二，**長距離依賴困難**。RNN 的狀態遞迴為

$$
h_i=\phi(W_xx_i+W_hh_{i-1}+b_h)
$$

若將 $h_{T_x}$ 對較早狀態 $h_i$ 求導，會出現很多 Jacobian 的連乘：

$$
\frac{\partial h_{T_x}}{\partial h_i}
=\prod_{k=i+1}^{T_x}\frac{\partial h_k}{\partial h_{k-1}}
$$

當矩陣連乘的范數小於 1 時，梯度容易消失；大於 1 時，梯度容易爆炸。因此 decoder 若只依賴 $h_{T_x}$，模型很難穩定地學到輸入前段與輸出之間的關係。

第三，**decoder 每一步需要的輸入資訊不同**。例如翻譯時，decoder 產生第 1 個詞時可能需要看輸入第 1 到第 2 個詞；產生第 5 個詞時可能需要看輸入第 4 到第 6 個詞。固定向量 $c$ 無法針對不同輸出時間步動態改變。

Attention 的核心想法是：decoder 在每一個輸出時間步 $t$，都重新對所有 encoder hidden states $h_1,\ldots,h_{T_x}$ 分配權重，形成當下需要的 context vector：

$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

其中 $\alpha_{t,i}$ 表示 decoder 在第 $t$ 步應該關注第 $i$ 個輸入位置的程度。

因此 Attention 不是替代 RNN，而是補強 RNN 的序列壓縮缺陷：它讓 decoder 不必只靠最後一個 hidden state，而能直接使用所有 encoder states。

---

## 3. 沒有 Attention 的 Seq2Seq RNN

### 3.1 Encoder

單向 RNN encoder 定義為

$$
a_i^{enc}=W_xx_i+W_hh_{i-1}+b_h
$$

$$
h_i=\tanh(a_i^{enc})
$$

其中 $h_0$ 通常設為零向量。

最後只取

$$
c=h_{T_x}
$$

作為整句輸入的摘要。

### 3.2 Decoder

decoder 的遞迴可寫成

$$
a_t^{dec}=W_ss_{t-1}+W_yy_{t-1}+W_cc+b_s
$$

$$
s_t=\tanh(a_t^{dec})
$$

輸出層為

$$
z_t=W_os_t+b_o
$$

$$
\hat{y}_t=\text{softmax}(z_t)
$$

問題是每一個 $t$ 都使用同一個 $c$。也就是說，decoder 無法根據目前要生成的 token，選擇性地查看不同輸入位置。

---

## 4. 加入 Attention 的 Seq2Seq RNN

加入 Attention 後，不再只使用固定的 $c=h_{T_x}$。encoder 會保留所有 hidden states：

$$
H=(h_1,h_2,\ldots,h_{T_x})
$$

decoder 在第 $t$ 步會根據上一個 decoder state $s_{t-1}$ 與每個 encoder state $h_i$ 計算分數：

$$
e_{t,i}=\text{score}(s_{t-1},h_i)
$$

再透過 softmax 轉成權重：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x}\exp(e_{t,j})}
$$

最後形成 context vector：

$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

decoder 第 $t$ 步改為使用動態 context vector $c_t$：

$$
s_t=\tanh(W_ss_{t-1}+W_yy_{t-1}+W_cc_t+b_s)
$$

---

## 5. Encoder 的前向傳播

對 $i=1,2,\ldots,T_x$：

$$
a_i^{enc}=W_xx_i+W_hh_{i-1}+b_h
$$

$$
h_i=\tanh(a_i^{enc})
$$

這裡的 $h_i$ 是第 $i$ 個輸入位置的語意表示。Attention 會保留所有 $h_i$，而不是只保留 $h_{T_x}$。

---

## 6. Attention 的前向傳播

Attention 有多種形式。這裡使用常見的 additive attention，也稱 Bahdanau attention。

### 6.1 Alignment score

對 decoder 時間步 $t$ 與 encoder 位置 $i$，先計算

$$
u_{t,i}=W_as_{t-1}+U_ah_i+b_a
$$

$$
r_{t,i}=\tanh(u_{t,i})
$$

$$
e_{t,i}=v_a^Tr_{t,i}
$$

其中：

- $W_a$ 作用在 decoder state 上
- $U_a$ 作用在 encoder state 上
- $v_a$ 將非線性表示轉成純量分數
- $e_{t,i}$ 越大，代表第 $t$ 步越應該關注第 $i$ 個輸入位置

### 6.2 Softmax attention weights

將所有 $e_{t,i}$ 正規化：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x}\exp(e_{t,j})}
$$

因此

$$
\sum_{i=1}^{T_x}\alpha_{t,i}=1
$$

且

$$
0\le \alpha_{t,i}\le 1
$$

### 6.3 Context vector

用 attention weights 對所有 encoder states 做加權平均：

$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

$c_t$ 是 decoder 在第 $t$ 步實際讀取的輸入資訊摘要。

---

## 7. Decoder 與輸出層的前向傳播

### 7.1 Decoder state

令 $y_{t-1}$ 表示前一個輸出 token 的 embedding。在訓練時，常用 teacher forcing，也就是使用正確答案的前一個 token。

$$
a_t^{dec}=W_ss_{t-1}+W_yy_{t-1}+W_cc_t+b_s
$$

$$
s_t=\tanh(a_t^{dec})
$$

### 7.2 Output logits

$$
z_t=W_os_t+b_o
$$

### 7.3 Softmax probability

對字彙表中第 $k$ 個 token：

$$
\hat{y}_{t,k}=\frac{\exp(z_{t,k})}{\sum_{m=1}^{V}\exp(z_{t,m})}
$$

整個 $\hat{y}_t$ 是一個長度為 $V$ 的機率分佈。

---

## 8. 損失函數

若目標 token $y_t$ 用 one-hot 向量表示，cross-entropy loss 為

$$
\mathcal{L}_t=-\sum_{k=1}^{V}y_{t,k}\log \hat{y}_{t,k}
$$

整個輸出序列的 loss 為

$$
\mathcal{L}=\sum_{t=1}^{T_y}\mathcal{L}_t
$$

因為 $y_t$ 是 one-hot，若正確 token 的 index 是 $g_t$，則可簡化為

$$
\mathcal{L}_t=-\log \hat{y}_{t,g_t}
$$

---

## 9. 反向傳播總覽

整體計算圖可概括為

$$
\mathcal{L}\rightarrow \hat{y}_t\rightarrow z_t\rightarrow s_t\rightarrow c_t\rightarrow \alpha_{t,i},h_i\rightarrow e_{t,i}\rightarrow W_a,U_a,v_a
$$

同時，$s_t$ 也會透過 decoder 的時間遞迴傳回 $s_{t-1}$：

$$
s_t \rightarrow s_{t-1}\rightarrow s_{t-2}\rightarrow \cdots
$$

encoder states $h_i$ 的梯度則有兩個來源：

1. 來自 context vector：$c_t=\sum_i\alpha_{t,i}h_i$
2. 來自 alignment score：$e_{t,i}=v_a^T\tanh(W_as_{t-1}+U_ah_i+b_a)$

因此 encoder 的每個 $h_i$ 不只會收到 RNN 時間方向的梯度，也會直接收到所有 decoder step 傳回來的 attention 梯度。這是 Attention 改善長距離依賴的重要原因。

---

## 10. 輸出層與 Decoder 的反向傳播

### 10.1 Softmax + cross-entropy 的梯度

對 logits $z_t$，有標準結果：

$$
\frac{\partial \mathcal{L}_t}{\partial z_t}=\hat{y}_t-y_t
$$

記

$$
\delta_t^z=\hat{y}_t-y_t
$$

則輸出層參數梯度為

$$
\frac{\partial \mathcal{L}}{\partial W_o}=\sum_{t=1}^{T_y}\delta_t^z s_t^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_o}=\sum_{t=1}^{T_y}\delta_t^z
$$

傳回 decoder hidden state 的梯度為

$$
\frac{\partial \mathcal{L}}{\partial s_t}\Big|_{out}=W_o^T\delta_t^z
$$

### 10.2 Decoder tanh 的梯度

因為

$$
s_t=\tanh(a_t^{dec})
$$

所以

$$
\frac{\partial s_t}{\partial a_t^{dec}}=1-s_t\odot s_t
$$

其中 $\odot$ 表示逐元素相乘。

令 $g_t^s=\frac{\partial \mathcal{L}}{\partial s_t}$ 表示從所有後續路徑累積到 $s_t$ 的梯度，則

$$
\delta_t^s
=\frac{\partial \mathcal{L}}{\partial a_t^{dec}}
=g_t^s\odot(1-s_t\odot s_t)
$$

### 10.3 Decoder 參數梯度

由

$$
a_t^{dec}=W_ss_{t-1}+W_yy_{t-1}+W_cc_t+b_s
$$

可得

$$
\frac{\partial \mathcal{L}}{\partial W_s}
=\sum_{t=1}^{T_y}\delta_t^s s_{t-1}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial W_y}
=\sum_{t=1}^{T_y}\delta_t^s y_{t-1}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial W_c}
=\sum_{t=1}^{T_y}\delta_t^s c_t^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_s}
=\sum_{t=1}^{T_y}\delta_t^s
$$

傳回 context vector 的梯度為

$$
g_t^c
=\frac{\partial \mathcal{L}}{\partial c_t}
=W_c^T\delta_t^s
$$

傳回前一個 decoder state 的梯度為

$$
\frac{\partial \mathcal{L}}{\partial s_{t-1}}\Big|_{dec}=W_s^T\delta_t^s
$$

此外，$s_{t-1}$ 也會影響下一步的 attention score $e_{t,i}$，因此總梯度需加上 attention 路徑傳回的部分。

---

## 11. Attention 的反向傳播推導

Attention 的前向傳播為

$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x}\exp(e_{t,j})}
$$

$$
e_{t,i}=v_a^T\tanh(W_as_{t-1}+U_ah_i+b_a)
$$

以下固定某一個 decoder step $t$ 推導。

### 11.1 Context vector 對 attention weight 的梯度

已知

$$
g_t^c=\frac{\partial \mathcal{L}}{\partial c_t}
$$

因為

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

所以對 $\alpha_{t,i}$：

$$
\frac{\partial \mathcal{L}}{\partial \alpha_{t,i}}
=(g_t^c)^Th_i
$$

記

$$
g_{t,i}^{\alpha}=(g_t^c)^Th_i
$$

### 11.2 Context vector 對 encoder state 的直接梯度

同樣由

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

可得對 $h_i$ 的直接梯度：

$$
\frac{\partial \mathcal{L}}{\partial h_i}\Big|_{c_t}
=\alpha_{t,i}g_t^c
$$

這表示只要 $\alpha_{t,i}$ 大，第 $i$ 個 encoder state 就會從第 $t$ 個 decoder step 收到較強梯度。

### 11.3 Softmax 的梯度

softmax 的 Jacobian 為

$$
\frac{\partial \alpha_{t,j}}{\partial e_{t,i}}
=\alpha_{t,j}(\mathbf{1}_{i=j}-\alpha_{t,i})
$$

因此

$$
\frac{\partial \mathcal{L}}{\partial e_{t,i}}
=\sum_{j=1}^{T_x}\frac{\partial \mathcal{L}}{\partial \alpha_{t,j}}
\frac{\partial \alpha_{t,j}}{\partial e_{t,i}}
$$

代入 softmax Jacobian：

$$
\frac{\partial \mathcal{L}}{\partial e_{t,i}}
=\sum_{j=1}^{T_x}g_{t,j}^{\alpha}\alpha_{t,j}(\mathbf{1}_{i=j}-\alpha_{t,i})
$$

整理得

$$
\frac{\partial \mathcal{L}}{\partial e_{t,i}}
=\alpha_{t,i}\left(g_{t,i}^{\alpha}-\sum_{j=1}^{T_x}\alpha_{t,j}g_{t,j}^{\alpha}\right)
$$

記

$$
\delta_{t,i}^e
=\frac{\partial \mathcal{L}}{\partial e_{t,i}}
$$

### 11.4 Alignment score 的梯度

令

$$
u_{t,i}=W_as_{t-1}+U_ah_i+b_a
$$

$$
r_{t,i}=\tanh(u_{t,i})
$$

$$
e_{t,i}=v_a^Tr_{t,i}
$$

先對 $v_a$：

$$
\frac{\partial \mathcal{L}}{\partial v_a}
=\sum_{t=1}^{T_y}\sum_{i=1}^{T_x}\delta_{t,i}^e r_{t,i}
$$

對 $r_{t,i}$：

$$
\frac{\partial \mathcal{L}}{\partial r_{t,i}}
=\delta_{t,i}^e v_a
$$

對 $u_{t,i}$：

$$
\delta_{t,i}^u
=\frac{\partial \mathcal{L}}{\partial u_{t,i}}
=(\delta_{t,i}^e v_a)\odot(1-r_{t,i}\odot r_{t,i})
$$

### 11.5 Attention 參數梯度

由

$$
u_{t,i}=W_as_{t-1}+U_ah_i+b_a
$$

得到

$$
\frac{\partial \mathcal{L}}{\partial W_a}
=\sum_{t=1}^{T_y}\sum_{i=1}^{T_x}\delta_{t,i}^u s_{t-1}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial U_a}
=\sum_{t=1}^{T_y}\sum_{i=1}^{T_x}\delta_{t,i}^u h_i^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_a}
=\sum_{t=1}^{T_y}\sum_{i=1}^{T_x}\delta_{t,i}^u
$$

### 11.6 Attention 傳回 decoder state 的梯度

因為 $u_{t,i}$ 包含 $s_{t-1}$，所以

$$
\frac{\partial \mathcal{L}}{\partial s_{t-1}}\Big|_{attn}
=\sum_{i=1}^{T_x}W_a^T\delta_{t,i}^u
$$

這項梯度要加到 decoder BPTT 中。

### 11.7 Attention 傳回 encoder state 的梯度

$h_i$ 有兩條 attention 梯度路徑。

第一條是 context vector 直接路徑：

$$
\frac{\partial \mathcal{L}}{\partial h_i}\Big|_{c_t}
=\alpha_{t,i}g_t^c
$$

第二條是 score function 路徑：

$$
\frac{\partial \mathcal{L}}{\partial h_i}\Big|_{e_{t,i}}
=U_a^T\delta_{t,i}^u
$$

因此第 $t$ 個 decoder step 對 $h_i$ 的總梯度為

$$
\frac{\partial \mathcal{L}}{\partial h_i}\Big|_{t}
=\alpha_{t,i}g_t^c+U_a^T\delta_{t,i}^u
$$

累加所有 decoder steps：

$$
g_i^h\Big|_{attn}
=\sum_{t=1}^{T_y}\left(\alpha_{t,i}g_t^c+U_a^T\delta_{t,i}^u\right)
$$

這是 Attention 對 encoder hidden state 的直接梯度貢獻。

---

## 12. Encoder 的時間反向傳播 BPTT

encoder 前向傳播為

$$
a_i^{enc}=W_xx_i+W_hh_{i-1}+b_h
$$

$$
h_i=\tanh(a_i^{enc})
$$

Attention 已經對每個 $h_i$ 給出直接梯度：

$$
g_i^h\Big|_{attn}
=\sum_{t=1}^{T_y}\left(\alpha_{t,i}g_t^c+U_a^T\delta_{t,i}^u\right)
$$

但 $h_i$ 還會影響後面的 encoder state $h_{i+1},h_{i+2},\ldots$，因此 encoder 需要做時間反向傳播。

從 $i=T_x$ 到 $1$ 反向計算。令 $\bar{g}_i^h$ 表示累積到 $h_i$ 的總梯度：

$$
\bar{g}_i^h
=g_i^h\Big|_{attn}+W_h^T\delta_{i+1}^{enc}
$$

其中若 $i=T_x$，則沒有 $\delta_{T_x+1}^{enc}$，可視為零。

因為

$$
h_i=\tanh(a_i^{enc})
$$

所以

$$
\delta_i^{enc}
=\frac{\partial \mathcal{L}}{\partial a_i^{enc}}
=\bar{g}_i^h\odot(1-h_i\odot h_i)
$$

接著得到 encoder 參數梯度：

$$
\frac{\partial \mathcal{L}}{\partial W_x}
=\sum_{i=1}^{T_x}\delta_i^{enc}x_i^T
$$

$$
\frac{\partial \mathcal{L}}{\partial W_h}
=\sum_{i=1}^{T_x}\delta_i^{enc}h_{i-1}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_h}
=\sum_{i=1}^{T_x}\delta_i^{enc}
$$

若需要傳回 embedding 或前一層，則

$$
\frac{\partial \mathcal{L}}{\partial x_i}
=W_x^T\delta_i^{enc}
$$

### 12.1 Attention 為什麼改善梯度傳遞

沒有 Attention 時，decoder loss 主要透過最後一個 encoder state $h_{T_x}$ 回傳到早期 $h_i$：

$$
\frac{\partial \mathcal{L}}{\partial h_i}
=\frac{\partial \mathcal{L}}{\partial h_{T_x}}
\frac{\partial h_{T_x}}{\partial h_i}
$$

而

$$
\frac{\partial h_{T_x}}{\partial h_i}
=\prod_{k=i+1}^{T_x}\frac{\partial h_k}{\partial h_{k-1}}
$$

這會導致長距離梯度消失或爆炸。

有 Attention 時，每個 $h_i$ 都能直接連到每個 decoder step 的 loss：

$$
\mathcal{L}_t\rightarrow c_t\rightarrow h_i
$$

因此梯度包含直接項：

$$
\frac{\partial \mathcal{L}}{\partial h_i}\Big|_{c_t}
=\alpha_{t,i}g_t^c
$$

這條路徑不需要經過從 $h_{T_x}$ 到 $h_i$ 的長串 recurrent Jacobian，因此能減少資訊瓶頸，也讓模型更容易學會輸入與輸出之間的對齊關係。

---

## 13. 參數更新

所有參數可合併記為

$$
\Theta=\{W_x,W_h,b_h,W_s,W_y,W_c,b_s,W_o,b_o,W_a,U_a,v_a,b_a\}
$$

使用梯度下降法更新：

$$
\theta \leftarrow \theta-\eta\frac{\partial \mathcal{L}}{\partial \theta}
$$

其中 $\eta$ 是 learning rate。

實務中常使用 Adam、RMSProp 或帶有 gradient clipping 的 SGD。對 RNN 而言，gradient clipping 很常見，因為 recurrent Jacobian 連乘可能造成梯度爆炸。

---

## 14. 完整流程整理

### 14.1 前向傳播

1. 將輸入 tokens 轉成 embeddings $x_1,\ldots,x_{T_x}$。
2. encoder RNN 依序計算所有 hidden states：

$$
h_i=\tanh(W_xx_i+W_hh_{i-1}+b_h)
$$

3. decoder 在每個時間步 $t$ 計算 attention scores：

$$
e_{t,i}=v_a^T\tanh(W_as_{t-1}+U_ah_i+b_a)
$$

4. 使用 softmax 得到 attention weights：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})}
$$

5. 形成 context vector：

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

6. decoder 更新狀態：

$$
s_t=\tanh(W_ss_{t-1}+W_yy_{t-1}+W_cc_t+b_s)
$$

7. 輸出機率分佈：

$$
\hat{y}_t=\text{softmax}(W_os_t+b_o)
$$

8. 計算 cross-entropy loss：

$$
\mathcal{L}=\sum_t-\sum_k y_{t,k}\log\hat{y}_{t,k}
$$

### 14.2 反向傳播

1. 從 softmax + cross-entropy 得到

$$
\delta_t^z=\hat{y}_t-y_t
$$

2. 回傳到 decoder state $s_t$，再經過 tanh 得到 $\delta_t^s$。
3. 從 decoder 傳回 context vector：

$$
g_t^c=W_c^T\delta_t^s
$$

4. 從 $c_t$ 回傳到 $\alpha_{t,i}$ 與 $h_i$。
5. 從 softmax attention 回傳到 $e_{t,i}$：

$$
\delta_{t,i}^e
=\alpha_{t,i}\left(g_{t,i}^{\alpha}-\sum_j\alpha_{t,j}g_{t,j}^{\alpha}\right)
$$

6. 從 alignment score 回傳到 $W_a,U_a,v_a,b_a,s_{t-1},h_i$。
7. 每個 encoder state $h_i$ 累積所有 decoder steps 的 attention 梯度。
8. encoder RNN 再沿時間反向做 BPTT，更新 $W_x,W_h,b_h$。

---

## 核心結論

RNN 需要 Attention，主要不是因為 RNN 不能處理序列，而是因為傳統 Seq2Seq RNN 將整個輸入壓縮成單一最後狀態 $h_{T_x}$，會造成資訊瓶頸與長距離梯度傳遞困難。

Attention 透過

$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

讓 decoder 在每一個輸出時間步都能動態選擇輸入位置。從反向傳播角度看，Attention 讓 loss 能直接傳回每個 encoder hidden state $h_i$，不必完全依賴長距離 recurrent Jacobian 連乘。因此 Attention 同時改善了資訊存取與梯度傳遞，是 RNN Encoder-Decoder 架構中的關鍵改進。
