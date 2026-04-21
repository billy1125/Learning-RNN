# Seq2Seq 的數學推導

> 適合大學一年級程度，從零開始理解 **RNN Encoder-Decoder（Seq2Seq）** 的前向傳播、損失函數與時間反向傳播（BPTT）

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [RNN 的基本數學模型](#2-rnn-的基本數學模型)
3. [Seq2Seq 架構總覽](#3-seq2seq-架構總覽)
4. [Encoder 的前向傳播](#4-encoder-的前向傳播)
5. [Decoder 的前向傳播](#5-decoder-的前向傳播)
6. [輸出層與機率分佈](#6-輸出層與機率分佈)
7. [損失函數（Cross-Entropy）](#7-損失函數cross-entropy)
8. [反向傳播：BPTT 與 Seq2Seq 梯度推導](#8-反向傳播bptt-與-seq2seq-梯度推導)
9. [參數更新：梯度下降法](#9-參數更新梯度下降法)
10. [完整流程整理](#10-完整流程整理)

---

## 1. 基本符號定義

在開始推導之前，先統一符號。

### 1.1 序列資料的表示方式

假設：

- 輸入序列長度為 $T_x$
- 輸出序列長度為 $T_y$
- 輸入序列為

$$x = (x_1, x_2, \ldots, x_{T_x})$$

- 目標輸出序列為

$$y = (y_1, y_2, \ldots, y_{T_y})$$

在自然語言處理中：

- $x_t$ 可以代表第 $t$ 個輸入單字
- $y_t$ 可以代表第 $t$ 個輸出單字

通常每個 Token 會先轉成 one-hot 向量，再乘 embedding 矩陣變成稠密向量。

> Token 是大語言模型處理文字的最小單位。

大語言模型處理文字不是像人類一樣直接閱讀「單字」，而是將文字切碎成 Token。一個 Token 可能是一個單字、一個字母，甚至是像「ing」這樣的字根。

通常 1,000 個 Token 大約等於 750 個中文字，由於 AI 服務都是要截取文字內容作為輸入，因此通常按 Token 數量計費。

### 1.2 主要符號表

| 符號 | 意義 |
|------|------|
| $T_x$ | 輸入序列長度 |
| $T_y$ | 輸出序列長度 |
| $x_t$ | 第 $t$ 個輸入 token 的向量表示 |
| $y_t$ | 第 $t$ 個目標輸出 token |
| $\hat{y}_t$ | 第 $t$ 個時間步的預測分佈 |
| $h_t^{\text{enc}}$ | Encoder 在第 $t$ 步的隱藏狀態 |
| $h_t^{\text{dec}}$ | Decoder 在第 $t$ 步的隱藏狀態 |
| $c$ | context vector，通常取 encoder 最後隱藏狀態 |
| $W_{xh}$ | 輸入到隱藏層的權重矩陣 |
| $W_{hh}$ | 隱藏狀態到下一步隱藏狀態的權重矩陣 |
| $b_h$ | 隱藏層偏差 |
| $W_{hy}$ | 隱藏層到輸出層的權重矩陣 |
| $b_y$ | 輸出層偏差 |
| $\mathcal{L}$ | 整體損失函數 |
| $V$ | 字彙表大小 |

---

## 2. RNN 的基本數學模型

Seq2Seq 的核心是 **Recurrent Neural Network（RNN）**。在 RNN 中，同一組參數會沿著時間重複使用。

### 2.1 單一步驟的 RNN 更新

給定目前輸入 $x_t$ 與前一時刻隱藏狀態 $h_{t-1}$，新的隱藏狀態為：

$$h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

其中：

- $x_t \in \mathbb{R}^{d_x}$
- $h_t \in \mathbb{R}^{d_h}$
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$
- $b_h \in \mathbb{R}^{d_h}$
- $\phi$ 通常可取 $\tanh$ 或其他 activation

### 2.2 為什麼 RNN 能處理序列

因為 $h_t$ 同時依賴：

1. 目前輸入 $x_t$
2. 前一時刻的記憶 $h_{t-1}$

所以 $h_t$ 可以看成「到第 $t$ 步為止，整段序列資訊的壓縮表示」。

### 2.3 常見 activation：$\tanh$

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

其導數為：

$$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$$

因此若

$$h_t = \tanh(a_t)$$

則

$$\frac{\partial h_t}{\partial a_t} = 1 - h_t \odot h_t$$

這在 BPTT 中會反覆用到。

---

## 3. Seq2Seq 架構總覽

Seq2Seq（Sequence-to-Sequence）通常由兩部分組成：

1. **Encoder**：讀入輸入序列 $x_1, x_2, \ldots, x_{T_x}$
2. **Decoder**：根據 encoder 的摘要資訊，逐步產生輸出序列

### 3.1 Encoder-Decoder 的概念

Encoder 將整段輸入序列壓縮成一個 context vector：

$$c = h_{T_x}^{\text{enc}}$$

Decoder 以這個 $c$ 作為初始條件，逐步產生輸出：

$$\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_{T_y}$$

### 3.2 最基本的 Seq2Seq（無 Attention）

最簡化版本中：

- Encoder 最後一個 hidden state 當作 context vector
- Decoder 的初始 hidden state 由這個 context 決定

例如：

$$h_0^{\text{dec}} = c = h_{T_x}^{\text{enc}}$$

---

## 4. Encoder 的前向傳播

假設 encoder 使用最基本的 RNN。

### 4.1 Encoder 遞迴公式

對於 $t = 1, 2, \ldots, T_x$：

$$a_t^{\text{enc}} = W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}} + b_h^{\text{enc}}$$

$$h_t^{\text{enc}} = \tanh(a_t^{\text{enc}})$$

其中初始狀態通常設為：

$$h_0^{\text{enc}} = 0$$

### 4.2 最後 hidden state 作為 context

當 encoder 全部讀完之後，得到：

$$c = h_{T_x}^{\text{enc}}$$

這個 $c$ 代表輸入句子的摘要資訊。

### 4.3 維度檢查

若：

- $x_t \in \mathbb{R}^{d_x}$
- $h_t^{\text{enc}} \in \mathbb{R}^{d_h}$

則：

$$W_{xh}^{\text{enc}}: d_h \times d_x$$

$$W_{hh}^{\text{enc}}: d_h \times d_h$$

因此：

$$W_{xh}^{\text{enc}}x_t + W_{hh}^{\text{enc}}h_{t-1}^{\text{enc}} + b_h^{\text{enc}} \in \mathbb{R}^{d_h}$$

維度正確。✓

---

## 5. Decoder 的前向傳播

Decoder 會根據前一步輸出與自己的前一 hidden state，逐步生成新的 token。

### 5.1 Decoder 初始條件

最簡單設定：

$$h_0^{\text{dec}} = c$$

並且 decoder 第一個輸入通常是特殊起始符號 `<BOS>`，記為 $y_0$。

### 5.2 Decoder 遞迴公式

對於 $t = 1, 2, \ldots, T_y$：

$$a_t^{\text{dec}} = W_{yh}^{\text{dec}} y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + b_h^{\text{dec}}$$

$$h_t^{\text{dec}} = \tanh(a_t^{\text{dec}})$$

其中 $y_{t-1}^{\text{in}}$ 是 decoder 在第 $t$ 步使用的輸入：

- 訓練時通常用真實答案 $y_{t-1}$（teacher forcing）
- 推論時通常用模型上一時刻預測出的 token

### 5.3 Teacher Forcing

訓練時，我們通常不是餵模型自己的預測，而是餵正確答案：

$$y_{t-1}^{\text{in}} = y_{t-1}^{\text{true}}$$

這樣可以讓訓練更穩定，也比較容易收斂。

---

## 6. 輸出層與機率分佈

Decoder 每一步 hidden state 都要轉成對整個字彙表的機率分佈。

### 6.1 線性投影到 logits

$$o_t = W_{hy} h_t^{\text{dec}} + b_y$$

其中：

- $h_t^{\text{dec}} \in \mathbb{R}^{d_h}$
- $o_t \in \mathbb{R}^{V}$
- $W_{hy} \in \mathbb{R}^{V \times d_h}$
- $b_y \in \mathbb{R}^{V}$

### 6.2 Softmax 轉成機率

對第 $k$ 個字：

$$\hat{y}_{t,k} = \frac{e^{o_{t,k}}}{\sum_{j=1}^{V} e^{o_{t,j}}}$$

因此整個輸出向量 $\hat{y}_t$ 滿足：

$$\sum_{k=1}^{V} \hat{y}_{t,k} = 1$$

這代表在第 $t$ 步，模型對所有字彙的預測機率分佈。

---

## 7. 損失函數（Cross-Entropy）

Seq2Seq 最常見的是分類問題：在每一個時間步，從字彙表中選出正確 token。

### 7.1 單一步驟的交叉熵

若真實標籤 $y_t$ 是 one-hot 向量，則第 $t$ 步的 loss 為：

$$\mathcal{L}_t = -\sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

因為 $y_t$ 是 one-hot，只有正確類別那一項為 1，所以也可以寫成：

$$\mathcal{L}_t = -\log \hat{y}_{t,\text{target}}$$

### 7.2 整段序列的總損失

整個 decoder 產生 $T_y$ 個 token，因此總損失為：

$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}_t = -\sum_{t=1}^{T_y} \sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

若要取平均，也可寫為：

$$\mathcal{L}_{\text{avg}} = \frac{1}{T_y} \mathcal{L}$$

### 7.3 Softmax + Cross-Entropy 的漂亮結果

這是深度學習中非常重要的結果。

若

$$\hat{y}_t = \text{softmax}(o_t)$$

搭配交叉熵損失，則對 logits 的梯度為：

$$\boxed{\frac{\partial \mathcal{L}_t}{\partial o_t} = \hat{y}_t - y_t}$$

這個公式使得反向傳播大幅簡化。

---

## 8. 反向傳播：BPTT 與 Seq2Seq 梯度推導

Seq2Seq 的反向傳播本質上是：

1. Decoder 端沿時間反向傳播
2. 梯度傳回 context vector $c$
3. 再經由 encoder 沿時間反向傳播

這就是 **Backpropagation Through Time（BPTT）**。

### 8.1 輸出層梯度

對第 $t$ 步，先定義：

$$\delta_t^{o} = \frac{\partial \mathcal{L}}{\partial o_t}$$

由 softmax + cross-entropy 可得：

$$\boxed{\delta_t^{o} = \hat{y}_t - y_t}$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T_y} \delta_t^{o} (h_t^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_y} = \sum_{t=1}^{T_y} \delta_t^{o}$$

而對 decoder hidden state 的梯度為：

$$\frac{\partial \mathcal{L}}{\partial h_t^{\text{dec}}}\Big|_{\text{from output}} = W_{hy}^T \delta_t^{o}$$

### 8.2 Decoder hidden state 的時間反傳

由於 $h_t^{\text{dec}}$ 不只影響第 $t$ 步輸出，也會透過遞迴影響後面的 hidden state，因此總梯度必須把兩部分相加。

定義 decoder 預激活：

$$a_t^{\text{dec}} = W_{yh}^{\text{dec}} y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + b_h^{\text{dec}}$$

$$h_t^{\text{dec}} = \tanh(a_t^{\text{dec}})$$

令

$$\delta_t^{\text{dec}} = \frac{\partial \mathcal{L}}{\partial a_t^{\text{dec}}}$$

則由鏈鎖律：

$$\frac{\partial \mathcal{L}}{\partial h_t^{\text{dec}}} = W_{hy}^T \delta_t^o + (W_{hh}^{\text{dec}})^T \delta_{t+1}^{\text{dec}}$$

注意第二項來自「未來時間步」的反傳。

再乘上 $\tanh$ 的導數：

$$\boxed{\delta_t^{\text{dec}} = \left( W_{hy}^T \delta_t^o + (W_{hh}^{\text{dec}})^T \delta_{t+1}^{\text{dec}} \right) \odot (1 - h_t^{\text{dec}} \odot h_t^{\text{dec}})}$$

其中邊界條件通常取：

$$\delta_{T_y+1}^{\text{dec}} = 0$$

### 8.3 Decoder 參數梯度

對 decoder 的參數：

$$\frac{\partial \mathcal{L}}{\partial W_{yh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (y_{t-1}^{\text{in}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (h_{t-1}^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}}$$

### 8.4 梯度如何傳到 context vector

因為 decoder 初始 hidden state 由 context 給定：

$$h_0^{\text{dec}} = c$$

所以整個 decoder 的損失會對 $c$ 有梯度：

$$\frac{\partial \mathcal{L}}{\partial c} = \frac{\partial \mathcal{L}}{\partial h_0^{\text{dec}}}$$

而這個梯度正是從 decoder 第一個時間步反傳回來：

$$\frac{\partial \mathcal{L}}{\partial c} = (W_{hh}^{\text{dec}})^T \delta_1^{\text{dec}}$$

更精確地說，若把 $h_0^{\text{dec}}$ 視為一個節點，則所有透過 decoder 時間展開傳回的梯度都會累積到這裡。

### 8.5 Encoder 的時間反傳

因為

$$c = h_{T_x}^{\text{enc}}$$

所以：

$$\frac{\partial \mathcal{L}}{\partial h_{T_x}^{\text{enc}}} = \frac{\partial \mathcal{L}}{\partial c}$$

接著 encoder 像一般 RNN 一樣沿時間反向傳播。

定義：

$$a_t^{\text{enc}} = W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}} + b_h^{\text{enc}}$$

$$h_t^{\text{enc}} = \tanh(a_t^{\text{enc}})$$

令

$$\delta_t^{\text{enc}} = \frac{\partial \mathcal{L}}{\partial a_t^{\text{enc}}}$$

則對最後一步：

$$\delta_{T_x}^{\text{enc}} = \frac{\partial \mathcal{L}}{\partial h_{T_x}^{\text{enc}}} \odot (1 - h_{T_x}^{\text{enc}} \odot h_{T_x}^{\text{enc}})$$

而對一般時間步 $t = T_x-1, \ldots, 1$：

$$\boxed{\delta_t^{\text{enc}} = \left((W_{hh}^{\text{enc}})^T \delta_{t+1}^{\text{enc}}\right) \odot (1 - h_t^{\text{enc}} \odot h_t^{\text{enc}})}$$

### 8.6 Encoder 參數梯度

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} (h_{t-1}^{\text{enc}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}}$$

---

## 9. 參數更新：梯度下降法

當所有梯度都算出來後，就可以更新參數。

### 9.1 梯度下降更新規則

對任一參數 $\theta$：

$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}$$

其中 $\alpha$ 是學習率。

例如：

$$W_{hy} \leftarrow W_{hy} - \alpha \frac{\partial \mathcal{L}}{\partial W_{hy}}$$

$$W_{xh}^{\text{enc}} \leftarrow W_{xh}^{\text{enc}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}}$$

$$W_{hh}^{\text{dec}} \leftarrow W_{hh}^{\text{dec}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}}$$

### 9.2 為什麼 RNN 容易梯度消失

觀察 encoder 與 decoder 的時間反傳公式，都 repeatedly 乘上：

$$W_{hh}^T$$

以及 activation 的導數，例如：

$$1 - h_t \odot h_t$$

若這些值的範數小於 1，長時間連乘後就可能變得非常小：

$$\left\|(W_{hh})^T (W_{hh})^T \cdots (W_{hh})^T\right\| \to 0$$

這就是 **vanishing gradient**。

反之若範數過大，也可能產生 **exploding gradient**。

這也是為什麼後來常用 LSTM / GRU 改善 Seq2Seq。

---

## 10. 完整流程整理

最後，把整個最基本的 Seq2Seq 數學流程整理如下。

### 10.1 Forward

#### Encoder：

$$h_t^{\text{enc}} = \tanh(W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}} + b_h^{\text{enc}}), \quad t=1,\dots,T_x$$

$$c = h_{T_x}^{\text{enc}}$$

#### Decoder：

$$h_0^{\text{dec}} = c$$

$$h_t^{\text{dec}} = \tanh(W_{yh}^{\text{dec}} y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + b_h^{\text{dec}}), \quad t=1,\dots,T_y$$

$$o_t = W_{hy} h_t^{\text{dec}} + b_y$$

$$\hat{y}_t = \text{softmax}(o_t)$$

### 10.2 Loss

$$\mathcal{L} = -\sum_{t=1}^{T_y} \sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

### 10.3 Backward

#### 輸出層：

$$\delta_t^o = \hat{y}_t - y_t$$

#### Decoder BPTT：

$$\delta_t^{\text{dec}} = \left( W_{hy}^T \delta_t^o + (W_{hh}^{\text{dec}})^T \delta_{t+1}^{\text{dec}} \right) \odot (1 - h_t^{\text{dec}} \odot h_t^{\text{dec}})$$

#### Encoder BPTT：

$$\delta_t^{\text{enc}} = \left((W_{hh}^{\text{enc}})^T \delta_{t+1}^{\text{enc}}\right) \odot (1 - h_t^{\text{enc}} \odot h_t^{\text{enc}})$$

### 10.4 參數梯度

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T_y} \delta_t^o (h_t^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{yh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (y_{t-1}^{\text{in}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (h_{t-1}^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} (h_{t-1}^{\text{enc}})^T$$

---

## 總結：Seq2Seq 的六個核心公式

$$\boxed{
\begin{aligned}
&\textbf{(1) Encoder 更新：} && h_t^{\text{enc}} = \tanh(W_{xh}^{\text{enc}}x_t + W_{hh}^{\text{enc}}h_{t-1}^{\text{enc}} + b_h^{\text{enc}}) \\
&\textbf{(2) Context 向量：} && c = h_{T_x}^{\text{enc}} \\
&\textbf{(3) Decoder 更新：} && h_t^{\text{dec}} = \tanh(W_{yh}^{\text{dec}}y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}}h_{t-1}^{\text{dec}} + b_h^{\text{dec}}) \\
&\textbf{(4) 輸出機率：} && \hat{y}_t = \text{softmax}(W_{hy}h_t^{\text{dec}} + b_y) \\
&\textbf{(5) 序列損失：} && \mathcal{L} = -\sum_{t=1}^{T_y}\sum_{k=1}^{V} y_{t,k}\log\hat{y}_{t,k} \\
&\textbf{(6) BPTT 核心：} && \delta_t = \text{未來梯度 + 當前輸出梯度，再乘 activation 導數}
\end{aligned}
}$$

這些公式構成了最基本 **RNN-based Seq2Seq** 的理論基礎。理解它們之後，再往上學 attention、bidirectional encoder、LSTM、GRU 或 Transformer，就會清楚很多。

---

*參考概念：微積分鏈鎖律 · 線性代數矩陣運算 · RNN 時間展開 · Cross-Entropy · Backpropagation Through Time (BPTT)*
