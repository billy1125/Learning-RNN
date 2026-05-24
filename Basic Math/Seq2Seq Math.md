# Seq2Seq 的數學推導

> 適合大學一年級程度，從零開始理解 **RNN Encoder-Decoder（Seq2Seq）** 的前向傳播、損失函數與時間反向傳播（BPTT）

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [RNN 的基本數學模型](#2-rnn-的基本數學模型)
3. [Seq2Seq 架構總覽](#3-seq2seq-架構總覽)
4. [機率視角：Seq2Seq 在學什麼？](#4-機率視角seq2seq-在學什麼)
5. [Encoder 的前向傳播](#5-encoder-的前向傳播)
6. [Decoder 的前向傳播](#6-decoder-的前向傳播)
7. [輸出層與條件機率分佈](#7-輸出層與條件機率分佈)
8. [損失函數（Cross-Entropy 與最大似然）](#8-損失函數cross-entropy-與最大似然)
9. [反向傳播：BPTT 與 Seq2Seq 梯度推導](#9-反向傳播bptt-與-seq2seq-梯度推導)
10. [推論階段：從 $p(y|x)$ 解出 $\hat{y}$](#10-推論階段從-pyx-解出-haty)
11. [參數更新：梯度下降法](#11-參數更新梯度下降法)
12. [完整流程整理](#12-完整流程整理)

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
| $\theta$ | 模型所有可訓練參數的集合 |
| $p_\theta(y\|x)$ | 由參數 $\theta$ 所定義的條件機率分布 |

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

## 4. 機率視角：Seq2Seq 在學什麼？

前面我們把 Seq2Seq 描述為「Encoder 把輸入壓成一個向量，Decoder 一步一步生成輸出」。
但從**機率論**的角度看，這個模型其實在做一件更明確的事：

> **學一個條件機率分布 $p_\theta(y \mid x)$。**

也就是說，給定輸入序列 $x$，模型要學會「整段輸出序列 $y$ 的機率長什麼樣子」。

這一節是整份推導裡最關鍵的一塊，它把所有後續的 loss、softmax、cross-entropy、teacher forcing、inference 都串了起來。

---

### 4.1 我們真正想要建模的東西

對任何一對 $(x, y)$，我們想知道：

$$p(y \mid x) = p(y_1, y_2, \ldots, y_{T_y} \mid x_1, x_2, \ldots, x_{T_x})$$

但 $y$ 是一整個序列，候選的整段組合有 $V^{T_y}$ 種，**不可能直接列舉並建模**。

所以我們用**機率論的鏈鎖律（chain rule of probability）**把它拆開。

---

### 4.2 機率鏈鎖律：把整段序列拆成逐步預測

對任何聯合機率分布，恆等式：

$$\boxed{p(y_1, y_2, \ldots, y_{T_y} \mid x) = \prod_{t=1}^{T_y} p(y_t \mid y_1, \ldots, y_{t-1}, x)}$$

簡寫成：

$$p(y \mid x) = \prod_{t=1}^{T_y} p(y_t \mid y_{<t}, x)$$

其中 $y_{<t} = (y_1, \ldots, y_{t-1})$ 表示「第 $t$ 步之前已經生成的所有 token」。

這條公式告訴我們：

> **要建模一整段序列的機率，只需要在每一步建模「下一個 token 的條件機率」即可。**

這也是為什麼 Seq2Seq 可以「一步一步生成」，因為它在數學上本來就被拆成這種逐步形式。

---

### 4.3 Encoder 在機率視角下的角色

Encoder 的作用，是把整段輸入 $x$ 壓縮成一個固定向量 $c$：

$$c = h_{T_x}^{\text{enc}} = \text{Encoder}(x)$$

我們假設**所有與 $x$ 有關的資訊都凝聚在 $c$ 裡**，因此：

$$p_\theta(y \mid x) \approx p_\theta(y \mid c)$$

換句話說，encoder 提供了一個**充分摘要（sufficient summary）**。
這也是為什麼最基本的 Seq2Seq 把 encoder 視為「條件機率的條件部分」：

$$x \;\longrightarrow\; c \;\longrightarrow\; \text{條件 } p(y \mid c)$$

> **Encoder 不直接輸出機率，它只負責提供條件。**

---

### 4.4 Decoder 在機率視角下的角色

Decoder 每一步輸出的 softmax 機率分佈，正是條件機率：

$$\boxed{p_\theta(y_t \mid y_{<t}, x) = \text{softmax}(W_{hy}\, h_t^{\text{dec}} + b_y)}$$

也就是說：

- decoder 的 hidden state $h_t^{\text{dec}}$ 同時編碼了：
  - encoder 給的 $c$（透過 $h_0^{\text{dec}} = c$ 傳進來）
  - 已經生成的 token 歷史 $y_{<t}$（透過 recurrent connection 累積）
- softmax 的每個元素，就是「下一個 token 是某個字的機率」

因此「decoder 一步一步算出 softmax」這件事，在機率上就是：

$$\text{Decoder} : (c, y_{<t}) \;\longmapsto\; p_\theta(y_t \mid y_{<t}, x)$$

---

### 4.5 把整段 $p(y|x)$ 寫成模型形式

把 4.2 的 chain rule 和 4.4 的 decoder 條件機率合起來：

$$\boxed{p_\theta(y \mid x) = \prod_{t=1}^{T_y} p_\theta(y_t \mid y_{<t}, x) = \prod_{t=1}^{T_y} \text{softmax}\bigl(W_{hy}\, h_t^{\text{dec}} + b_y\bigr)_{y_t}}$$

也就是：

> **整段序列的機率 = 每一步條件機率的乘積。**

這條公式有兩個重要含意：

1. 訓練時，loss 就是這個機率的「負對數」（見第 8 節）。
2. 推論時，要找一段 $y$ 讓這個機率最大（見第 10 節）。

---

### 4.6 把整個 Seq2Seq 的「機率圖」畫出來

完整的訊息流是：

$$
\underbrace{x}_{\text{輸入}}
\;\xrightarrow{\text{Encoder}}\;
\underbrace{c}_{\text{摘要}}
\;\xrightarrow{\text{Decoder}}\;
\underbrace{p_\theta(y \mid x) = \prod_{t} p_\theta(y_t \mid y_{<t}, x)}_{\text{條件機率分布}}
$$

也可以白話地寫成：

$$\text{Encoder} \;\longrightarrow\; p_\theta(y \mid x) \;\longrightarrow\; \text{Decoder}$$

換句話說：

- **Encoder 提供條件**
- **Decoder 把條件機率算出來**
- **整個 Seq2Seq 等價於在學 $p_\theta(y \mid x)$**

理解這一點後，後面所有公式（softmax、cross entropy、BPTT、beam search）其實都是這個機率模型的延伸。

---

### 4.7 Teacher Forcing 在機率上的含義

訓練時的 teacher forcing（用真正的 $y^*_{t-1}$ 作為 decoder 的輸入）並不是工程上的小技巧，
它對應的是**估計條件機率 $p_\theta(y_t \mid y^*_{<t}, x)$**：

$$\text{Teacher Forcing 在算}:\quad p_\theta(y_t \mid y^*_{<t}, x)$$

這正是 chain rule 中的條件機率本身：在 $t$ 步之前的 token 已知（即正確答案）的條件下，預測第 $t$ 步。

換句話說：

> 訓練時學的，是「在已知前面正確答案的條件下，第 $t$ 步該輸出什麼」。

---

## 5. Encoder 的前向傳播

假設 encoder 使用最基本的 RNN。

### 5.1 Encoder 遞迴公式

對於 $t = 1, 2, \ldots, T_x$：

$$a_t^{\text{enc}} = W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}} + b_h^{\text{enc}}$$

$$h_t^{\text{enc}} = \tanh(a_t^{\text{enc}})$$

其中初始狀態通常設為：

$$h_0^{\text{enc}} = 0$$

### 5.2 最後 hidden state 作為 context

當 encoder 全部讀完之後，得到：

$$c = h_{T_x}^{\text{enc}}$$

這個 $c$ 代表輸入句子的摘要資訊。
從機率視角看（第 4 節），$c$ 是條件機率 $p_\theta(y \mid x)$ 中「條件」這個部分的壓縮表示。

### 5.3 維度檢查

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

## 6. Decoder 的前向傳播

Decoder 會根據前一步輸出與自己的前一 hidden state，逐步生成新的 token。

### 6.1 Decoder 初始條件

最簡單設定：

$$h_0^{\text{dec}} = c$$

並且 decoder 第一個輸入通常是特殊起始符號 `<BOS>`，記為 $y_0$。

### 6.2 Decoder 遞迴公式

對於 $t = 1, 2, \ldots, T_y$：

$$a_t^{\text{dec}} = W_{yh}^{\text{dec}} y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + b_h^{\text{dec}}$$

$$h_t^{\text{dec}} = \tanh(a_t^{\text{dec}})$$

其中 $y_{t-1}^{\text{in}}$ 是 decoder 在第 $t$ 步使用的輸入：

- 訓練時通常用真實答案 $y_{t-1}$（teacher forcing）
- 推論時通常用模型上一時刻預測出的 token

### 6.3 Teacher Forcing

訓練時，我們通常不是餵模型自己的預測，而是餵正確答案：

$$y_{t-1}^{\text{in}} = y_{t-1}^{\text{true}}$$

這樣可以讓訓練更穩定，也比較容易收斂。

如同 4.7 節所述，這在機率上對應的是 $p_\theta(y_t \mid y^*_{<t}, x)$。

---

## 7. 輸出層與條件機率分佈

Decoder 每一步 hidden state 都要轉成對整個字彙表的機率分佈。
這個機率分佈，**就是第 4 節中 $p_\theta(y_t \mid y_{<t}, x)$ 的具體實作**。

### 7.1 線性投影到 logits

$$o_t = W_{hy} h_t^{\text{dec}} + b_y$$

其中：

- $h_t^{\text{dec}} \in \mathbb{R}^{d_h}$
- $o_t \in \mathbb{R}^{V}$
- $W_{hy} \in \mathbb{R}^{V \times d_h}$
- $b_y \in \mathbb{R}^{V}$

### 7.2 Softmax 轉成條件機率

對第 $k$ 個字：

$$\hat{y}_{t,k} = p_\theta(y_t = k \mid y_{<t}, x) = \frac{e^{o_{t,k}}}{\sum_{j=1}^{V} e^{o_{t,j}}}$$

因此整個輸出向量 $\hat{y}_t$ 滿足：

$$\sum_{k=1}^{V} \hat{y}_{t,k} = 1$$

這代表在第 $t$ 步，模型對所有字彙的預測機率分佈，並且**正是 chain rule 拆出來的那個條件機率**。

### 7.3 整段序列的機率（再次強調）

把所有時間步乘起來，就得到第 4.5 節給出的完整模型：

$$p_\theta(y \mid x) = \prod_{t=1}^{T_y} \hat{y}_{t, y_t}$$

其中 $\hat{y}_{t, y_t}$ 是在第 $t$ 步、正確 token $y_t$ 上的 softmax 機率值。

---

## 8. 損失函數（Cross-Entropy 與最大似然）

Seq2Seq 的損失函數，從表面看是「cross-entropy」，從本質看是「**最大似然估計（Maximum Likelihood Estimation, MLE）**」。
這一節說明兩者其實是同一件事。

### 8.1 最大似然原則

在機率視角下，訓練資料是一堆 $(x, y)$ pair。我們希望調整參數 $\theta$，讓模型對「真正出現的 $y$」給出**越大越好**的機率：

$$\theta^* = \arg\max_{\theta} \; p_\theta(y \mid x)$$

由 4.5 節：

$$p_\theta(y \mid x) = \prod_{t=1}^{T_y} p_\theta(y_t \mid y_{<t}, x)$$

連乘很難最佳化，所以取 $\log$：

$$\log p_\theta(y \mid x) = \sum_{t=1}^{T_y} \log p_\theta(y_t \mid y_{<t}, x)$$

最大化 $\log p$ 等價於最小化 $-\log p$：

$$\boxed{\mathcal{L} = -\log p_\theta(y \mid x) = -\sum_{t=1}^{T_y} \log p_\theta(y_t \mid y_{<t}, x)}$$

這就是 Seq2Seq 訓練的目標函數，名稱叫做**負對數似然（negative log-likelihood, NLL）**。

---

### 8.2 為什麼 NLL = Cross-Entropy？

對單一步：

$$-\log p_\theta(y_t \mid y_{<t}, x) = -\log \hat{y}_{t, y_t}$$

由於 $y_t$ 是 one-hot，所以這等於：

$$\mathcal{L}_t = -\sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

這正是**交叉熵（cross-entropy）**的定義。

換言之：

> **Seq2Seq 訓練 = 對 $p_\theta(y|x)$ 做最大似然 = 對每一步做 cross-entropy。**

這三件事在數學上完全等價。

---

### 8.3 整段序列的總損失

整個 decoder 產生 $T_y$ 個 token，因此總損失為：

$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}_t = -\sum_{t=1}^{T_y} \sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

若要取平均，也可寫為：

$$\mathcal{L}_{\text{avg}} = \frac{1}{T_y} \mathcal{L}$$

### 8.4 Softmax + Cross-Entropy 的漂亮結果

這是深度學習中非常重要的結果。

若

$$\hat{y}_t = \text{softmax}(o_t)$$

搭配交叉熵損失，則對 logits 的梯度為：

$$\boxed{\frac{\partial \mathcal{L}_t}{\partial o_t} = \hat{y}_t - y_t}$$

這個公式使得反向傳播大幅簡化。

---

## 9. 反向傳播：BPTT 與 Seq2Seq 梯度推導

Seq2Seq 的反向傳播本質上是：

1. Decoder 端沿時間反向傳播
2. 梯度傳回 context vector $c$
3. 再經由 encoder 沿時間反向傳播

這就是 **Backpropagation Through Time（BPTT）**。

從機率視角看，這整個 BPTT 在做的事情，就是：

$$\nabla_\theta \, [-\log p_\theta(y \mid x)]$$

也就是「讓真實 $y$ 出現的機率變大」的方向。

### 9.1 輸出層梯度

對第 $t$ 步，先定義：

$$\delta_t^{o} = \frac{\partial \mathcal{L}}{\partial o_t}$$

由 softmax + cross-entropy 可得：

$$\boxed{\delta_t^{o} = \hat{y}_t - y_t}$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T_y} \delta_t^{o} (h_t^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_y} = \sum_{t=1}^{T_y} \delta_t^{o}$$

而對 decoder hidden state 的梯度為：

$$\frac{\partial \mathcal{L}}{\partial h_t^{\text{dec}}}\Big|_{\text{from output}} = W_{hy}^T \delta_t^{o}$$

### 9.2 Decoder hidden state 的時間反傳

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

### 9.3 Decoder 參數梯度

對 decoder 的參數：

$$\frac{\partial \mathcal{L}}{\partial W_{yh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (y_{t-1}^{\text{in}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (h_{t-1}^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}}$$

### 9.4 梯度如何傳到 context vector

因為 decoder 初始 hidden state 由 context 給定：

$$h_0^{\text{dec}} = c$$

所以整個 decoder 的損失會對 $c$ 有梯度：

$$\frac{\partial \mathcal{L}}{\partial c} = \frac{\partial \mathcal{L}}{\partial h_0^{\text{dec}}}$$

而這個梯度正是從 decoder 第一個時間步反傳回來：

$$\frac{\partial \mathcal{L}}{\partial c} = (W_{hh}^{\text{dec}})^T \delta_1^{\text{dec}}$$

更精確地說，若把 $h_0^{\text{dec}}$ 視為一個節點，則所有透過 decoder 時間展開傳回的梯度都會累積到這裡。

### 9.5 Encoder 的時間反傳

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

### 9.6 Encoder 參數梯度

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} (h_{t-1}^{\text{enc}})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}}$$

---

## 10. 推論階段：從 $p(y|x)$ 解出 $\hat{y}$

訓練完成後，模型參數 $\theta$ 已經學好了 $p_\theta(y|x)$。
但接下來該怎麼**用**這個機率分布來生成輸出？這就是推論（inference / decoding）要做的事。

從機率角度，推論可以寫成一個明確的最佳化問題：

$$\boxed{\hat{y} = \arg\max_{y} \; p_\theta(y \mid x) = \arg\max_{y} \prod_{t=1}^{T_y} p_\theta(y_t \mid y_{<t}, x)}$$

也就是「在所有可能的整段序列裡，找出最有可能的那一段」。

但這個 $\arg\max$ 的搜尋空間是 $V^{T_y}$，**指數級成長**，幾乎不可能窮舉。
因此實作上會用近似方法。

---

### 10.1 Greedy Decoding：每一步取最大

最簡單的方法是「每一步都選當下機率最大的 token」：

$$\hat{y}_t = \arg\max_{k \in \{1,\ldots,V\}} p_\theta(y_t = k \mid \hat{y}_{<t}, x)$$

也就是：

- 第 1 步取機率最大的 token，輸出
- 把這個 token 餵回 decoder
- 第 2 步再取機率最大的 token
- 一直到輸出 `<eos>` 為止

優點：簡單、快。
缺點：**短視**——當下最好不一定全段最好。

---

### 10.2 Beam Search：保留多條路徑

Beam search 是個折衷方法。
每一步保留前 $k$ 個機率最高的候選序列（$k$ 稱為 beam size），下一步再從這 $k$ 條路徑各自展開：

$$\text{每一步}:\quad \text{Top-}k\!\!\sum_{t'=1}^{t} \log p_\theta(y_{t'} \mid y_{<t'}, x)$$

注意這裡用 $\log$ 加法替代乘法，避免數值下溢。

- $k=1$ 退化為 greedy decoding
- $k$ 越大，搜尋越充分，但也越慢
- 實務上常用 $k = 4 \sim 10$

---

### 10.3 機率視角的整體圖像

把訓練與推論合在一起：

| 階段 | 目標 | 數學形式 |
|------|------|----------|
| 訓練 | 學 $p_\theta(y \mid x)$ | $\theta^* = \arg\max_\theta \log p_\theta(y \mid x)$ |
| 推論 | 從 $p_\theta(y \mid x)$ 取出最可能的 $y$ | $\hat{y} = \arg\max_y p_\theta(y \mid x)$ |

> **訓練是讓機率對；推論是把機率轉成輸出。**

---

## 11. 參數更新：梯度下降法

當所有梯度都算出來後，就可以更新參數。

### 11.1 梯度下降更新規則

對任一參數 $\theta$：

$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}$$

其中 $\alpha$ 是學習率。

例如：

$$W_{hy} \leftarrow W_{hy} - \alpha \frac{\partial \mathcal{L}}{\partial W_{hy}}$$

$$W_{xh}^{\text{enc}} \leftarrow W_{xh}^{\text{enc}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}}$$

$$W_{hh}^{\text{dec}} \leftarrow W_{hh}^{\text{dec}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}}$$

### 11.2 為什麼 RNN 容易梯度消失

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

## 12. 完整流程整理

最後，把整個最基本的 Seq2Seq 數學流程整理如下。

### 12.1 機率模型

$$\boxed{p_\theta(y \mid x) = \prod_{t=1}^{T_y} p_\theta(y_t \mid y_{<t}, x)}$$

- Encoder：把 $x$ 壓成 $c$，提供條件
- Decoder：每一步輸出 $p_\theta(y_t \mid y_{<t}, x)$

### 12.2 Forward

#### Encoder：

$$h_t^{\text{enc}} = \tanh(W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}} + b_h^{\text{enc}}), \quad t=1,\dots,T_x$$

$$c = h_{T_x}^{\text{enc}}$$

#### Decoder：

$$h_0^{\text{dec}} = c$$

$$h_t^{\text{dec}} = \tanh(W_{yh}^{\text{dec}} y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + b_h^{\text{dec}}), \quad t=1,\dots,T_y$$

$$o_t = W_{hy} h_t^{\text{dec}} + b_y$$

$$\hat{y}_t = \text{softmax}(o_t) = p_\theta(y_t \mid y_{<t}, x)$$

### 12.3 Loss

$$\mathcal{L} = -\log p_\theta(y \mid x) = -\sum_{t=1}^{T_y} \sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

### 12.4 Backward

#### 輸出層：

$$\delta_t^o = \hat{y}_t - y_t$$

#### Decoder BPTT：

$$\delta_t^{\text{dec}} = \left( W_{hy}^T \delta_t^o + (W_{hh}^{\text{dec}})^T \delta_{t+1}^{\text{dec}} \right) \odot (1 - h_t^{\text{dec}} \odot h_t^{\text{dec}})$$

#### Encoder BPTT：

$$\delta_t^{\text{enc}} = \left((W_{hh}^{\text{enc}})^T \delta_{t+1}^{\text{enc}}\right) \odot (1 - h_t^{\text{enc}} \odot h_t^{\text{enc}})$$

### 12.5 參數梯度

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T_y} \delta_t^o (h_t^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{yh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (y_{t-1}^{\text{in}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{dec}}} = \sum_{t=1}^{T_y} \delta_t^{\text{dec}} (h_{t-1}^{\text{dec}})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\text{enc}}} = \sum_{t=1}^{T_x} \delta_t^{\text{enc}} (h_{t-1}^{\text{enc}})^T$$

### 12.6 推論

$$\hat{y} = \arg\max_y p_\theta(y \mid x) \quad\text{（用 greedy 或 beam search 近似）}$$

---

## 總結：Seq2Seq 的七個核心公式

$$\boxed{
\begin{aligned}
&\textbf{(0) 機率分解：} && p_\theta(y \mid x) = \prod_{t=1}^{T_y} p_\theta(y_t \mid y_{<t}, x) \\
&\textbf{(1) Encoder 更新：} && h_t^{\text{enc}} = \tanh(W_{xh}^{\text{enc}}x_t + W_{hh}^{\text{enc}}h_{t-1}^{\text{enc}} + b_h^{\text{enc}}) \\
&\textbf{(2) Context 向量：} && c = h_{T_x}^{\text{enc}} \\
&\textbf{(3) Decoder 更新：} && h_t^{\text{dec}} = \tanh(W_{yh}^{\text{dec}}y_{t-1}^{\text{in}} + W_{hh}^{\text{dec}}h_{t-1}^{\text{dec}} + b_h^{\text{dec}}) \\
&\textbf{(4) 條件機率：} && p_\theta(y_t \mid y_{<t}, x) = \text{softmax}(W_{hy}h_t^{\text{dec}} + b_y) \\
&\textbf{(5) 訓練目標：} && \mathcal{L} = -\log p_\theta(y \mid x) = -\sum_{t=1}^{T_y}\sum_{k=1}^{V} y_{t,k}\log\hat{y}_{t,k} \\
&\textbf{(6) 推論：} && \hat{y} = \arg\max_y p_\theta(y \mid x)
\end{aligned}
}$$

理解了 **(0) 機率分解** 之後，整個 Seq2Seq 從訓練到推論的數學形式就完全串起來了：

$$\text{Encoder} \;\longrightarrow\; p_\theta(y\mid x) \;\longrightarrow\; \text{Decoder}$$

這是後來 attention、bidirectional encoder、LSTM、GRU、Transformer，乃至於今日大型語言模型，**全部共用的機率骨架**。

---

*參考概念：微積分鏈鎖律 · 線性代數矩陣運算 · 機率鏈鎖律（chain rule of probability）· RNN 時間展開 · Cross-Entropy · 最大似然估計（MLE）· Backpropagation Through Time (BPTT) · Beam Search*
