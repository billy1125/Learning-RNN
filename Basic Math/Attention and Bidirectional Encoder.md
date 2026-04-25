# Attention 與 Bidirectional Encoder 的數學推導

> 適合大學一年級程度，從零開始理解 **Seq2Seq with Attention + Bidirectional Encoder** 的前向傳播、注意力權重、損失函數與時間反向傳播（BPTT）

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [為什麼需要 Attention 與雙向 Encoder](#2-為什麼需要-attention-與雙向-encoder)
3. [Bidirectional Encoder 的基本數學模型](#3-bidirectional-encoder-的基本數學模型)
4. [Encoder 的前向傳播](#4-encoder-的前向傳播)
5. [Attention 機制的前向傳播](#5-attention-機制的前向傳播)
6. [Decoder 的前向傳播](#6-decoder-的前向傳播)
7. [輸出層與機率分佈](#7-輸出層與機率分佈)
8. [損失函數（Cross-Entropy）](#8-損失函數cross-entropy)
9. [反向傳播：Attention 與 Bi-Encoder 的梯度推導](#9-反向傳播attention-與-bi-encoder-的梯度推導)
10. [參數更新：梯度下降法](#10-參數更新梯度下降法)
11. [完整流程整理](#11-完整流程整理)

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
| $\overrightarrow{h}_t$ | forward encoder 在第 $t$ 步的隱藏狀態 |
| $\overleftarrow{h}_t$ | backward encoder 在第 $t$ 步的隱藏狀態 |
| $h_t^{\text{enc}}$ | encoder 在第 $t$ 步的雙向表示 |
| $s_t$ | decoder 在第 $t$ 步的隱藏狀態 |
| $e_{t,i}$ | decoder 第 $t$ 步對 encoder 第 $i$ 步的對齊分數 |
| $\alpha_{t,i}$ | attention 權重 |
| $c_t$ | 第 $t$ 步的 context vector |
| $W_a, U_a, v_a$ | attention 的參數 |
| $W_{xh}, W_{hh}$ | RNN 內部權重矩陣 |
| $W_o$ | decoder 輸出層權重 |
| $b_o$ | 輸出層偏差 |
| $\mathcal{L}$ | 整體損失函數 |
| $V$ | 字彙表大小 |

---

## 2. 為什麼需要 Attention 與雙向 Encoder

最早的 Seq2Seq 會把整句輸入壓縮成單一 context vector：

$$c = h_{T_x}^{\text{enc}}$$

<img src="../image/encoder-decoder.jpeg" alt="Encoder Decoder" style="width: 70%;"/>

這樣做的問題是：

1. **輸入句子太長時，最後一個 hidden state 不容易保留所有資訊**
2. **decoder 在每一步都只能看到同一個固定向量**
3. **encoder 若只有單向，位置 $t$ 的表示只能看見前面，不能同時看見左右文**

因此後來常加入兩個改進：

### 2.1 Attention 的想法

decoder 在第 $t$ 步輸出時，不再只看固定的單一向量，而是對所有 encoder hidden states 重新加權平均：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

也就是說，decoder 每一步都可以「回頭看」輸入序列中最 relevant 的位置。

### 2.2 Bidirectional Encoder 的想法

若只用單向 encoder，則 $h_t$ 只能整合 $x_1$ 到 $x_t$ 的資訊。

Bidirectional Encoder 會同時做：

- 從左到右的 forward RNN
- 從右到左的 backward RNN

然後把兩邊結果接起來：

$$h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

因此第 $t$ 個位置的 encoder 表示，能同時包含「左邊上下文」與「右邊上下文」。

<img src="../image/attention-mechanism.webp" alt="Attention Mechanism"/>

圖片來源： https://towardsdatascience.com/rethinking-thinking-how-do-attention-mechanisms-actually-work-a6f67d313f99/

---

## 3. Bidirectional Encoder 的基本數學模型

Bidirectional Encoder 本質上是兩個方向相反的 RNN。

### 3.1 Forward RNN

對於 $t = 1,2,\ldots,T_x$：

$$\overrightarrow{a}_t = W_{xh}^{\rightarrow} x_t + W_{hh}^{\rightarrow} \overrightarrow{h}_{t-1} + b_h^{\rightarrow}$$

$$\overrightarrow{h}_t = \tanh(\overrightarrow{a}_t)$$

其中初始狀態通常設為：

$$\overrightarrow{h}_0 = 0$$

### 3.2 Backward RNN

對於 $t = T_x, T_x-1, \ldots, 1$：

$$\overleftarrow{a}_t = W_{xh}^{\leftarrow} x_t + W_{hh}^{\leftarrow} \overleftarrow{h}_{t+1} + b_h^{\leftarrow}$$

$$\overleftarrow{h}_t = \tanh(\overleftarrow{a}_t)$$

其中初始條件可視為：

$$\overleftarrow{h}_{T_x+1} = 0$$

### 3.3 合併雙向表示

每個位置 $t$ 的最終 encoder hidden state 為：

$$h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

若：

- $\overrightarrow{h}_t \in \mathbb{R}^{d_h}$
- $\overleftarrow{h}_t \in \mathbb{R}^{d_h}$

則：

$$h_t^{\text{enc}} \in \mathbb{R}^{2d_h}$$

這個維度比單向 encoder 多一倍。

### 3.4 常見 activation：$\tanh$

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

其導數為：

$$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$$

因此若

$$h = \tanh(a)$$

則

$$\frac{\partial h}{\partial a} = 1 - h \odot h$$

這在 attention 模型的 BPTT 中仍然會重複用到。

---

## 4. Encoder 的前向傳播

現在把整個雙向 encoder 的 forward 過程整理起來。

### 4.1 Forward 方向的遞迴

$$\overrightarrow{h}_t = \tanh(W_{xh}^{\rightarrow} x_t + W_{hh}^{\rightarrow} \overrightarrow{h}_{t-1} + b_h^{\rightarrow})$$

### 4.2 Backward 方向的遞迴

$$\overleftarrow{h}_t = \tanh(W_{xh}^{\leftarrow} x_t + W_{hh}^{\leftarrow} \overleftarrow{h}_{t+1} + b_h^{\leftarrow})$$

### 4.3 雙向合併

$$h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t], \quad t=1,\dots,T_x$$

注意：這裡不再只取最後一步 hidden state 當成唯一摘要，而是保留整串：

$$H = (h_1^{\text{enc}}, h_2^{\text{enc}}, \ldots, h_{T_x}^{\text{enc}})$$

之後 attention 會直接使用整個 $H$。

### 4.4 維度檢查

若：

- $x_t \in \mathbb{R}^{d_x}$
- $\overrightarrow{h}_t, \overleftarrow{h}_t \in \mathbb{R}^{d_h}$

則：

$$W_{xh}^{\rightarrow}, W_{xh}^{\leftarrow} \in \mathbb{R}^{d_h \times d_x}$$

$$W_{hh}^{\rightarrow}, W_{hh}^{\leftarrow} \in \mathbb{R}^{d_h \times d_h}$$

而合併後：

$$h_t^{\text{enc}} \in \mathbb{R}^{2d_h}$$

維度正確。✓

---

## 5. Attention 機制的前向傳播

Attention 的核心問題是：

> decoder 在第 $t$ 步，應該關注輸入序列的哪幾個位置？

### 5.1 對齊分數（alignment score）

假設 decoder 在第 $t-1$ 步的 hidden state 為 $s_{t-1}$，那麼它對 encoder 第 $i$ 個 hidden state 的分數可寫成：

$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}})$$

這是一種常見的 **additive attention**（Bahdanau attention）寫法。

其中：

- $s_{t-1}$ 表示 decoder 目前狀態
- $h_i^{\text{enc}}$ 表示輸入序列第 $i$ 個位置的表示
- $e_{t,i}$ 越大，代表 decoder 越想關注第 $i$ 個位置

### 5.2 Softmax 轉成 attention 權重

因為我們希望所有權重加總為 1，所以對 $i=1,\dots,T_x$ 做 softmax：

$$\alpha_{t,i} = \frac{e^{e_{t,i}}}{\sum_{j=1}^{T_x} e^{e_{t,j}}}$$

因此：

$$\sum_{i=1}^{T_x} \alpha_{t,i} = 1$$

這樣 $\alpha_{t,i}$ 就能理解成：

> 在 decoder 第 $t$ 步時，模型分配給輸入位置 $i$ 的注意力比例。

### 5.3 Context vector

有了 attention 權重後，就可以做加權平均：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

這裡的 $c_t$ 不再是固定不變的單一向量，而是每個 decoder 時間步都不同。

### 5.4 為什麼這樣合理

若某個輸入位置很重要，例如翻譯時目前要輸出的字主要對應到輸入第 $i$ 個字，那麼模型就會學到：

$$\alpha_{t,i} \approx 1$$

其他位置的權重接近 0。

因此：

$$c_t \approx h_i^{\text{enc}}$$

也就是說，decoder 在這一步幾乎只看該位置的資訊。

---

## 6. Decoder 的前向傳播

有了 attention 之後，decoder 不只依賴前一 hidden state 和前一輸出，還會依賴目前的 context vector $c_t$。

### 6.1 Decoder 初始條件

一種常見做法是用 encoder 最後的雙向狀態經過線性映射初始化 decoder：

$$s_0 = \tanh(W_{\text{init}} [\overrightarrow{h}_{T_x}; \overleftarrow{h}_1] + b_{\text{init}})$$

這裡：

- $\overrightarrow{h}_{T_x}$ 包含從左到右讀完整句的資訊
- $\overleftarrow{h}_1$ 包含從右到左讀完整句的資訊

因此合併後可作為較完整的初始摘要。

### 6.2 Decoder 遞迴公式

對於 $t = 1,2,\ldots,T_y$：

先計算 attention：

$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}})$$

$$\alpha_{t,i} = \frac{e^{e_{t,i}}}{\sum_{j=1}^{T_x} e^{e_{t,j}}}$$

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

再更新 decoder hidden state：

$$a_t^{\text{dec}} = W_y y_{t-1}^{\text{in}} + W_s s_{t-1} + W_c c_t + b_s$$

$$s_t = \tanh(a_t^{\text{dec}})$$

其中 $y_{t-1}^{\text{in}}$ 是 decoder 的輸入：

- 訓練時通常用真實答案 $y_{t-1}$（teacher forcing）
- 推論時通常用模型上一時刻預測的 token

### 6.3 Teacher Forcing

訓練時常設：

$$y_{t-1}^{\text{in}} = y_{t-1}^{\text{true}}$$

好處是讓 decoder 在訓練初期不至於因為前一步預測錯誤而一路偏掉。

---

## 7. 輸出層與機率分佈

decoder hidden state $s_t$ 與 context vector $c_t$ 都會影響輸出。

### 7.1 線性投影到 logits

一個常見做法是先把兩者結合：

$$o_t = W_o [s_t ; c_t] + b_o$$

其中：

- $s_t \in \mathbb{R}^{d_s}$
- $c_t \in \mathbb{R}^{2d_h}$
- $[s_t;c_t] \in \mathbb{R}^{d_s + 2d_h}$
- $o_t \in \mathbb{R}^{V}$

### 7.2 Softmax 轉成機率

對第 $k$ 個字：

$$\hat{y}_{t,k} = \frac{e^{o_{t,k}}}{\sum_{j=1}^{V} e^{o_{t,j}}}$$

因此整個輸出向量 $\hat{y}_t$ 滿足：

$$\sum_{k=1}^{V} \hat{y}_{t,k} = 1$$

這代表在第 $t$ 步，模型對整個字彙表的預測機率分佈。

---

## 8. 損失函數（Cross-Entropy）

在 Seq2Seq 中，每個 decoder 時間步都是一次分類問題。

### 8.1 單一步驟的交叉熵

若真實標籤 $y_t$ 是 one-hot 向量，則第 $t$ 步 loss 為：

$$\mathcal{L}_t = -\sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k}$$

由於 $y_t$ 是 one-hot，也可寫成：

$$\mathcal{L}_t = -\log \hat{y}_{t,\text{target}}$$

### 8.2 整段序列的總損失

$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}_t = -\sum_{t=1}^{T_y}\sum_{k=1}^{V} y_{t,k}\log \hat{y}_{t,k}$$

若要取平均，也可寫為：

$$\mathcal{L}_{\text{avg}} = \frac{1}{T_y}\mathcal{L}$$

### 8.3 Softmax + Cross-Entropy 的重要結果

若

$$\hat{y}_t = \text{softmax}(o_t)$$

則對 logits 的梯度仍然有漂亮結果：

$$\boxed{\frac{\partial \mathcal{L}_t}{\partial o_t} = \hat{y}_t - y_t}$$

這使得 attention 模型的輸出層反向傳播與基本 Seq2Seq 相同。

---

## 9. 反向傳播：Attention 與 Bi-Encoder 的梯度推導

Attention 模型的梯度流動，和基本 Seq2Seq 最大差異在於：

1. 每個 decoder 時間步都會透過 $c_t$ 連到所有 encoder hidden states
2. encoder 每個位置都可能被多個 decoder 時間步同時使用
3. Bidirectional Encoder 還要把梯度拆成 forward 與 backward 兩條鏈

---

### 9.1 輸出層梯度

定義：

$$\delta_t^o = \frac{\partial \mathcal{L}}{\partial o_t}$$

則由 softmax + cross-entropy 可得：

$$\boxed{\delta_t^o = \hat{y}_t - y_t}$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_o} = \sum_{t=1}^{T_y} \delta_t^o [s_t;c_t]^T$$

$$\frac{\partial \mathcal{L}}{\partial b_o} = \sum_{t=1}^{T_y} \delta_t^o$$

而對 concatenated 向量的梯度為：

$$\frac{\partial \mathcal{L}}{\partial [s_t;c_t]} = W_o^T \delta_t^o$$

把它拆開後可得：

$$g_t^{(s,\text{out})} = \frac{\partial \mathcal{L}}{\partial s_t}\Big|_{\text{from output}}$$

$$g_t^{(c)} = \frac{\partial \mathcal{L}}{\partial c_t}$$

其中 $g_t^{(c)}$ 之後會傳進 attention。

### 9.2 Decoder hidden state 的時間反傳

decoder 預激活為：

$$a_t^{\text{dec}} = W_y y_{t-1}^{\text{in}} + W_s s_{t-1} + W_c c_t + b_s$$

$$s_t = \tanh(a_t^{\text{dec}})$$

令：

$$\delta_t^{\text{dec}} = \frac{\partial \mathcal{L}}{\partial a_t^{\text{dec}}}$$

則總的 $\frac{\partial \mathcal{L}}{\partial s_t}$ 會來自兩部分：

1. 第 $t$ 步輸出層
2. 第 $t+1$ 步 decoder 遞迴

因此：

$$\frac{\partial \mathcal{L}}{\partial s_t} = g_t^{(s,\text{out})} + W_s^T \delta_{t+1}^{\text{dec}} + g_t^{(\text{attn-back})}$$

這裡的 $g_t^{(\text{attn-back})}$ 是因為 $s_t$ 也會影響下一步 attention score，所以 attention 也會反傳梯度給 decoder hidden state。

再乘上 $\tanh$ 導數：

$$\boxed{\delta_t^{\text{dec}} = \left(g_t^{(s,\text{out})} + W_s^T \delta_{t+1}^{\text{dec}} + g_t^{(\text{attn-back})}\right)\odot (1-s_t\odot s_t)}$$

邊界條件通常設：

$$\delta_{T_y+1}^{\text{dec}} = 0$$

### 9.3 Context vector 如何把梯度傳到 encoder

因為：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

所以對每個 encoder hidden state，

$$\frac{\partial \mathcal{L}}{\partial h_i^{\text{enc}}}\Big|_{\text{direct from } c_t} = \sum_{t=1}^{T_y} \alpha_{t,i} \, g_t^{(c)}$$

這表示：

> encoder 第 $i$ 個 hidden state 會同時收到所有 decoder 時間步傳回來的梯度，而且每一步的權重由 $\alpha_{t,i}$ 決定。

若某一位置在很多步都被高度關注，那它收到的總梯度就會比較大。

### 9.4 Attention score 的梯度

由於 $c_t$ 也依賴 $\alpha_{t,i}$，因此還要先對 attention 權重求導。

因為：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

故：

$$\frac{\partial \mathcal{L}}{\partial \alpha_{t,i}} = \left(g_t^{(c)}\right)^T h_i^{\text{enc}}$$

接著，$\alpha_{t,i}$ 是 softmax over $e_{t,i}$，所以若定義

$$g_{t,i}^{(\alpha)} = \frac{\partial \mathcal{L}}{\partial \alpha_{t,i}}$$

則對 score 的梯度可由 softmax Jacobian 得到：

$$\frac{\partial \mathcal{L}}{\partial e_{t,i}} = \sum_{j=1}^{T_x} \frac{\partial \mathcal{L}}{\partial \alpha_{t,j}}\frac{\partial \alpha_{t,j}}{\partial e_{t,i}}$$

實作上常寫成 softmax backward 的標準形式。若記

$$\delta_{t,i}^{(e)} = \frac{\partial \mathcal{L}}{\partial e_{t,i}}$$

則這個量會再傳進 additive attention 的內部 $\tanh$。

### 9.5 Additive attention 內部梯度

attention score 定義為：

$$e_{t,i} = v_a^T z_{t,i}$$

其中

$$z_{t,i} = \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}})$$

因此：

$$\frac{\partial \mathcal{L}}{\partial v_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(e)} z_{t,i}$$

而對 $z_{t,i}$ 的梯度為：

$$\frac{\partial \mathcal{L}}{\partial z_{t,i}} = \delta_{t,i}^{(e)} v_a$$

再經過 $\tanh$：

$$\delta_{t,i}^{(z)} = \left(\delta_{t,i}^{(e)} v_a\right)\odot (1-z_{t,i}\odot z_{t,i})$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(z)} s_{t-1}^T$$

$$\frac{\partial \mathcal{L}}{\partial U_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(z)} (h_i^{\text{enc}})^T$$

並且 attention 也會回傳梯度到：

- decoder state $s_{t-1}$
- encoder state $h_i^{\text{enc}}$

具體地：

$$g_t^{(\text{attn-back})} = \sum_{i=1}^{T_x} W_a^T \delta_{t+1,i}^{(z)}$$

以及

$$\frac{\partial \mathcal{L}}{\partial h_i^{\text{enc}}}\Big|_{\text{from score}} = \sum_{t=1}^{T_y} U_a^T \delta_{t,i}^{(z)}$$

所以 encoder hidden state 的總梯度為：

$$\boxed{
\frac{\partial \mathcal{L}}{\partial h_i^{\text{enc}}}
=
\sum_{t=1}^{T_y} \alpha_{t,i} g_t^{(c)}
+
\sum_{t=1}^{T_y} U_a^T \delta_{t,i}^{(z)}
}$$

第一項來自 context vector 的加權和，第二項來自 attention score 本身。

### 9.6 Bidirectional Encoder 的梯度拆解

因為

$$h_i^{\text{enc}} = [\overrightarrow{h}_i ; \overleftarrow{h}_i]$$

所以若定義

$$g_i^{\text{enc}} = \frac{\partial \mathcal{L}}{\partial h_i^{\text{enc}}}$$

則可把它拆成兩部分：

$$g_i^{\text{enc}} = [g_i^{\rightarrow} ; g_i^{\leftarrow}]$$

接下來：

- $g_i^{\rightarrow}$ 進入 forward RNN 的 BPTT
- $g_i^{\leftarrow}$ 進入 backward RNN 的 BPTT

### 9.7 Forward Encoder 的時間反傳

forward encoder 為：

$$\overrightarrow{a}_t = W_{xh}^{\rightarrow} x_t + W_{hh}^{\rightarrow}\overrightarrow{h}_{t-1} + b_h^{\rightarrow}$$

$$\overrightarrow{h}_t = \tanh(\overrightarrow{a}_t)$$

令：

$$\delta_t^{\rightarrow} = \frac{\partial \mathcal{L}}{\partial \overrightarrow{a}_t}$$

則：

$$\boxed{
\delta_t^{\rightarrow}
=
\left(
g_t^{\rightarrow}
+
(W_{hh}^{\rightarrow})^T \delta_{t+1}^{\rightarrow}
\right)
\odot
(1-\overrightarrow{h}_t\odot\overrightarrow{h}_t)
}$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\rightarrow}} = \sum_{t=1}^{T_x} \delta_t^{\rightarrow} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\rightarrow}} = \sum_{t=1}^{T_x} \delta_t^{\rightarrow} (\overrightarrow{h}_{t-1})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\rightarrow}} = \sum_{t=1}^{T_x} \delta_t^{\rightarrow}$$

### 9.8 Backward Encoder 的時間反傳

backward encoder 為：

$$\overleftarrow{a}_t = W_{xh}^{\leftarrow} x_t + W_{hh}^{\leftarrow}\overleftarrow{h}_{t+1} + b_h^{\leftarrow}$$

$$\overleftarrow{h}_t = \tanh(\overleftarrow{a}_t)$$

令：

$$\delta_t^{\leftarrow} = \frac{\partial \mathcal{L}}{\partial \overleftarrow{a}_t}$$

由於 backward RNN 的時間方向相反，所以其反傳公式會寫成：

$$\boxed{
\delta_t^{\leftarrow}
=
\left(
g_t^{\leftarrow}
+
(W_{hh}^{\leftarrow})^T \delta_{t-1}^{\leftarrow}
\right)
\odot
(1-\overleftarrow{h}_t\odot\overleftarrow{h}_t)
}$$

因此：

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\leftarrow}} = \sum_{t=1}^{T_x} \delta_t^{\leftarrow} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\leftarrow}} = \sum_{t=1}^{T_x} \delta_t^{\leftarrow} (\overleftarrow{h}_{t+1})^T$$

$$\frac{\partial \mathcal{L}}{\partial b_h^{\leftarrow}} = \sum_{t=1}^{T_x} \delta_t^{\leftarrow}$$

---

## 10. 參數更新：梯度下降法

當所有梯度都算出來後，就可以更新參數。

### 10.1 梯度下降更新規則

對任一參數 $\theta$：

$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}$$

其中 $\alpha$ 是學習率。

例如：

$$W_o \leftarrow W_o - \alpha \frac{\partial \mathcal{L}}{\partial W_o}$$

$$W_a \leftarrow W_a - \alpha \frac{\partial \mathcal{L}}{\partial W_a}$$

$$U_a \leftarrow U_a - \alpha \frac{\partial \mathcal{L}}{\partial U_a}$$

$$W_{xh}^{\rightarrow} \leftarrow W_{xh}^{\rightarrow} - \alpha \frac{\partial \mathcal{L}}{\partial W_{xh}^{\rightarrow}}$$

$$W_{xh}^{\leftarrow} \leftarrow W_{xh}^{\leftarrow} - \alpha \frac{\partial \mathcal{L}}{\partial W_{xh}^{\leftarrow}}$$

### 10.2 為什麼 Attention 通常比固定 context 更有效

因為固定 context 的模型必須把整句壓成一個向量，長句時資訊瓶頸明顯。

Attention 改成：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

代表 decoder 每一步都能重新讀取輸入序列，所以：

1. 長距離資訊較不容易遺失
2. 對齊關係更清楚
3. 梯度可以直接從 decoder 傳到 encoder 的各個位置

### 10.3 為什麼 Bidirectional Encoder 更有效

因為每個 encoder hidden state 都能同時看見左右文：

$$h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

例如一句話中的某個詞，其含義常常要同時依賴前後文。雙向編碼能更完整表示這種資訊。

---

## 11. 完整流程整理

最後，把整個 Attention + Bidirectional Encoder 的核心數學流程整理如下。

### 11.1 Forward

#### Bidirectional Encoder：

$$\overrightarrow{h}_t = \tanh(W_{xh}^{\rightarrow} x_t + W_{hh}^{\rightarrow} \overrightarrow{h}_{t-1} + b_h^{\rightarrow}), \quad t=1,\dots,T_x$$

$$\overleftarrow{h}_t = \tanh(W_{xh}^{\leftarrow} x_t + W_{hh}^{\leftarrow} \overleftarrow{h}_{t+1} + b_h^{\leftarrow}), \quad t=T_x,\dots,1$$

$$h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

#### Attention：

$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}})$$

$$\alpha_{t,i} = \frac{e^{e_{t,i}}}{\sum_{j=1}^{T_x} e^{e_{t,j}}}$$

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i^{\text{enc}}$$

#### Decoder：

$$a_t^{\text{dec}} = W_y y_{t-1}^{\text{in}} + W_s s_{t-1} + W_c c_t + b_s$$

$$s_t = \tanh(a_t^{\text{dec}})$$

$$o_t = W_o [s_t;c_t] + b_o$$

$$\hat{y}_t = \text{softmax}(o_t)$$

### 11.2 Loss

$$\mathcal{L} = -\sum_{t=1}^{T_y}\sum_{k=1}^{V} y_{t,k}\log \hat{y}_{t,k}$$

### 11.3 Backward

#### 輸出層：

$$\delta_t^o = \hat{y}_t - y_t$$

#### Decoder BPTT：

$$\delta_t^{\text{dec}} = \left(g_t^{(s,\text{out})} + W_s^T \delta_{t+1}^{\text{dec}} + g_t^{(\text{attn-back})}\right)\odot (1-s_t\odot s_t)$$

#### Attention 對 encoder 的總梯度：

$$\frac{\partial \mathcal{L}}{\partial h_i^{\text{enc}}}
=
\sum_{t=1}^{T_y} \alpha_{t,i} g_t^{(c)}
+
\sum_{t=1}^{T_y} U_a^T \delta_{t,i}^{(z)}$$

#### Forward Encoder BPTT：

$$\delta_t^{\rightarrow}
=
\left(
g_t^{\rightarrow}
+
(W_{hh}^{\rightarrow})^T \delta_{t+1}^{\rightarrow}
\right)
\odot
(1-\overrightarrow{h}_t\odot\overrightarrow{h}_t)$$

#### Backward Encoder BPTT：

$$\delta_t^{\leftarrow}
=
\left(
g_t^{\leftarrow}
+
(W_{hh}^{\leftarrow})^T \delta_{t-1}^{\leftarrow}
\right)
\odot
(1-\overleftarrow{h}_t\odot\overleftarrow{h}_t)$$

### 11.4 參數梯度

$$\frac{\partial \mathcal{L}}{\partial W_o} = \sum_{t=1}^{T_y} \delta_t^o [s_t;c_t]^T$$

$$\frac{\partial \mathcal{L}}{\partial W_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(z)} s_{t-1}^T$$

$$\frac{\partial \mathcal{L}}{\partial U_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(z)} (h_i^{\text{enc}})^T$$

$$\frac{\partial \mathcal{L}}{\partial v_a} = \sum_{t=1}^{T_y}\sum_{i=1}^{T_x} \delta_{t,i}^{(e)} z_{t,i}$$

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\rightarrow}} = \sum_{t=1}^{T_x} \delta_t^{\rightarrow} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\rightarrow}} = \sum_{t=1}^{T_x} \delta_t^{\rightarrow} (\overrightarrow{h}_{t-1})^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{xh}^{\leftarrow}} = \sum_{t=1}^{T_x} \delta_t^{\leftarrow} x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}^{\leftarrow}} = \sum_{t=1}^{T_x} \delta_t^{\leftarrow} (\overleftarrow{h}_{t+1})^T$$

---

## 總結：Attention + Bi-Encoder 的七個核心公式

$$\boxed{
\begin{aligned}
&\textbf{(1) 雙向編碼：} && h_t^{\text{enc}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t] \\
&\textbf{(2) 對齊分數：} && e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}}) \\
&\textbf{(3) 注意力權重：} && \alpha_{t,i} = \text{softmax}(e_{t,i}) \\
&\textbf{(4) 動態 context：} && c_t = \sum_{i=1}^{T_x}\alpha_{t,i} h_i^{\text{enc}} \\
&\textbf{(5) Decoder 更新：} && s_t = \tanh(W_y y_{t-1}^{\text{in}} + W_s s_{t-1} + W_c c_t + b_s) \\
&\textbf{(6) 輸出機率：} && \hat{y}_t = \text{softmax}(W_o[s_t;c_t] + b_o) \\
&\textbf{(7) 核心觀念：} && \text{每個 decoder 時間步都能重新關注整個輸入序列}
\end{aligned}
}$$

這些公式構成了最基本 **RNN-based Seq2Seq with Attention + Bidirectional Encoder** 的理論基礎。理解這套推導之後，再往上學 multi-head attention、self-attention、Transformer encoder-decoder，就會順很多。

---

*參考概念：微積分鏈鎖律 · 線性代數矩陣運算 · RNN 時間展開 · Softmax Jacobian · Cross-Entropy · Backpropagation Through Time (BPTT) · Bahdanau Attention*
