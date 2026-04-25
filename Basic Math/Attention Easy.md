# Attention 機制完整推導 — 高中生版

> 保留完整的前向傳播、Attention、反向傳播（BPTT）推導歷程，
> 但所有數學只用高中程度（無矩陣），並附上完整數值模擬驗證每一步。

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [為什麼需要 Attention](#2-為什麼需要-attention)
3. [核心數學工具：tanh 與 softmax](#3-核心數學工具tanh-與-softmax)
4. [Encoder 的前向傳播](#4-encoder-的前向傳播)
5. [Attention 機制的前向傳播](#5-attention-機制的前向傳播)
6. [Decoder 的前向傳播](#6-decoder-的前向傳播)
7. [輸出層與機率分佈](#7-輸出層與機率分佈)
8. [損失函數（Cross-Entropy）](#8-損失函數cross-entropy)
9. [反向傳播：梯度推導](#9-反向傳播梯度推導)
10. [參數更新：梯度下降法](#10-參數更新梯度下降法)
11. [完整數值模擬](#11-完整數值模擬)
12. [完整流程整理](#12-完整流程整理)

---

## 1. 基本符號定義

### 1.1 序列資料的表示方式

假設我們在做翻譯，輸入英文、輸出中文：

```
輸入（英文）：  I    love  cats
輸出（中文）：  我    愛    貓
```

- 輸入序列長度 $T_x = 3$（三個英文字）
- 輸出序列長度 $T_y = 3$（三個中文字）
- 輸入序列寫成：$x_1, x_2, \ldots, x_{T_x}$
- 目標輸出序列寫成：$y_1, y_2, \ldots, y_{T_y}$

每個字會先轉成一個數字（在真實模型中是向量，但本文為了清晰只用單一數字）。

> **Token** 是模型處理文字的最小單位。1000 個 Token 大約等於 750 個中文字。

### 1.2 主要符號表

| 符號 | 意義 |
|------|------|
| $T_x$ | 輸入序列長度 |
| $T_y$ | 輸出序列長度 |
| $x_t$ | 第 $t$ 個輸入的數值 |
| $y_t$ | 第 $t$ 個正確輸出的標籤 |
| $\hat{y}_t$ | 第 $t$ 步的預測機率 |
| $h_t^{\text{enc}}$ | Encoder 在第 $t$ 步的 hidden state |
| $s_t$ | Decoder 在第 $t$ 步的 hidden state |
| $e_{t,i}$ | Decoder 第 $t$ 步對 Encoder 第 $i$ 步的對齊分數 |
| $\alpha_{t,i}$ | Attention 權重（$\sum_i \alpha_{t,i} = 1$） |
| $c_t$ | 第 $t$ 步的 context vector（加權摘要） |
| $w_a,\, u_a,\, v$ | Attention 參數 |
| $w,\, u,\, b$ | Encoder RNN 參數 |
| $w_s,\, w_c,\, b_s$ | Decoder RNN 參數 |
| $w_o,\, b_o$ | 輸出層參數 |
| $L$ | 損失函數 |

---

## 2. 為什麼需要 Attention

### 2.1 原本的問題

最早的 Seq2Seq 模型會把整句輸入壓縮成一個固定的數字（或向量）：

$$c = h_{T_x}^{\text{enc}}$$

這樣做有三個問題：

1. **句子太長時，最後的 hidden state 裝不下所有資訊**
2. **Decoder 每一步都只能看到同一個固定值，無法隨需求調整**
3. **對齊關係不清楚**：輸出「愛」時，模型應該看「love」，但它被迫看整句的壓縮結果

### 2.2 Attention 的解法

Decoder 在第 $t$ 步，不再只看一個固定值，而是對所有 Encoder 的 hidden states 做**加權平均**：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i^{\text{enc}}$$

其中 $\alpha_{t,i}$ 是「Decoder 第 $t$ 步對輸入第 $i$ 位置的注意力比例」。

這樣 Decoder 每一步都能「回頭看」輸入中最重要的位置，而且每步看的焦點可以不同。

---

## 3. 核心數學工具：tanh 與 softmax

### 3.1 tanh（雙曲正切）

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

它把任意數字壓進 $(-1,\, +1)$ 之間，是 RNN 常用的激活函數。

**常用數值：**

| $x$ | $\tanh(x)$ |
|-----|-----------|
| $-2$ | $-0.964$ |
| $-1$ | $-0.762$ |
| $0$ | $0.000$ |
| $1$ | $+0.762$ |
| $2$ | $+0.964$ |

**導數**（反向傳播時用到）：

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

因此，若已知 $h = \tanh(a)$，則：

$$\frac{dh}{da} = 1 - h^2$$

**範例**：$\tanh(0.5) = 0.462$，其導數 $= 1 - 0.462^2 = 1 - 0.213 = 0.787$

### 3.2 softmax

softmax 把一組數字轉成「加總為 1 的機率」。

若有三個數 $e_1, e_2, e_3$：

$$\alpha_i = \frac{e^{e_i}}{e^{e_1} + e^{e_2} + e^{e_3}}$$

**範例**：輸入 $(0.505,\ 0.751,\ 0.127)$

$$e^{0.505} = 1.657,\quad e^{0.751} = 2.119,\quad e^{0.127} = 1.135$$

$$\text{總和} = 4.911$$

$$\alpha_1 = 0.337,\quad \alpha_2 = 0.431,\quad \alpha_3 = 0.231$$

驗算：$0.337 + 0.431 + 0.231 = 0.999 \approx 1$ ✓

---

## 4. Encoder 的前向傳播

### 4.1 Encoder 的工作

Encoder 依序讀入每個輸入字，將「目前讀到的整體資訊」壓縮進一個數字 $h_t$（hidden state）。

### 4.2 前向計算公式

對於 $t = 1, 2, \ldots, T_x$，先算**預激活值** $a_t$，再套 tanh：

$$a_t = w \cdot x_t + u \cdot h_{t-1} + b$$

$$h_t^{\text{enc}} = \tanh(a_t)$$

其中：
- $w$ 是輸入的權重（影響當前輸入有多重要）
- $u$ 是遞迴權重（影響上一步記憶有多重要）
- $b$ 是偏差值（bias）
- 初始狀態 $h_0 = 0$

### 4.3 為什麼保留每一步的 $h_t^{\text{enc}}$？

和舊的 Seq2Seq 只保留最後一步不同，Attention 模型**保留所有步驟的 hidden state**：

$$H = (h_1^{\text{enc}},\, h_2^{\text{enc}},\, \ldots,\, h_{T_x}^{\text{enc}})$$

之後 Attention 機制就是從這個 $H$ 中動態選取資訊。

### 4.4 數值模擬

設定參數：$w = 0.8$，$u = 0.5$，$b = 0.1$，初始 $h_0 = 0$

輸入：$x_1 = 0.5$（"I"），$x_2 = 1.0$（"love"），$x_3 = -0.5$（"cats"）

**t = 1：**

$$a_1 = 0.8 \times 0.5 + 0.5 \times 0 + 0.1 = 0.5$$

$$h_1^{\text{enc}} = \tanh(0.5) = 0.462$$

**t = 2：**

$$a_2 = 0.8 \times 1.0 + 0.5 \times 0.462 + 0.1 = 0.8 + 0.231 + 0.1 = 1.131$$

$$h_2^{\text{enc}} = \tanh(1.131) = 0.812$$

**t = 3：**

$$a_3 = 0.8 \times (-0.5) + 0.5 \times 0.812 + 0.1 = -0.4 + 0.406 + 0.1 = 0.106$$

$$h_3^{\text{enc}} = \tanh(0.106) = 0.106$$

**Encoder 輸出：**

$$h_1^{\text{enc}} = 0.462, \quad h_2^{\text{enc}} = 0.812, \quad h_3^{\text{enc}} = 0.106$$

---

## 5. Attention 機制的前向傳播

### 5.1 對齊分數（alignment score）

Decoder 在第 $t$ 步，用當前的狀態 $s_{t-1}$ 去詢問：

> 「我現在的狀態是 $s_{t-1}$，輸入的每個位置 $i$ 對我有多重要？」

計算**對齊分數**（additive attention，Bahdanau attention）：

$$e_{t,i} = v \cdot \tanh(w_a \cdot s_{t-1} + u_a \cdot h_i^{\text{enc}})$$

其中 $v,\, w_a,\, u_a$ 是 Attention 自己的可學習參數。

- $s_{t-1}$：Decoder 目前的狀態（「我現在在做什麼」）
- $h_i^{\text{enc}}$：Encoder 第 $i$ 位置的記憶（「輸入第 $i$ 個字是什麼感覺」）
- $e_{t,i}$ 越大 → Decoder 越想關注第 $i$ 個位置

### 5.2 轉成 Attention 權重

對所有位置的分數做 softmax，得到**機率式**的注意力比例：

$$\alpha_{t,i} = \frac{e^{e_{t,i}}}{\displaystyle\sum_{j=1}^{T_x} e^{e_{t,j}}}$$

保證：

$$\sum_{i=1}^{T_x} \alpha_{t,i} = 1$$

可以理解為：Decoder 在第 $t$ 步把 100% 的注意力分配給各個輸入位置的比例。

### 5.3 Context vector（動態摘要）

用 Attention 權重對所有 Encoder hidden states 做加權平均：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i^{\text{enc}}$$

這裡的 $c_t$ 每步都不同——它是根據 Decoder 當前需求「量身打造」的輸入摘要。

**極端情況**：若 $\alpha_{t,2} \approx 1$（其餘接近 0），則：

$$c_t \approx h_2^{\text{enc}}$$

Decoder 這一步幾乎只看輸入第 2 個位置的資訊。

### 5.4 數值模擬（Decoder 輸出第 1 個字，$s_0 = 0$）

設定 Attention 參數：$w_a = 1.0$，$u_a = 1.2$，$v = 1.0$

**計算對齊分數 $e_{1,i}$（$s_0 = 0$，$w_a \cdot s_0 = 0$）：**

$$e_{1,1} = 1.0 \times \tanh(0 + 1.2 \times 0.462) = \tanh(0.554) = 0.505$$

$$e_{1,2} = 1.0 \times \tanh(0 + 1.2 \times 0.812) = \tanh(0.974) = 0.751$$

$$e_{1,3} = 1.0 \times \tanh(0 + 1.2 \times 0.106) = \tanh(0.127) = 0.127$$

**做 softmax 得到 Attention 權重：**

$$e^{0.505} = 1.657,\quad e^{0.751} = 2.119,\quad e^{0.127} = 1.135, \quad \text{總和} = 4.911$$

$$\alpha_{1,1} = \frac{1.657}{4.911} = 0.337, \quad \alpha_{1,2} = \frac{2.119}{4.911} = 0.431, \quad \alpha_{1,3} = \frac{1.135}{4.911} = 0.231$$

**計算 context vector $c_1$：**

$$c_1 = 0.337 \times 0.462 + 0.431 \times 0.812 + 0.231 \times 0.106$$

$$c_1 = 0.156 + 0.350 + 0.024 = 0.530$$

> 📌 $\alpha_{1,2} = 0.431$ 最大，Decoder 第 1 步最關注「love」（位置 2）。

---

## 6. Decoder 的前向傳播

### 6.1 Decoder 的初始狀態

Decoder 的起始 hidden state 通常用 Encoder 最後的輸出來初始化：

$$s_0 = \tanh(w_{\text{init}} \cdot h_{T_x}^{\text{enc}} + b_{\text{init}})$$

簡化版本（本文直接設 $s_0 = 0$）。

### 6.2 Decoder 的遞迴公式

每一步，Decoder 會把三件事情合併：

1. 前一個 hidden state $s_{t-1}$（「我上一步的狀態」）
2. Context vector $c_t$（「Attention 告訴我要看哪裡」）
3. 前一個輸出（訓練時用真實答案，稱為 teacher forcing）

$$a_t^{\text{dec}} = w_s \cdot s_{t-1} + w_c \cdot c_t + b_s$$

$$s_t = \tanh(a_t^{\text{dec}})$$

### 6.3 Teacher Forcing

訓練時常把前一步的**正確答案**當成輸入（而非模型的預測），這樣做讓訓練更穩定。

### 6.4 數值模擬（第 1 步，$s_0 = 0$，$c_1 = 0.530$）

設 Decoder 參數：$w_s = 0.7$，$w_c = 0.9$，$b_s = 0.0$

$$a_1^{\text{dec}} = 0.7 \times 0 + 0.9 \times 0.530 + 0.0 = 0.477$$

$$s_1 = \tanh(0.477) = 0.444$$

---

## 7. 輸出層與機率分佈

### 7.1 從 hidden state 到輸出分數

Decoder 的 hidden state $s_t$ 與 context vector $c_t$ 合併後，通過輸出層得到每個字的分數（logits）：

$$o_t = w_o \cdot s_t + b_o$$

（簡化版本：把 $s_t$ 和 $c_t$ 合併後視為一個數值，乘上各字的輸出權重）

### 7.2 Softmax 轉成機率

對每個字 $k$：

$$\hat{y}_{t,k} = \frac{e^{o_{t,k}}}{\displaystyle\sum_{j=1}^{V} e^{o_{t,j}}}$$

其中 $V$ 是字彙表大小。結果滿足 $\sum_k \hat{y}_{t,k} = 1$。

### 7.3 數值模擬（簡化詞彙表只有 2 個字：A 和 B）

設輸出權重：$w_o^{(A)} = 1.5$，$w_o^{(B)} = -0.5$，偏差 $b_o = 0$

$$o_1^{(A)} = 1.5 \times 0.444 = 0.666$$

$$o_1^{(B)} = -0.5 \times 0.444 = -0.222$$

做 softmax：

$$e^{0.666} = 1.946,\quad e^{-0.222} = 0.801,\quad \text{總和} = 2.747$$

$$\hat{y}_1^{(A)} = \frac{1.946}{2.747} = 0.708, \quad \hat{y}_1^{(B)} = \frac{0.801}{2.747} = 0.292$$

> 模型預測字 A 的機率是 70.8%，字 B 是 29.2%。

---

## 8. 損失函數（Cross-Entropy）

### 8.1 單一步驟的交叉熵

若正確答案是字 A（用 one-hot 表示：$y^{(A)}=1, y^{(B)}=0$），則：

$$L_t = -\sum_{k} y_t^{(k)} \log \hat{y}_t^{(k)} = -(1 \times \log \hat{y}_t^{(A)} + 0 \times \log \hat{y}_t^{(B)}) = -\log \hat{y}_t^{(A)}$$

**直觀理解**：

- 若 $\hat{y}^{(A)} = 1.0$（完全確信正確答案）：$L = -\log(1) = 0$
- 若 $\hat{y}^{(A)} = 0.5$（一半一半）：$L = -\log(0.5) = 0.693$
- 若 $\hat{y}^{(A)} = 0.1$（幾乎不信）：$L = -\log(0.1) = 2.303$

損失越小，預測越準。

### 8.2 整段序列的總損失

$$L = \sum_{t=1}^{T_y} L_t = -\sum_{t=1}^{T_y} \log \hat{y}_t^{(\text{正確答案})}$$

### 8.3 Softmax + Cross-Entropy 的重要結果

當我們把 softmax 和 cross-entropy 組合在一起，對輸出分數 $o_t^{(k)}$ 的梯度有一個非常漂亮的形式：

$$\boxed{\frac{\partial L_t}{\partial o_t^{(k)}} = \hat{y}_t^{(k)} - y_t^{(k)}}$$

**推導說明**：

設正確答案是類別 $k^*$，則 $y^{(k^*)} = 1$，其他 $y^{(k)} = 0$。

由鏈鎖律：

$$\frac{\partial L_t}{\partial o_t^{(k)}} = \sum_j \frac{\partial L_t}{\partial \hat{y}_t^{(j)}} \cdot \frac{\partial \hat{y}_t^{(j)}}{\partial o_t^{(k)}}$$

其中 softmax 的偏微分為：

$$\frac{\partial \hat{y}_t^{(j)}}{\partial o_t^{(k)}} = \hat{y}_t^{(j)}(\delta_{jk} - \hat{y}_t^{(k)})$$

（$\delta_{jk}$ 在 $j=k$ 時為 1，其他為 0）

代入 $\frac{\partial L_t}{\partial \hat{y}_t^{(j)}} = -\frac{y_t^{(j)}}{\hat{y}_t^{(j)}}$ 後整理，最終得到：

$$\frac{\partial L_t}{\partial o_t^{(k)}} = \hat{y}_t^{(k)} - y_t^{(k)}$$

這是一個非常優雅的結果，讓輸出層反向傳播變得很簡單。

### 8.4 數值模擬（正確答案是字 A）

$$L_1 = -\log(0.708) = 0.346$$

$$\frac{\partial L_1}{\partial o_1^{(A)}} = 0.708 - 1 = -0.292 \quad (\text{要增加 A 的分數})$$

$$\frac{\partial L_1}{\partial o_1^{(B)}} = 0.292 - 0 = +0.292 \quad (\text{要減少 B 的分數})$$

---

## 9. 反向傳播：梯度推導

反向傳播的核心是**鏈鎖律**（Chain Rule）：

$$\frac{dL}{d\theta} = \frac{dL}{d\text{輸出}} \times \frac{d\text{輸出}}{d\theta}$$

逐層往回傳，每一步都套這個規則。

Attention 模型的梯度流動有三個特點：
1. 每個 Decoder 時間步都透過 $c_t$ 連到所有 Encoder hidden states
2. Encoder 每個位置可能被多個 Decoder 時間步使用
3. 梯度可以直接從 Decoder「跳回」Encoder 的特定位置

### 9.1 輸出層梯度

由第 8.3 節的結果，定義：

$$\delta_t^o = \frac{\partial L}{\partial o_t}$$

則：

$$\boxed{\delta_t^o = \hat{y}_t - y_t}$$

輸出層參數的梯度：

$$\frac{\partial L}{\partial w_o^{(k)}} = \sum_{t=1}^{T_y} \delta_t^{o,(k)} \cdot s_t$$

$$\frac{\partial L}{\partial b_o} = \sum_{t=1}^{T_y} \delta_t^o$$

然後梯度往回傳到 $s_t$：

$$g_t^{(s, \text{out})} = \frac{\partial L}{\partial s_t}\Big|_{\text{來自輸出層}} = \sum_k w_o^{(k)} \cdot \delta_t^{o,(k)}$$

同時梯度也往回傳到 $c_t$（之後傳進 Attention）：

$$g_t^{(c)} = \frac{\partial L}{\partial c_t} = \sum_k w_o^{(k)} \cdot \delta_t^{o,(k)}$$

（在純量版本中，$g_t^{(s,\text{out})}$ 和 $g_t^{(c)}$ 由輸出層的不同部分計算，簡化版這裡統一用同一公式。）

### 9.2 Decoder hidden state 的時間反傳（BPTT）

Decoder 每一步的 hidden state $s_t$ 影響兩件事：
1. 第 $t$ 步的輸出（透過 $o_t$）
2. 第 $t+1$ 步的 Decoder 狀態（透過遞迴）
3. 第 $t+1$ 步的 Attention score（$s_t$ 進入下一步的 $e_{t+1,i}$）

因此 $s_t$ 的**總梯度**是三部分之和：

$$\frac{\partial L}{\partial s_t} = g_t^{(s,\text{out})} + w_s \cdot \delta_{t+1}^{\text{dec}} + g_t^{(\text{attn-back})}$$

其中 $g_t^{(\text{attn-back})}$ 是從下一步 Attention 傳回來的梯度。

再乘上 tanh 的導數（穿越激活函數）：

$$\boxed{\delta_t^{\text{dec}} = \frac{\partial L}{\partial a_t^{\text{dec}}} = \frac{\partial L}{\partial s_t} \times (1 - s_t^2)}$$

邊界條件：$\delta_{T_y+1}^{\text{dec}} = 0$（最後一步之後沒有梯度傳入）

**鏈鎖律說明**：

$$\frac{\partial L}{\partial a_t^{\text{dec}}} = \frac{\partial L}{\partial s_t} \cdot \frac{\partial s_t}{\partial a_t^{\text{dec}}} = \frac{\partial L}{\partial s_t} \cdot (1 - s_t^2)$$

因為 $s_t = \tanh(a_t^{\text{dec}})$，所以 $\frac{\partial s_t}{\partial a_t^{\text{dec}}} = 1 - \tanh^2(a_t^{\text{dec}}) = 1 - s_t^2$。

### 9.3 Context vector 如何把梯度傳回 Encoder

因為：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i^{\text{enc}}$$

對 Encoder 第 $i$ 個 hidden state，第 $t$ 步 context vector 傳來的梯度是：

$$\frac{\partial L}{\partial h_i^{\text{enc}}}\bigg|_{\text{來自} c_t} = \alpha_{t,i} \cdot g_t^{(c)}$$

由於所有 $T_y$ 個 Decoder 步驟都會用到 $h_i^{\text{enc}}$，所以**總梯度要加總所有步驟**：

$$\frac{\partial L}{\partial h_i^{\text{enc}}}\bigg|_{\text{context}} = \sum_{t=1}^{T_y} \alpha_{t,i} \cdot g_t^{(c)}$$

> 若某個 Encoder 位置在很多步都被高度關注（$\alpha_{t,i}$ 大），它收到的總梯度就越大，模型就會更努力學好該位置的表示。

**同時**，$h_i^{\text{enc}}$ 也影響 Attention score $e_{t,i}$（見 9.4 節），所以還有第二路梯度。

### 9.4 Attention score 的梯度

因為 $c_t$ 依賴 $\alpha_{t,i}$，而 $\alpha_{t,i}$ 又是 $e_{t,i}$ 的 softmax，所以梯度要從 $c_t$ 往回穿過 softmax。

**第一步**：從 $c_t$ 傳到 $\alpha_{t,i}$：

$$\frac{\partial L}{\partial \alpha_{t,i}} = g_t^{(c)} \cdot h_i^{\text{enc}}$$

**第二步**：從 $\alpha_{t,i}$ 穿過 softmax 傳到 $e_{t,i}$（softmax 反傳）：

Softmax 的導數利用以下性質：

$$\frac{\partial \alpha_{t,j}}{\partial e_{t,i}} = \alpha_{t,j}(\delta_{ij} - \alpha_{t,i})$$

（$\delta_{ij}$ 在 $i=j$ 時為 1，否則為 0）

定義 $\delta_{t,i}^{(e)} = \frac{\partial L}{\partial e_{t,i}}$，則：

$$\delta_{t,i}^{(e)} = \alpha_{t,i} \left( \frac{\partial L}{\partial \alpha_{t,i}} - \sum_{j=1}^{T_x} \alpha_{t,j} \frac{\partial L}{\partial \alpha_{t,j}} \right)$$

這是 softmax 反傳的標準公式。

### 9.5 Additive Attention 內部梯度

Attention score 定義為：

$$e_{t,i} = v \cdot z_{t,i}$$

其中：

$$z_{t,i} = \tanh(w_a \cdot s_{t-1} + u_a \cdot h_i^{\text{enc}})$$

**對 $v$ 的梯度**（加總所有 $t, i$）：

$$\frac{\partial L}{\partial v} = \sum_{t=1}^{T_y} \sum_{i=1}^{T_x} \delta_{t,i}^{(e)} \cdot z_{t,i}$$

**往回穿過 tanh**（對 $z_{t,i}$ 的梯度，再乘導數）：

$$\delta_{t,i}^{(z)} = \delta_{t,i}^{(e)} \cdot v \cdot (1 - z_{t,i}^2)$$

**對 $w_a$ 的梯度**：

$$\frac{\partial L}{\partial w_a} = \sum_{t=1}^{T_y} \sum_{i=1}^{T_x} \delta_{t,i}^{(z)} \cdot s_{t-1}$$

**對 $u_a$ 的梯度**：

$$\frac{\partial L}{\partial u_a} = \sum_{t=1}^{T_y} \sum_{i=1}^{T_x} \delta_{t,i}^{(z)} \cdot h_i^{\text{enc}}$$

**Attention 回傳給 Encoder 的梯度（第二路）**：

$$\frac{\partial L}{\partial h_i^{\text{enc}}}\bigg|_{\text{score}} = \sum_{t=1}^{T_y} u_a \cdot \delta_{t,i}^{(z)}$$

**Encoder hidden state 的總梯度（兩路合計）**：

$$\boxed{\frac{\partial L}{\partial h_i^{\text{enc}}} = \underbrace{\sum_{t=1}^{T_y} \alpha_{t,i} \cdot g_t^{(c)}}_{\text{來自 context 加權}} + \underbrace{\sum_{t=1}^{T_y} u_a \cdot \delta_{t,i}^{(z)}}_{\text{來自 attention score}}}$$

- 第一項：context vector 的加權路徑（「你貢獻了多少到摘要」）
- 第二項：attention score 的路徑（「你讓 attention 打了多少分」）

**Attention 回傳給 Decoder $s_{t-1}$ 的梯度**（$s_{t-1}$ 也影響 $e_{t,i}$）：

$$g_t^{(\text{attn-back})} = \sum_{i=1}^{T_x} w_a \cdot \delta_{t+1,i}^{(z)}$$

### 9.6 Encoder 的時間反傳（BPTT）

Encoder 的公式為：

$$a_t = w \cdot x_t + u \cdot h_{t-1}^{\text{enc}} + b$$

$$h_t^{\text{enc}} = \tanh(a_t)$$

定義 $\delta_t = \frac{\partial L}{\partial a_t}$。

$h_t^{\text{enc}}$ 的總梯度來自兩部分：
1. Attention 傳來的梯度 $g_t^{\text{enc}} = \frac{\partial L}{\partial h_t^{\text{enc}}}$（已在 9.5 節算出）
2. 下一步 Encoder 遞迴傳來的梯度 $u \cdot \delta_{t+1}$

$$\frac{\partial L}{\partial h_t^{\text{enc}}} = g_t^{\text{enc}} + u \cdot \delta_{t+1}$$

再乘 tanh 導數：

$$\boxed{\delta_t = \left(g_t^{\text{enc}} + u \cdot \delta_{t+1}\right) \times (1 - (h_t^{\text{enc}})^2)}$$

邊界條件：$\delta_{T_x+1} = 0$

**Encoder 參數的梯度**：

$$\frac{\partial L}{\partial w} = \sum_{t=1}^{T_x} \delta_t \cdot x_t$$

$$\frac{\partial L}{\partial u} = \sum_{t=1}^{T_x} \delta_t \cdot h_{t-1}^{\text{enc}}$$

$$\frac{\partial L}{\partial b} = \sum_{t=1}^{T_x} \delta_t$$

---

## 10. 參數更新：梯度下降法

### 10.1 梯度下降更新規則

對任意參數 $\theta$，沿著梯度的反方向更新：

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

其中 $\eta$ 是**學習率**（learning rate），控制每次更新的步幅大小。

更新所有參數：

$$w_o \leftarrow w_o - \eta \cdot \frac{\partial L}{\partial w_o}$$

$$v \leftarrow v - \eta \cdot \frac{\partial L}{\partial v}, \quad w_a \leftarrow w_a - \eta \cdot \frac{\partial L}{\partial w_a}, \quad u_a \leftarrow u_a - \eta \cdot \frac{\partial L}{\partial u_a}$$

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}, \quad u \leftarrow u - \eta \cdot \frac{\partial L}{\partial u}, \quad b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

### 10.2 為什麼 Attention 更有效

| 對比項目 | 傳統 Seq2Seq | Attention Seq2Seq |
|---------|------------|------------------|
| Context | 固定，$c = h_{T_x}^{\text{enc}}$ | 動態，$c_t = \sum \alpha_{t,i} h_i^{\text{enc}}$ |
| 長句資訊 | 容易遺失 | 可直接存取任意位置 |
| 梯度路徑 | 只有一條從 decoder 到 encoder 末端 | 多條，直接回傳到 encoder 各位置 |
| 對齊可視化 | 無 | 可觀察 $\alpha_{t,i}$ 看模型在看哪裡 |

---

## 11. 完整數值模擬

### 設定

| 項目 | 數值 |
|------|------|
| 輸入 | $x_1=0.5$（I），$x_2=1.0$（love），$x_3=-0.5$（cats） |
| Encoder 參數 | $w=0.8$，$u=0.5$，$b=0.1$ |
| Attention 參數 | $w_a=1.0$，$u_a=1.2$，$v=1.0$ |
| Decoder 參數 | $w_s=0.7$，$w_c=0.9$，$b_s=0.0$ |
| 輸出層 | $w_o^{(A)}=1.5$，$w_o^{(B)}=-0.5$，$b_o=0$ |
| 正確答案 | 字 A（$y^{(A)}=1$，$y^{(B)}=0$） |
| 學習率 | $\eta = 0.1$ |

---

### 📊 前向傳播完整計算

**【Encoder】**

| 步驟 | 預激活 $a_t$ | $h_t^{\text{enc}} = \tanh(a_t)$ |
|------|------------|-------------------------------|
| $t=1$（"I"） | $0.8(0.5)+0.5(0)+0.1=0.500$ | $0.462$ |
| $t=2$（"love"） | $0.8(1.0)+0.5(0.462)+0.1=1.131$ | $0.812$ |
| $t=3$（"cats"） | $0.8(-0.5)+0.5(0.812)+0.1=0.106$ | $0.106$ |

**【Attention（$s_0=0$）】**

| 步驟 | 公式 | 數值 |
|------|------|------|
| $e_{1,1}$ | $\tanh(1.2 \times 0.462)=\tanh(0.554)$ | $0.505$ |
| $e_{1,2}$ | $\tanh(1.2 \times 0.812)=\tanh(0.974)$ | $0.751$ |
| $e_{1,3}$ | $\tanh(1.2 \times 0.106)=\tanh(0.127)$ | $0.127$ |
| softmax 總和 | $e^{0.505}+e^{0.751}+e^{0.127}$ | $4.911$ |
| $\alpha_{1,1}$ | $1.657/4.911$ | $0.337$ |
| $\alpha_{1,2}$ | $2.119/4.911$ | $\mathbf{0.431}$ （最大）|
| $\alpha_{1,3}$ | $1.135/4.911$ | $0.231$ |
| $c_1$ | $0.337(0.462)+0.431(0.812)+0.231(0.106)$ | $0.530$ |

**【Decoder】**

| 步驟 | 公式 | 數值 |
|------|------|------|
| $a_1^{\text{dec}}$ | $0.7(0)+0.9(0.530)+0$ | $0.477$ |
| $s_1$ | $\tanh(0.477)$ | $0.444$ |
| $o_1^{(A)}$ | $1.5 \times 0.444$ | $0.666$ |
| $o_1^{(B)}$ | $-0.5 \times 0.444$ | $-0.222$ |
| $\hat{y}_1^{(A)}$ | softmax | $0.708$ |
| $\hat{y}_1^{(B)}$ | softmax | $0.292$ |
| 損失 $L$ | $-\log(0.708)$ | $\mathbf{0.346}$ |

---

### 📊 反向傳播完整計算

**【第 1 步：輸出層梯度】**

$$\delta_1^{o,(A)} = \hat{y}_1^{(A)} - y_1^{(A)} = 0.708 - 1 = -0.292$$

$$\delta_1^{o,(B)} = \hat{y}_1^{(B)} - y_1^{(B)} = 0.292 - 0 = +0.292$$

**【第 2 步：傳到 Decoder hidden state】**

$$g_1^{(s,\text{out})} = w_o^{(A)} \cdot \delta_1^{o,(A)} + w_o^{(B)} \cdot \delta_1^{o,(B)}$$

$$= 1.5 \times (-0.292) + (-0.5) \times 0.292 = -0.438 - 0.146 = -0.584$$

**【第 3 步：穿越 tanh（進入 Decoder BPTT）】**

$$1 - s_1^2 = 1 - 0.444^2 = 1 - 0.197 = 0.803$$

$$\delta_1^{\text{dec}} = (-0.584) \times 0.803 = -0.469$$

**【第 4 步：傳到 context vector $c_1$】**

$$g_1^{(c)} = w_c \cdot \delta_1^{\text{dec}} = 0.9 \times (-0.469) = -0.422$$

**【第 5 步：context 路徑傳到各 Encoder hidden state】**

$$\frac{\partial L}{\partial h_i^{\text{enc}}}\bigg|_{\text{context}} = \alpha_{1,i} \cdot g_1^{(c)}$$

$$\frac{\partial L}{\partial h_1^{\text{enc}}} = 0.337 \times (-0.422) = -0.142$$

$$\frac{\partial L}{\partial h_2^{\text{enc}}} = 0.431 \times (-0.422) = -0.182 \quad (\text{最大})$$

$$\frac{\partial L}{\partial h_3^{\text{enc}}} = 0.231 \times (-0.422) = -0.098$$

**【第 6 步：Encoder 穿越 tanh（BPTT）】**

以 $t=3$ 為例（$\delta_4 = 0$，因為 $t=3$ 是最後一步）：

$$g_3^{\text{enc}} = -0.098 \quad \text{（來自 Attention）}$$

$$\delta_3 = (g_3^{\text{enc}} + u \cdot \delta_4) \times (1 - (h_3^{\text{enc}})^2)$$

$$= (-0.098 + 0.5 \times 0) \times (1 - 0.106^2) = -0.098 \times 0.989 = -0.097$$

以 $t=2$ 為例：

$$g_2^{\text{enc}} = -0.182$$

$$\delta_2 = (-0.182 + 0.5 \times (-0.097)) \times (1 - 0.812^2)$$

$$= (-0.182 - 0.048) \times (1 - 0.660) = (-0.230) \times 0.340 = -0.078$$

**【第 7 步：計算 Encoder 參數梯度並更新】**

$$\frac{\partial L}{\partial w} = \delta_1 \cdot x_1 + \delta_2 \cdot x_2 + \delta_3 \cdot x_3$$

先算 $\delta_1$（類似步驟，此處簡示）：$\delta_1 \approx -0.051$

$$\frac{\partial L}{\partial w} = (-0.051)(0.5) + (-0.078)(1.0) + (-0.097)(-0.5)$$

$$= -0.026 - 0.078 + 0.049 = -0.055$$

**參數更新**（$\eta = 0.1$）：

$$w \leftarrow 0.8 - 0.1 \times (-0.055) = 0.8 + 0.0055 = 0.8055$$

---

### 📊 一輪訓練前後對比

| 參數 | 更新前 | 梯度 | 更新後（$\eta=0.1$） |
|------|--------|------|---------------------|
| $w$（Encoder 輸入權重） | $0.800$ | $-0.055$ | $0.806$ |
| $w_o^{(A)}$（輸出層 A） | $1.500$ | $\delta_1^{o,(A)} \cdot s_1 = -0.130$ | $1.513$ |
| $w_o^{(B)}$（輸出層 B） | $-0.500$ | $\delta_1^{o,(B)} \cdot s_1 = +0.130$ | $-0.513$ |

> 📌 字 A 的輸出權重增大，字 B 的輸出權重減小，模型更傾向預測字 A。這正是「學習」的本質！

---

## 12. 完整流程整理

### 12.1 前向傳播

**Encoder：**

$$a_t = w \cdot x_t + u \cdot h_{t-1}^{\text{enc}} + b$$

$$h_t^{\text{enc}} = \tanh(a_t), \quad t = 1, \ldots, T_x$$

**Attention：**

$$e_{t,i} = v \cdot \tanh(w_a \cdot s_{t-1} + u_a \cdot h_i^{\text{enc}})$$

$$\alpha_{t,i} = \frac{e^{e_{t,i}}}{\displaystyle\sum_{j=1}^{T_x} e^{e_{t,j}}}$$

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i^{\text{enc}}$$

**Decoder：**

$$a_t^{\text{dec}} = w_s \cdot s_{t-1} + w_c \cdot c_t + b_s$$

$$s_t = \tanh(a_t^{\text{dec}})$$

$$o_t^{(k)} = w_o^{(k)} \cdot s_t + b_o$$

$$\hat{y}_t^{(k)} = \text{softmax}(o_t^{(k)})$$

**損失：**

$$L = -\sum_{t=1}^{T_y} \log \hat{y}_t^{(\text{正確答案})}$$

### 12.2 反向傳播

**輸出層：**

$$\delta_t^o = \hat{y}_t - y_t$$

**Decoder BPTT：**

$$\delta_t^{\text{dec}} = \left(g_t^{(s,\text{out})} + w_s \cdot \delta_{t+1}^{\text{dec}} + g_t^{(\text{attn-back})}\right) \times (1 - s_t^2)$$

**Attention 對 Encoder 的總梯度：**

$$\frac{\partial L}{\partial h_i^{\text{enc}}} = \sum_{t=1}^{T_y} \alpha_{t,i} \cdot g_t^{(c)} + \sum_{t=1}^{T_y} u_a \cdot \delta_{t,i}^{(z)}$$

**Encoder BPTT：**

$$\delta_t = \left(g_t^{\text{enc}} + u \cdot \delta_{t+1}\right) \times (1 - (h_t^{\text{enc}})^2)$$

### 12.3 梯度流動示意

```
損失 L
  │
  ▼ δᵒ = ŷ - y（softmax + cross-entropy 的漂亮結果）
輸出層
  │
  ▼ 乘 wₒ
Decoder hidden state sₜ ◄────── 下一步 Decoder 的遞迴梯度
  │                     ◄────── Attention 傳回的梯度
  ▼ 乘 (1 - sₜ²)
Decoder 預激活 aₜᵈᵉᶜ
  │
  ▼ 對 wc 求導
Context vector cₜ
  │
  ▼ 按 αᵢ 分配（+ attention score 第二路）
Encoder hidden state hᵢᵉⁿᶜ（每個位置都能直接收到梯度！）
  │
  ▼ 乘 (1 - hᵢ²)
Encoder 預激活 aᵢ
  │
  ▼ 對 w, u, b 求導
Encoder 參數更新
```

### 12.4 五個核心公式

$$\boxed{
\begin{aligned}
&\textbf{(1) 對齊分數：} && e_{t,i} = v \cdot \tanh(w_a \cdot s_{t-1} + u_a \cdot h_i^{\text{enc}}) \\
&\textbf{(2) 注意力權重：} && \alpha_{t,i} = \text{softmax}(e_{t,i}) \\
&\textbf{(3) 動態 context：} && c_t = \sum_{i} \alpha_{t,i} \cdot h_i^{\text{enc}} \\
&\textbf{(4) 輸出層梯度：} && \delta_t^o = \hat{y}_t - y_t \\
&\textbf{(5) 穿越 tanh：} && \delta_{\text{前層}} = \delta_{\text{後層}} \times (1 - h^2)
\end{aligned}
}$$

---

## 延伸學習路線

理解本文之後，可以繼續學習：

1. **GRU / LSTM**：更強大的 RNN 單元，用門控機制解決梯度消失問題
2. **Self-Attention**：輸入序列自己對自己做 Attention（Transformer 的核心）
3. **Multi-Head Attention**：同時用多組 Attention，捕捉不同面向的關係
4. **Transformer**：完全用 Attention 取代 RNN，是 GPT、BERT 的基礎

---

*數學工具：tanh 函數 · 鏈鎖律（Chain Rule）· softmax · cross-entropy · 加權平均 · 梯度下降*
