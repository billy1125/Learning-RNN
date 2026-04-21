# RNN 的數學推導

> 適合大學一年級程度，從零開始理解 **Recurrent Neural Network（RNN）** 的前向傳播、損失函數與時間反向傳播（BPTT）

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [RNN 的基本數學模型](#2-rnn-的基本數學模型)
3. [前向傳播（Forward Pass）](#3-前向傳播forward-pass)
4. [損失函數（Cross-Entropy）](#4-損失函數cross-entropy)
5. [反向傳播（BPTT）](#5-反向傳播bptt)
6. [參數梯度整理](#6-參數梯度整理)
7. [梯度問題（Vanishing / Exploding）](#7-梯度問題vanishing--exploding)
8. [完整流程整理](#8-完整流程整理)

---

## 1. 基本符號定義

### 1.1 序列表示

RNN 的輸入是一個有**順序**的序列，例如一句話、一段時間的股價。給定長度為 $T$ 的輸入序列與對應標籤：

$$x = (x_1, x_2, \ldots, x_T), \qquad y = (y_1, y_2, \ldots, y_T)$$

### 1.2 符號表

| 符號 | 意義 |
|------|------|
| $x_t$ | 第 $t$ 步的輸入向量 |
| $h_t$ | 第 $t$ 步的隱藏狀態（hidden state），代表「記憶」 |
| $a_t$ | 隱藏層激活前的線性組合（pre-activation） |
| $o_t$ | 輸出層的線性組合（logits） |
| $\hat{y}_t$ | 第 $t$ 步的預測機率分佈 |
| $W_x$ | 輸入層 → 隱藏層的權重矩陣 |
| $W_h$ | 隱藏層 → 隱藏層的權重矩陣（循環連接） |
| $W_y$ | 隱藏層 → 輸出層的權重矩陣 |
| $b, c$ | 隱藏層與輸出層的偏差向量 |
| $\phi$ | 隱藏層激活函數（通常為 $\tanh$） |
| $L_t$ | 第 $t$ 步的損失 |
| $L$ | 序列總損失 |

---

## 2. RNN 的基本數學模型

### 2.1 核心思想：帶有記憶的神經元

普通神經網路每次計算都是獨立的，不記得之前看過什麼。RNN 的改進在於引入**隱藏狀態 $h_t$**，每一步都把「前一刻的記憶 $h_{t-1}$」和「當前輸入 $x_t$」合在一起計算：

$$h_t = f(x_t,\ h_{t-1})$$

這使得 $h_t$ 中隱含了從 $x_1$ 到 $x_t$ 的所有歷史資訊。

### 2.2 單一時間步的計算

每一步做兩件事：

**步驟一：線性組合**

$$a_t = W_x x_t + W_h h_{t-1} + b$$

- $W_x x_t$：處理當前輸入
- $W_h h_{t-1}$：引入上一步的記憶
- $b$：偏差項

**步驟二：非線性激活**

$$h_t = \phi(a_t)$$

通常使用 $\tanh$，讓輸出壓縮在 $(-1, 1)$ 之間。

### 2.3 維度檢查

設輸入維度 $d_x$、隱藏層維度 $d_h$：

| 矩陣 | 維度 | 說明 |
|------|------|------|
| $W_x$ | $d_h \times d_x$ | 輸入 → 隱藏 |
| $W_h$ | $d_h \times d_h$ | 隱藏 → 隱藏 |
| $W_y$ | $d_y \times d_h$ | 隱藏 → 輸出 |

計算 $a_t = W_x x_t + W_h h_{t-1} + b$ 的維度：

$$\underbrace{(d_h \times d_x)}_{W_x} \underbrace{(d_x \times 1)}_{x_t} + \underbrace{(d_h \times d_h)}_{W_h} \underbrace{(d_h \times 1)}_{h_{t-1}} + \underbrace{(d_h \times 1)}_{b} = \underbrace{(d_h \times 1)}_{a_t} \checkmark$$

---

## 3. 前向傳播（Forward Pass）

### 3.1 完整計算流程

初始條件：$h_0 = \mathbf{0}$（序列開始前沒有記憶）

對每一個時間步 $t = 1, 2, \ldots, T$，依序計算：

$$a_t = W_x x_t + W_h h_{t-1} + b \tag{隱藏層線性組合}$$

$$h_t = \phi(a_t) \tag{隱藏狀態}$$

$$o_t = W_y h_t + c \tag{輸出層線性組合}$$

$$\hat{y}_t = \text{softmax}(o_t) \tag{輸出機率}$$

### 3.2 Softmax 函數

Softmax 把 $o_t$ 中的每個數字轉換成機率，確保所有類別的機率加總為 $1$：

$$\hat{y}_{t,i} = \frac{e^{o_{t,i}}}{\displaystyle\sum_j e^{o_{t,j}}}$$

### 3.3 展開的計算圖

把 RNN 按時間展開，就像一個很深的神經網路，**每一層共用同一組參數** $W_x, W_h, W_y$：

```
x₁       x₂       x₃       ···   xT
 │        │        │               │
 ▼        ▼        ▼               ▼
h₀ ──→ [a₁,h₁] ──→ [a₂,h₂] ──→ [a₃,h₃] ──→ ··· ──→ [aT,hT]
          │           │            │                     │
          ▼           ▼            ▼                     ▼
         ŷ₁          ŷ₂           ŷ₃                   ŷT
          │           │            │                     │
         L₁          L₂           L₃                   LT
```

---

## 4. 損失函數（Cross-Entropy）

### 4.1 單步損失

對於分類問題，第 $t$ 步使用**交叉熵損失**：

$$L_t = -\sum_i y_{t,i} \log \hat{y}_{t,i}$$

其中 $y_{t,i}$ 是 one-hot 編碼（正確類別為 $1$，其餘為 $0$）。

### 4.2 序列總損失

把每一步的損失加起來：

$$L = \sum_{t=1}^{T} L_t$$

### 4.3 Softmax + Cross-Entropy 的導數

這個組合有一個非常乾淨的結果，對輸出 $o_t$ 的梯度為：

$$\frac{\partial L_t}{\partial o_t} = \hat{y}_t - y_t$$

直覺理解：預測機率比真實標籤**高**，就往下調；比真實標籤**低**，就往上調。

---

## 5. 反向傳播（BPTT）

BPTT（Backpropagation Through Time）是把時間展開後的網路，從 $t = T$ 往回到 $t = 1$ 應用鏈鎖律。

### 5.1 定義三個誤差項

為了讓推導整齊，定義每一層的「誤差訊號」：

$$\delta_t^o := \frac{\partial L}{\partial o_t}, \qquad \delta_t^h := \frac{\partial L}{\partial h_t}, \qquad \delta_t^a := \frac{\partial L}{\partial a_t}$$

其中 $\delta_t^a$ 是最核心的一項，因為所有參數 $W_x, W_h, b$ 都直接影響 $a_t$，所以只要算出 $\delta_t^a$，三個參數的梯度便可以一次得到。

### 5.2 輸出層梯度

由 Softmax + Cross-Entropy 的結果直接得到：

$$\boxed{\delta_t^o = \hat{y}_t - y_t}$$

對 $W_y$ 的梯度（由鏈鎖律，$\frac{\partial o_t}{\partial W_y} = h_t$）：

$$\frac{\partial L_t}{\partial W_y} = \delta_t^o \cdot h_t^\top$$

對整個序列求和：

$$\boxed{\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} \delta_t^o h_t^\top}$$

$$\boxed{\frac{\partial L}{\partial c} = \sum_{t=1}^{T} \delta_t^o}$$

### 5.3 隱藏層梯度：兩條影響路徑

這是 RNN 與普通神經網路最不同的地方。在時間步 $t$，$h_t$ 影響損失有**兩條路徑**：

**路徑 A：影響當前輸出**

$$h_t \rightarrow o_t \rightarrow L_t$$

因為 $o_t = W_y h_t + c$，所以：

$$\delta_t^h \Big|_{\text{當前}} = \frac{\partial L_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial h_t} = W_y^\top \delta_t^o$$

**路徑 B：影響未來時間步**

$$h_t \rightarrow a_{t+1} \rightarrow h_{t+1} \rightarrow \cdots \rightarrow L$$

因為 $a_{t+1} = W_h h_t + W_x x_{t+1} + b$，所以：

$$\delta_t^h \Big|_{\text{未來}} = \frac{\partial L}{\partial a_{t+1}} \cdot \frac{\partial a_{t+1}}{\partial h_t} = W_h^\top \delta_{t+1}^a$$

**合併兩條路徑（相加）：**

$$\delta_t^h = W_y^\top \delta_t^o + W_h^\top \delta_{t+1}^a$$

### 5.4 Pre-activation 的誤差訊號

由 $h_t = \phi(a_t)$，用鏈鎖律：

$$\delta_t^a = \frac{\partial L}{\partial a_t} = \frac{\partial L}{\partial h_t} \odot \frac{\partial h_t}{\partial a_t} = \delta_t^h \odot \phi'(a_t)$$

代入 $\delta_t^h$：

$$\boxed{\delta_t^a = \left( W_y^\top \delta_t^o + W_h^\top \delta_{t+1}^a \right) \odot \phi'(a_t)}$$

這就是 BPTT 的**核心遞迴公式**，從 $t = T$ 往回計算到 $t = 1$。

### 5.5 以 tanh 為例

若 $\phi = \tanh$，則 $\phi'(a_t) = 1 - \tanh^2(a_t) = 1 - h_t^2$，代入得：

$$\boxed{\delta_t^a = \left( W_y^\top \delta_t^o + W_h^\top \delta_{t+1}^a \right) \odot (1 - h_t^2)}$$

### 5.6 邊界條件

序列在 $t = T$ 結束，沒有「下一步」，因此：

$$\delta_{T+1}^a = \mathbf{0}$$

所以最後一步退化為：

$$\delta_T^a = \left( W_y^\top \delta_T^o \right) \odot \phi'(a_T)$$

### 5.7 反向傳播流程圖

```
t=T                t=T-1              t=1
δᵀₒ → δᵀₐ ──→ δᵀ⁻¹ₒ → δᵀ⁻¹ₐ ──→ ··· → δ¹ₐ
       ↓  W_h^T         ↓  W_h^T             ↓
      ∂L/∂W_h         ∂L/∂W_h            ∂L/∂W_h
      ∂L/∂W_x         ∂L/∂W_x            ∂L/∂W_x
```

每一步算出 $\delta_t^a$ 後，就能立刻累積該步對參數的梯度貢獻。

---

## 6. 參數梯度整理

有了每一步的 $\delta_t^a$，所有參數的梯度都能統一用同一套公式得出：

由 $a_t = W_x x_t + W_h h_{t-1} + b$，分別對三個參數求偏導並對序列求和：

**對隱藏-隱藏權重 $W_h$：**

$$\boxed{\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \delta_t^a \, h_{t-1}^\top}$$

**對輸入-隱藏權重 $W_x$：**

$$\boxed{\frac{\partial L}{\partial W_x} = \sum_{t=1}^{T} \delta_t^a \, x_t^\top}$$

**對隱藏層偏差 $b$：**

$$\boxed{\frac{\partial L}{\partial b} = \sum_{t=1}^{T} \delta_t^a}$$

> **規律：** 三個公式的結構完全一樣——都是「誤差訊號 × 對應輸入的轉置」，這正是因為把誤差統一定義在 $a_t$ 上的好處。

---

## 7. 梯度問題（Vanishing / Exploding）

### 7.1 問題根源

在 BPTT 中，梯度從 $t = T$ 反傳到 $t = 1$ 時，每經過一步都要乘上一次 $W_h^\top$。若序列長度為 $T$，大略來說梯度的大小正比於：

$$\delta_1^h \sim \left( W_h^\top \right)^T \delta_T^h$$

### 7.2 梯度消失（Vanishing Gradient）

若 $W_h$ 的最大特徵值 $< 1$：

$$\left\| W_h^T \right\|^k \rightarrow 0 \quad \text{as } k \rightarrow \infty$$

→ 梯度指數衰減，距離輸出較遠的時間步幾乎無法學習，**RNN 失去長期記憶能力**。

### 7.3 梯度爆炸（Exploding Gradient）

若 $W_h$ 的最大特徵值 $> 1$：

$$\left\| W_h^T \right\|^k \rightarrow \infty \quad \text{as } k \rightarrow \infty$$

→ 梯度指數增大，訓練不穩定，參數更新幅度過大，損失可能發散。

### 7.4 實務解法

| 問題 | 解法 |
|------|------|
| 梯度爆炸 | **梯度裁剪（Gradient Clipping）**：若梯度範數超過閾值就等比例縮小 |
| 梯度消失 | 使用 **LSTM / GRU**：透過「門控機制」讓梯度有專屬的通道直接流通，不被反覆壓縮 |
| 兩者 | **Truncated BPTT**：只往回傳 $K$ 步，犧牲極長期依賴以換取訓練穩定性 |

**Truncated BPTT 的邊界設定：**

$$\delta_t^a \approx 0 \quad \text{for } t < T - K$$

---

## 8. 完整流程整理

### 前向傳播（$t = 1 \to T$）

$$h_0 = \mathbf{0}$$

$$a_t = W_x x_t + W_h h_{t-1} + b$$

$$h_t = \tanh(a_t)$$

$$o_t = W_y h_t + c$$

$$\hat{y}_t = \text{softmax}(o_t)$$

$$L = \sum_{t=1}^{T} L_t = -\sum_{t=1}^{T} \sum_i y_{t,i} \log \hat{y}_{t,i}$$

### 反向傳播（$t = T \to 1$）

初始條件：$\delta_{T+1}^a = \mathbf{0}$

$$\delta_t^o = \hat{y}_t - y_t$$

$$\delta_t^a = \left( W_y^\top \delta_t^o + W_h^\top \delta_{t+1}^a \right) \odot (1 - h_t^2)$$

### 參數梯度（累積所有時間步）

$$\frac{\partial L}{\partial W_y} = \sum_t \delta_t^o h_t^\top, \qquad \frac{\partial L}{\partial c} = \sum_t \delta_t^o$$

$$\frac{\partial L}{\partial W_h} = \sum_t \delta_t^a h_{t-1}^\top, \qquad \frac{\partial L}{\partial W_x} = \sum_t \delta_t^a x_t^\top, \qquad \frac{\partial L}{\partial b} = \sum_t \delta_t^a$$

### 參數更新

$$\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}, \qquad \theta \in \{W_x,\, W_h,\, W_y,\, b,\, c\}$$

---

## 總結

RNN 的三個核心設計思想：

1. **時間上的參數共享**：$W_x, W_h, W_y$ 在每個時間步都一樣，大幅減少參數量
2. **Hidden state 作為記憶**：$h_t$ 把過去的資訊濃縮成一個向量，傳給下一步
3. **梯度沿時間反傳（BPTT）**：利用鏈鎖律，誤差從 $t=T$ 一路往回傳遞，每步都修正對應的參數

BPTT 的核心遞迴公式（以 tanh 為例）：

$$\boxed{\delta_t^a = \left( W_y^\top \delta_t^o + W_h^\top \delta_{t+1}^a \right) \odot (1 - h_t^2)}$$

理解這一條公式，就理解了 RNN 的學習原理。

---

*延伸閱讀：LSTM · GRU · Transformer（以注意力機制取代循環結構）*