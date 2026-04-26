# LSTM 與 GRU 的數學推導

> 適合理解 RNN 基礎後進一步學習，從零開始推導 **Long Short-Term Memory（LSTM）** 與 **Gated Recurrent Unit（GRU）** 的前向傳播與時間反向傳播（BPTT）

---

## 目錄

**LSTM**

1. [基本符號定義](#1-基本符號定義)
2. [LSTM 的前向傳播](#2-lstm-的前向傳播forward-pass)
3. [損失函數](#3-損失函數)
4. [反向傳播（BPTT）：從輸出層開始](#4-反向傳播bptt從輸出層開始)
5. [反向傳播：Cell State 的兩條路徑](#5-反向傳播cell-state-的兩條路徑)
6. [反向傳播：分流到各 Gate](#6-反向傳播分流到各-gate)
7. [反向傳播：回到 Pre-activation](#7-反向傳播回到-pre-activation)
8. [參數梯度整理](#8-參數梯度整理)
9. [完整 BPTT 迴圈整理](#9-完整-bptt-迴圈整理)

**GRU**

10. [GRU 簡介與核心概念](#10-gru-簡介與核心概念)
11. [GRU 的前向傳播](#11-gru-的前向傳播)
12. [GRU 的反向傳播](#12-gru-的反向傳播)
13. [GRU 參數梯度整理](#13-gru-參數梯度整理)
14. [GRU 完整反向傳播流程](#14-gru-完整反向傳播流程)

---

## 1. 基本符號定義

### 1.1 為什麼需要 LSTM？

RNN 雖然引入了隱藏狀態 $h_t$ 作為「記憶」，但因梯度在反向傳播時每步都要乘上 $W_h^\top$，長序列下梯度會指數消失，導致 RNN 難以學到距離較遠的依賴關係。

LSTM 的解法是引入第二條記憶通道——**cell state $C_t$**。透過三個「閘門（gate）」精確控制資訊的保留、寫入與讀出，讓梯度有一條更直接的傳遞路徑。

![lstm](../image/lstm.webp)

![lstm in time series](../image/lstm-time-series.webp)

圖片來源： https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/

### 1.2 符號表

在時間步 $t$，LSTM 使用兩個狀態向量：

| 符號 | 意義 |
|------|------|
| $x_t$ | 第 $t$ 步的輸入向量 |
| $h_t$ | Hidden state（對外輸出的記憶） |
| $C_t$ | Cell state（內部長期記憶） |
| $f_t$ | Forget gate（決定遺忘多少舊記憶） |
| $i_t$ | Input gate（決定寫入多少新資訊） |
| $o_t$ | Output gate（決定讀出多少記憶） |
| $\tilde{C}_t$ | Candidate cell（本步欲寫入的候選值） |
| $z_t$ | 輸出層 logits |
| $\hat{y}_t$ | Softmax 後的預測機率 |
| $y_t$ | 真實標籤（one-hot） |

### 1.3 參數表（分開權重寫法）

每個 gate 各有一組輸入權重 $W$（處理 $x_t$）與循環權重 $U$（處理 $h_{t-1}$）：

| 參數組 | 作用 |
|--------|------|
| $W_f, U_f, b_f$ | Forget gate |
| $W_i, U_i, b_i$ | Input gate |
| $W_o, U_o, b_o$ | Output gate |
| $W_c, U_c, b_c$ | Candidate cell |
| $W_y, b_y$ | 輸出層 |

---

## 2. LSTM 的前向傳播（Forward Pass）

初始條件：$h_0 = \mathbf{0}$，$C_0 = \mathbf{0}$

對每個時間步 $t = 1, 2, \ldots, T$，依序計算：

### 2.1 三個 Gate 與候選記憶

**Forget gate**——決定要遺忘多少舊 cell state：

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

**Input gate**——決定要寫入多少新資訊：

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

**Output gate**——決定要讀出多少記憶：

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

**Candidate cell**——本步欲寫入的候選值：

$$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

### 2.2 Cell State 更新（核心）

$$\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$

直覺理解：
- $f_t \odot C_{t-1}$：保留舊記憶中值得保留的部分
- $i_t \odot \tilde{C}_t$：把新資訊寫入記憶

### 2.3 Hidden State

$$h_t = o_t \odot \tanh(C_t)$$

先對 $C_t$ 取 $\tanh$ 壓縮到 $(-1, 1)$，再讓 output gate 決定「讀出」多少。

### 2.4 輸出層

$$z_t = W_y h_t + b_y \qquad \hat{y}_t = \text{softmax}(z_t)$$

### 2.5 展開的計算圖

```
x₁        x₂        x₃               xT
 │         │         │                 │
 ▼         ▼         ▼                 ▼
h₀,C₀ → [gates,h₁,C₁] → [gates,h₂,C₂] → ··· → [gates,hT,CT]
              │                │                       │
             ŷ₁               ŷ₂                      ŷT
              │                │                       │
             L₁               L₂                      LT
```

與普通 RNN 相比，LSTM 多了一條沿時間傳遞的 **cell state 高速公路**，這正是梯度能長距離流通的關鍵。

---

## 3. 損失函數

### 3.1 單步損失與序列總損失

第 $t$ 步的交叉熵損失：

$$L_t = -\sum_k y_{t,k} \log \hat{y}_{t,k}$$

序列總損失：

$$L = \sum_{t=1}^{T} L_t$$

### 3.2 Softmax + Cross-Entropy 的導數

與 RNN 相同，這個組合有非常乾淨的梯度：

$$\frac{\partial L_t}{\partial z_t} = \hat{y}_t - y_t$$

定義 $\delta^y_t \equiv \hat{y}_t - y_t$，後續推導中會反覆用到。

---

## 4. 反向傳播（BPTT）：從輸出層開始

### 4.1 輸出層參數梯度

由 $z_t = W_y h_t + b_y$，利用鏈鎖律：

$$\boxed{\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} \delta^y_t h_t^\top}, \qquad \boxed{\frac{\partial L}{\partial b_y} = \sum_{t=1}^{T} \delta^y_t}$$

### 4.2 Hidden State 的總梯度：兩條影響路徑

$h_t$ 影響損失的路徑有兩條——這與 RNN 的情形完全相同：

**路徑 A：影響當前輸出**

$$\left.\frac{\partial L}{\partial h_t}\right|_{\text{當前}} = W_y^\top \delta^y_t$$

**路徑 B：影響未來時間步**（透過 $h_{t+1}$ 的各個 gate 回傳）

$$\left.\frac{\partial L}{\partial h_t}\right|_{\text{未來}} = U_f^\top \delta^{a_f}_{t+1} + U_i^\top \delta^{a_i}_{t+1} + U_o^\top \delta^{a_o}_{t+1} + U_c^\top \delta^{a_c}_{t+1}$$

定義 $\delta^{h,\text{future}}_t$ 為從未來時間步反傳回來的部分，則：

$$\boxed{\delta^h_t = W_y^\top \delta^y_t + \delta^{h,\text{future}}_t}$$

> **與 RNN 的差異：** 在 RNN 中，$h_t$ 只透過單一矩陣 $W_h$ 影響下一步。LSTM 中，$h_t$ 同時進入四個 gate 的 pre-activation，因此未來路徑是四項之和。

---

## 5. 反向傳播：Cell State 的兩條路徑

LSTM 最關鍵的地方在於，除了 $h_t$ 之外，**cell state $C_t$ 也是梯度的載體**，且它有自己的兩條路徑。

定義 $\delta^c_t \equiv \dfrac{\partial L}{\partial C_t}$。

**路徑 A：經由 $h_t$**

由 $h_t = o_t \odot \tanh(C_t)$：

$$\frac{\partial h_t}{\partial C_t} = o_t \odot \bigl(1 - \tanh^2(C_t)\bigr)$$

所以從 $h_t$ 回傳到 $C_t$ 的梯度為：

$$\left.\frac{\partial L}{\partial C_t}\right|_{\text{via } h_t} = \delta^h_t \odot o_t \odot \bigl(1 - \tanh^2(C_t)\bigr)$$

**路徑 B：直接沿 cell state 從未來傳回**

由 $C_{t+1} = f_{t+1} \odot C_t + \cdots$，所以：

$$\left.\frac{\partial L}{\partial C_t}\right|_{\text{未來}} = \delta^c_{t+1} \odot f_{t+1}$$

**合併兩條路徑：**

$$\boxed{\delta^c_t = \delta^h_t \odot o_t \odot \bigl(1 - \tanh^2(C_t)\bigr) + \delta^c_{t+1} \odot f_{t+1}}$$

這是 LSTM BPTT 最核心的遞迴公式之一。第二項 $\delta^c_{t+1} \odot f_{t+1}$ 讓梯度能**沿 cell state 長距離傳遞而不被反覆壓縮**，這正是 LSTM 比普通 RNN 更不易梯度消失的根本原因。

---

## 6. 反向傳播：分流到各 Gate

有了 $\delta^c_t$，就能利用 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ 對每個分支求梯度：

**Output gate 的梯度**（來自 $h_t = o_t \odot \tanh(C_t)$）：

$$\delta^o_t \equiv \frac{\partial L}{\partial o_t} = \delta^h_t \odot \tanh(C_t)$$

**Forget gate 的梯度**：

$$\delta^f_t \equiv \frac{\partial L}{\partial f_t} = \delta^c_t \odot C_{t-1}$$

**Input gate 的梯度**：

$$\delta^i_t \equiv \frac{\partial L}{\partial i_t} = \delta^c_t \odot \tilde{C}_t$$

**Candidate cell 的梯度**：

$$\delta^{\tilde{c}}_t \equiv \frac{\partial L}{\partial \tilde{C}_t} = \delta^c_t \odot i_t$$

**傳給前一個 cell state 的梯度**（供下一輪 BPTT 使用）：

$$\delta^c_{t-1} = \delta^c_t \odot f_t$$

---

## 7. 反向傳播：回到 Pre-activation

各 gate 皆有一個 pre-activation（線性組合結果），梯度要再乘上 activation 函數的導數才能繼續往回傳。

定義各 gate 的 pre-activation：

$$a^f_t = W_f x_t + U_f h_{t-1} + b_f, \quad f_t = \sigma(a^f_t)$$
$$a^i_t = W_i x_t + U_i h_{t-1} + b_i, \quad i_t = \sigma(a^i_t)$$
$$a^o_t = W_o x_t + U_o h_{t-1} + b_o, \quad o_t = \sigma(a^o_t)$$
$$a^c_t = W_c x_t + U_c h_{t-1} + b_c, \quad \tilde{C}_t = \tanh(a^c_t)$$

利用 $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ 與 $\tanh'(x) = 1 - \tanh^2(x)$：

$$\boxed{\delta^{a_f}_t = \delta^f_t \odot f_t \odot (1 - f_t)}$$

$$\boxed{\delta^{a_i}_t = \delta^i_t \odot i_t \odot (1 - i_t)}$$

$$\boxed{\delta^{a_o}_t = \delta^o_t \odot o_t \odot (1 - o_t)}$$

$$\boxed{\delta^{a_c}_t = \delta^{\tilde{c}}_t \odot (1 - \tilde{C}_t^2)}$$

> **規律：** 四組公式的結構完全相同——都是「上游梯度 × activation 導數」，這與 RNN 中 $\delta^a_t = \delta^h_t \odot \phi'(a_t)$ 的邏輯如出一轍，只是 LSTM 多了四條並行通道。

---

## 8. 參數梯度整理

有了每一步的 $\delta^{a_f}_t, \delta^{a_i}_t, \delta^{a_o}_t, \delta^{a_c}_t$，所有參數的梯度都能用同一套公式得出，對整個序列求和：

**Forget gate 參數：**

$$\frac{\partial L}{\partial W_f} = \sum_t \delta^{a_f}_t x_t^\top, \quad \frac{\partial L}{\partial U_f} = \sum_t \delta^{a_f}_t h_{t-1}^\top, \quad \frac{\partial L}{\partial b_f} = \sum_t \delta^{a_f}_t$$

**Input gate 參數：**

$$\frac{\partial L}{\partial W_i} = \sum_t \delta^{a_i}_t x_t^\top, \quad \frac{\partial L}{\partial U_i} = \sum_t \delta^{a_i}_t h_{t-1}^\top, \quad \frac{\partial L}{\partial b_i} = \sum_t \delta^{a_i}_t$$

**Output gate 參數：**

$$\frac{\partial L}{\partial W_o} = \sum_t \delta^{a_o}_t x_t^\top, \quad \frac{\partial L}{\partial U_o} = \sum_t \delta^{a_o}_t h_{t-1}^\top, \quad \frac{\partial L}{\partial b_o} = \sum_t \delta^{a_o}_t$$

**Candidate cell 參數：**

$$\frac{\partial L}{\partial W_c} = \sum_t \delta^{a_c}_t x_t^\top, \quad \frac{\partial L}{\partial U_c} = \sum_t \delta^{a_c}_t h_{t-1}^\top, \quad \frac{\partial L}{\partial b_c} = \sum_t \delta^{a_c}_t$$

**輸出層參數：**

$$\frac{\partial L}{\partial W_y} = \sum_t \delta^y_t h_t^\top, \quad \frac{\partial L}{\partial b_y} = \sum_t \delta^y_t$$

**傳給輸入 $x_t$ 的梯度**（若需往更前層反傳）：

$$\frac{\partial L}{\partial x_t} = W_f^\top \delta^{a_f}_t + W_i^\top \delta^{a_i}_t + W_o^\top \delta^{a_o}_t + W_c^\top \delta^{a_c}_t$$

---

## 9. 完整 BPTT 迴圈整理

### 前向傳播（$t = 1 \to T$）

$$h_0 = \mathbf{0}, \quad C_0 = \mathbf{0}$$

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

$$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$h_t = o_t \odot \tanh(C_t)$$

$$z_t = W_y h_t + b_y, \quad \hat{y}_t = \text{softmax}(z_t)$$

$$L = \sum_{t=1}^{T} L_t = -\sum_{t=1}^{T} \sum_k y_{t,k} \log \hat{y}_{t,k}$$

### 反向傳播（$t = T \to 1$）

初始條件：$\delta^{h,\text{future}}_T = \mathbf{0}$，$\delta^c_{T+1} = \mathbf{0}$

**Step 1. 輸出層**

$$\delta^y_t = \hat{y}_t - y_t$$

$$\delta^h_t = W_y^\top \delta^y_t + \delta^{h,\text{future}}_t$$

**Step 2. Hidden → Output gate / Cell**

$$\delta^o_t = \delta^h_t \odot \tanh(C_t)$$

$$\delta^c_t = \delta^h_t \odot o_t \odot \bigl(1 - \tanh^2(C_t)\bigr) + \delta^c_{t+1} \odot f_{t+1}$$

**Step 3. Cell 分流**

$$\delta^f_t = \delta^c_t \odot C_{t-1}, \quad \delta^i_t = \delta^c_t \odot \tilde{C}_t, \quad \delta^{\tilde{c}}_t = \delta^c_t \odot i_t$$

**Step 4. Pre-activation 梯度**

$$\delta^{a_f}_t = \delta^f_t \odot f_t(1-f_t), \quad \delta^{a_i}_t = \delta^i_t \odot i_t(1-i_t)$$

$$\delta^{a_o}_t = \delta^o_t \odot o_t(1-o_t), \quad \delta^{a_c}_t = \delta^{\tilde{c}}_t \odot (1-\tilde{C}_t^2)$$

**Step 5. 累積參數梯度**（各參數加上本步貢獻）

**Step 6. 傳給前一時間步**

$$\delta^{h,\text{future}}_{t-1} = U_f^\top \delta^{a_f}_t + U_i^\top \delta^{a_i}_t + U_o^\top \delta^{a_o}_t + U_c^\top \delta^{a_c}_t$$

$$\delta^c_{t-1} = \delta^c_t \odot f_t$$

### 參數更新

$$\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}, \qquad \theta \in \{W_f, U_f, b_f,\ W_i, U_i, b_i,\ W_o, U_o, b_o,\ W_c, U_c, b_c,\ W_y, b_y\}$$

---

## 10. GRU 簡介與核心概念

### 10.1 GRU 是 LSTM 的簡化版

GRU（Gated Recurrent Unit）保留了 LSTM 的門控機制，但取消了獨立的 cell state $C_t$，直接用 hidden state 表示記憶，並把三個 gate 精簡為兩個：

| 項目 | GRU | LSTM |
|------|-----|------|
| Gate 數量 | 2（Update, Reset） | 3（Forget, Input, Output） |
| Cell state | 無 | 有 |
| 計算量 | 較少 | 較多 |
| 長期依賴能力 | 通常接近 LSTM | 稍強但較重 |

### 10.2 兩個 Gate 的直覺

**Update gate $u_t$**（控制「保留多少舊記憶」）
- 接近 $1$ → 保留舊狀態（長期記憶）
- 接近 $0$ → 使用新資訊

**Reset gate $r_t$**（控制「忽略多少舊記憶來生成候選值」）
- 接近 $0$ → 完全忘記過去，適合新上下文
- 接近 $1$ → 保留過去資訊參與候選計算

---

## 11. GRU 的前向傳播

初始條件：$h_0 = \mathbf{0}$

對每個時間步 $t = 1, 2, \ldots, T$：

### 11.1 Update Gate

$$u_t = \sigma(a^u_t), \qquad a^u_t = x_t W_{xu} + h_{t-1} W_{hu} + b_u$$

### 11.2 Reset Gate

$$r_t = \sigma(a^r_t), \qquad a^r_t = x_t W_{xr} + h_{t-1} W_{hr} + b_r$$

### 11.3 Candidate Hidden State

先計算 reset gate 遮罩後的舊記憶：

$$m_t = r_t \odot h_{t-1}$$

再計算候選值：

$$\tilde{h}_t = \tanh(a^h_t), \qquad a^h_t = x_t W_{xh} + m_t W_{hh} + b_h$$

### 11.4 Hidden State 更新

$$\boxed{h_t = u_t \odot h_{t-1} + (1 - u_t) \odot \tilde{h}_t}$$

在舊記憶 $h_{t-1}$ 與新候選 $\tilde{h}_t$ 之間做加權混合。

### 11.5 輸出層

由 hidden state $h_t$ 計算 logits，再經 softmax 得到預測機率：

$$z_t = h_t W_y + b_y \qquad \hat{y}_t = \text{softmax}(z_t)$$

其中 $W_y \in \mathbb{R}^{d_h \times V}$，$b_y \in \mathbb{R}^V$，$V$ 為詞彙表大小（或輸出維度）。

第 $t$ 步的交叉熵損失：

$$L_t = -\sum_k y_{t,k} \log \hat{y}_{t,k}$$

序列總損失：

$$L = \sum_{t=1}^{T} L_t$$

利用 Softmax + Cross-Entropy 的組合導數，輸出層的誤差訊號有非常簡潔的形式：

$$\boxed{\delta^y_t \equiv \frac{\partial L_t}{\partial z_t} = \hat{y}_t - y_t}$$

輸出層參數梯度對整個序列求和：

$$\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} h_t^\top \delta^y_t, \qquad \frac{\partial L}{\partial b_y} = \sum_{t=1}^{T} \delta^y_t$$

$\delta^y_t$ 也是反向傳播的起點——梯度將從此處往 $h_t$ 回傳，進入 GRU 各 gate 的反向鏈路。

---

## 12. GRU 的反向傳播

### 12.0 輸出層的向後傳播

反向傳播從輸出層開始，沿時間 $t = T \to 1$ 逐步回傳。

**Step A：輸出誤差訊號**

由 Softmax + Cross-Entropy 的組合導數：

$$\delta^y_t = \hat{y}_t - y_t$$

**Step B：輸出層參數梯度**

由 $z_t = h_t W_y + b_y$，利用鏈鎖律對整個序列累加：

$$\boxed{\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} h_t^\top \delta^y_t}, \qquad \boxed{\frac{\partial L}{\partial b_y} = \sum_{t=1}^{T} \delta^y_t}$$

**Step C：梯度傳回 $h_t$**

$h_t$ 影響損失的路徑有兩條：

- **路徑 A（當前輸出）**：$h_t \to z_t \to L_t$

$$\left.\frac{\partial L}{\partial h_t}\right|_{\text{當前}} = \delta^y_t W_y^\top$$

- **路徑 B（未來時間步）**：$h_t$ 作為 $h_{t+1}$ 計算中的 $h_{t-1}$，透過 update gate、reset gate 與 candidate state 傳回梯度（即下方 12.6 的 $\dfrac{\partial L}{\partial h_{t-1}}$，在時間步 $t+1$ 時計算）

$$\left.\frac{\partial L}{\partial h_t}\right|_{\text{未來}} = \delta^{h,\text{future}}_t$$

合併兩條路徑，定義後續各節使用的 $\delta^h_t$：

$$\boxed{\delta^h_t = \delta^y_t W_y^\top + \delta^{h,\text{future}}_t}$$

初始條件（最後一步）：$\delta^{h,\text{future}}_T = \mathbf{0}$。

---

定義 $\delta^h_t \equiv \dfrac{\partial L}{\partial h_t}$（來自當前輸出層與下一時間步兩部分之和）。

### 12.1 從 Hidden State 更新式反傳

由 $h_t = u_t \odot h_{t-1} + (1 - u_t) \odot \tilde{h}_t$，分別對三個分量求梯度：

對 $\tilde{h}_t$：

$$\frac{\partial L}{\partial \tilde{h}_t} = \delta^h_t \odot (1 - u_t)$$

對 $u_t$：

$$\frac{\partial L}{\partial u_t} = \delta^h_t \odot (h_{t-1} - \tilde{h}_t)$$

對 $h_{t-1}$ 的直接貢獻（僅來自 $u_t \odot h_{t-1}$ 這一項）：

$$\left.\frac{\partial L}{\partial h_{t-1}}\right|_{\text{direct}} = \delta^h_t \odot u_t$$

### 12.2 Candidate Hidden State 的反傳

由 $\tilde{h}_t = \tanh(a^h_t)$：

$$\delta^h_{a,t} \equiv \frac{\partial L}{\partial a^h_t} = \left(\delta^h_t \odot (1 - u_t)\right) \odot (1 - \tilde{h}_t^2)$$

### 12.3 從 $a^h_t$ 往 $m_t$ 反傳

由 $a^h_t = x_t W_{xh} + m_t W_{hh} + b_h$：

$$\frac{\partial L}{\partial m_t} = \delta^h_{a,t} W_{hh}^\top$$

又因 $m_t = r_t \odot h_{t-1}$，可得：

對 $r_t$：

$$\frac{\partial L}{\partial r_t} = \left(\delta^h_{a,t} W_{hh}^\top\right) \odot h_{t-1}$$

對 $h_{t-1}$（來自 candidate 路徑）：

$$\left.\frac{\partial L}{\partial h_{t-1}}\right|_{\tilde{h}\text{-path}} = \left(\delta^h_{a,t} W_{hh}^\top\right) \odot r_t$$

### 12.4 Reset Gate 的反傳

由 $r_t = \sigma(a^r_t)$，乘上 sigmoid 導數：

$$\boxed{\delta^r_{a,t} = \left[\left(\delta^h_{a,t} W_{hh}^\top\right) \odot h_{t-1}\right] \odot r_t(1 - r_t)}$$

### 12.5 Update Gate 的反傳

由 $u_t = \sigma(a^u_t)$，乘上 sigmoid 導數：

$$\boxed{\delta^u_{a,t} = \left[\delta^h_t \odot (h_{t-1} - \tilde{h}_t)\right] \odot u_t(1 - u_t)}$$

### 12.6 合併對 $h_{t-1}$ 的總梯度

$h_{t-1}$ 共有四條影響路徑，梯度需全部加總：

$$\boxed{\frac{\partial L}{\partial h_{t-1}} = \underbrace{\delta^h_t \odot u_t}_{\text{直接路徑}} + \underbrace{\left(\delta^h_{a,t} W_{hh}^\top\right) \odot r_t}_{\text{candidate 路徑}} + \underbrace{\delta^r_{a,t} W_{hr}^\top}_{\text{reset gate 路徑}} + \underbrace{\delta^u_{a,t} W_{hu}^\top}_{\text{update gate 路徑}}}$$

這就是 BPTT 時傳給前一時間步的 $\delta^{h,\text{future}}_{t-1}$。

---

## 13. GRU 參數梯度整理

**輸出層參數：**

$$\frac{\partial L}{\partial W_y} = \sum_t h_t^\top \delta^y_t, \quad \frac{\partial L}{\partial b_y} = \sum_t \delta^y_t$$

**Candidate hidden state 參數：**

$$\frac{\partial L}{\partial W_{xh}} = \sum_t x_t^\top \delta^h_{a,t}, \quad \frac{\partial L}{\partial W_{hh}} = \sum_t m_t^\top \delta^h_{a,t}, \quad \frac{\partial L}{\partial b_h} = \sum_t \delta^h_{a,t}$$

**Reset gate 參數：**

$$\frac{\partial L}{\partial W_{xr}} = \sum_t x_t^\top \delta^r_{a,t}, \quad \frac{\partial L}{\partial W_{hr}} = \sum_t h_{t-1}^\top \delta^r_{a,t}, \quad \frac{\partial L}{\partial b_r} = \sum_t \delta^r_{a,t}$$

**Update gate 參數：**

$$\frac{\partial L}{\partial W_{xu}} = \sum_t x_t^\top \delta^u_{a,t}, \quad \frac{\partial L}{\partial W_{hu}} = \sum_t h_{t-1}^\top \delta^u_{a,t}, \quad \frac{\partial L}{\partial b_u} = \sum_t \delta^u_{a,t}$$

**對輸入 $x_t$ 的梯度：**

$$\frac{\partial L}{\partial x_t} = \delta^u_{a,t} W_{xu}^\top + \delta^r_{a,t} W_{xr}^\top + \delta^h_{a,t} W_{xh}^\top$$

---

## 14. GRU 完整反向傳播流程

對整個序列，從 $t = T$ 往回迴圈，初始條件 $\delta^{h,\text{future}}_T = \mathbf{0}$，單一時間步的 backward 分六步：

**Step 0.** 輸出層反傳，計算本步 $\delta^h_t$

$$\delta^y_t = \hat{y}_t - y_t, \qquad \delta^h_t = \delta^y_t W_y^\top + \delta^{h,\text{future}}_t$$

**Step 1.** 從 $h_t$ 更新式分流

$$\delta^{\tilde{h}}_t = \delta^h_t \odot (1 - u_t), \qquad \delta^u_t = \delta^h_t \odot (h_{t-1} - \tilde{h}_t)$$

$$\delta^{h \to h_{t-1}}_{\text{direct}} = \delta^h_t \odot u_t$$

**Step 2.** Candidate 的 pre-activation 梯度

$$\delta^h_{a,t} = \delta^{\tilde{h}}_t \odot (1 - \tilde{h}_t^2)$$

**Step 3.** 透過 $W_{hh}$ 往 $m_t$ 反傳

$$\frac{\partial L}{\partial m_t} = \delta^h_{a,t} W_{hh}^\top, \qquad \delta^r_t = \frac{\partial L}{\partial m_t} \odot h_{t-1}, \qquad \delta^{h \to h_{t-1}}_{\text{cand}} = \frac{\partial L}{\partial m_t} \odot r_t$$

**Step 4.** Gate 的 pre-activation 梯度

$$\delta^r_{a,t} = \delta^r_t \odot r_t(1 - r_t), \qquad \delta^u_{a,t} = \delta^u_t \odot u_t(1 - u_t)$$

**Step 5.** 合併對 $h_{t-1}$ 的總梯度，作為下一步的 $\delta^{h,\text{future}}_{t-1}$

$$\frac{\partial L}{\partial h_{t-1}} = \delta^{h \to h_{t-1}}_{\text{direct}} + \delta^{h \to h_{t-1}}_{\text{cand}} + \delta^r_{a,t} W_{hr}^\top + \delta^u_{a,t} W_{hu}^\top$$

**（各步同時累積）** 輸出層與各 gate 的參數梯度

$$\frac{\partial L}{\partial W_y} \mathrel{+}= h_t^\top \delta^y_t, \quad \frac{\partial L}{\partial b_y} \mathrel{+}= \delta^y_t$$

$$\frac{\partial L}{\partial W_{xh}} \mathrel{+}= x_t^\top \delta^h_{a,t}, \quad \frac{\partial L}{\partial W_{hh}} \mathrel{+}= m_t^\top \delta^h_{a,t}, \quad \cdots \text{（其餘 gate 參數類同）}$$

---

## 總結

### LSTM 的三個核心設計思想

1. **雙狀態架構**：$h_t$ 負責對外輸出，$C_t$ 負責長期記憶，分工明確
2. **門控機制**：三個 gate 精確控制「遺忘、寫入、讀出」，讓網路自己學習何時更新記憶
3. **Cell state 高速公路**：梯度可沿 $C_t$ 長距離流通，緩解 RNN 的梯度消失問題

LSTM BPTT 最核心的兩條遞迴公式：

$$\boxed{\delta^h_t = W_y^\top \delta^y_t + U_f^\top \delta^{a_f}_{t+1} + U_i^\top \delta^{a_i}_{t+1} + U_o^\top \delta^{a_o}_{t+1} + U_c^\top \delta^{a_c}_{t+1}}$$

$$\boxed{\delta^c_t = \delta^h_t \odot o_t \odot \bigl(1 - \tanh^2(C_t)\bigr) + \delta^c_{t+1} \odot f_{t+1}}$$

### GRU 的核心設計思想

GRU 把 LSTM 的 forget/input gate 合併為單一 update gate，且省去獨立的 cell state，用更少的參數達到接近的效果。其 BPTT 的核心在於 $h_{t-1}$ 的四條路徑加總：

$$\boxed{\frac{\partial L}{\partial h_{t-1}} = \delta^h_t \odot u_t + \left(\delta^h_{a,t} W_{hh}^\top\right) \odot r_t + \delta^r_{a,t} W_{hr}^\top + \delta^u_{a,t} W_{hu}^\top}$$

---

*延伸閱讀：Transformer（以注意力機制取代循環結構） · Bidirectional LSTM · Stacked RNN*