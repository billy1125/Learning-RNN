# Basic LSTM Showcase

> 這份筆記用 **非常小的數值例子** 介紹 LSTM。
> 重點放在：**forget gate、input gate、output gate 到底各做什麼，為什麼它比普通 RNN 更能保留資訊。**

---

## 目錄

1. [LSTM 想解決什麼問題？](#1-lstm-想解決什麼問題)
2. [本例的設定](#2-本例的設定)
3. [LSTM 的五個核心量](#3-lstm-的五個核心量)
4. [Step 1 前向傳播](#4-step-1-前向傳播)
5. [Step 2 前向傳播](#5-step-2-前向傳播)
6. [輸出與 Loss](#6-輸出與-loss)
7. [Backward 的核心想法](#7-backward-的核心想法)
8. [Step 2 反向傳播數值示範](#8-step-2-反向傳播數值示範)
9. [Step 1 反向傳播數值示範](#9-step-1-反向傳播數值示範)
10. [LSTM 為什麼比較不容易忘記？](#10-lstm-為什麼比較不容易忘記)
11. [一句話總結](#11-一句話總結)

---

## 1. LSTM 想解決什麼問題？

普通 RNN 的 hidden state 會一直經過乘法與 $\tanh$，當序列很長時，早期資訊可能越傳越弱。

LSTM 的想法是：

- 不要每次都把舊記憶整個洗掉
- 改成設計幾個「門」來控制
  - 哪些要忘記
  - 哪些要加入
  - 哪些要輸出

所以 LSTM 的核心不是「算更複雜」，而是：

> **用更精細的開關，管理記憶。**

---

## 2. 本例的設定

我們做一個最小版本：

- 序列長度只有 2
- hidden state 只有 1 個數值
- cell state 也只有 1 個數值
- 最後做 2 類分類

輸入序列取：

$$
x_1 = 1.0, \quad x_2 = 0.5
$$

目標是最後輸出第 2 類，也就是：

$$
y^* = [0,1]
$$

初始狀態設為：

$$
h_0 = 0, \quad c_0 = 0
$$

---

## 3. LSTM 的五個核心量

在每個時間點 $t$，LSTM 會算出：

1. forget gate：$f_t$
2. input gate：$i_t$
3. candidate memory：$\tilde{c}_t$
4. cell state：$c_t$
5. output gate：$o_t$
6. hidden state：$h_t$

公式如下。

### 3.1 三個 gate（用 sigmoid）

$$
f_t = \sigma(w_f x_t + u_f h_{t-1} + b_f)
$$

$$
i_t = \sigma(w_i x_t + u_i h_{t-1} + b_i)
$$

$$
o_t = \sigma(w_o x_t + u_o h_{t-1} + b_o)
$$

### 3.2 candidate memory（用 tanh）

$$
\tilde{c}_t = \tanh(w_c x_t + u_c h_{t-1} + b_c)
$$

### 3.3 更新 cell state

$$
c_t = f_t c_{t-1} + i_t \tilde{c}_t
$$

這條式子最重要。

- $f_t c_{t-1}$：保留多少舊記憶
- $i_t \tilde{c}_t$：加入多少新內容

### 3.4 hidden state

$$
h_t = o_t \tanh(c_t)
$$

---

## 4. Step 1 前向傳播

### 4.1 參數設定

我們取簡單數字：

$$
w_f = 0.7, \quad u_f = 0.4, \quad b_f = -0.1
$$

$$
w_i = 0.6, \quad u_i = 0.3, \quad b_i = 0.1
$$

$$
w_c = 0.9, \quad u_c = 0.2, \quad b_c = 0
$$

$$
w_o = 0.5, \quad u_o = 0.1, \quad b_o = 0.05
$$

並使用近似值：

- $\sigma(0.6) \approx 0.646$
- $\sigma(0.45) \approx 0.611$
- $\sigma(0.55) \approx 0.634$
- $\tanh(0.9) \approx 0.716$

### 4.2 Forget gate

$$
f_1 = \sigma(0.7 \cdot 1 + 0.4 \cdot 0 - 0.1)
$$

$$
= \sigma(0.6) \approx 0.646
$$

但因為一開始 $c_0 = 0$，這一步 forget gate 雖然算了，實際上還沒什麼可忘。

### 4.3 Input gate

$$
i_1 = \sigma(0.6 \cdot 1 + 0.3 \cdot 0 + 0.1)
$$

$$
= \sigma(0.7) \approx 0.668
$$

### 4.4 Candidate memory

$$
\tilde{c}_1 = \tanh(0.9 \cdot 1 + 0.2 \cdot 0 + 0)
$$

$$
= \tanh(0.9) \approx 0.716
$$

### 4.5 Cell state

$$
c_1 = f_1 c_0 + i_1 \tilde{c}_1
$$

$$
= 0.646 \cdot 0 + 0.668 \cdot 0.716
$$

$$
\approx 0.478
$$

### 4.6 Output gate 與 hidden state

$$
o_1 = \sigma(0.5 \cdot 1 + 0.1 \cdot 0 + 0.05)
$$

$$
= \sigma(0.55) \approx 0.634
$$

又因為

$$
\tanh(c_1) = \tanh(0.478) \approx 0.444
$$

所以：

$$
h_1 = o_1 \tanh(c_1) = 0.634 \cdot 0.444 \approx 0.281
$$

---

## 5. Step 2 前向傳播

現在輸入 $x_2 = 0.5$，而且前一步已經有：

$$
h_1 \approx 0.281, \quad c_1 \approx 0.478
$$

---

### 5.1 Forget gate

$$
f_2 = \sigma(0.7 \cdot 0.5 + 0.4 \cdot 0.281 - 0.1)
$$

$$
= \sigma(0.35 + 0.1124 - 0.1)
$$

$$
= \sigma(0.3624) \approx 0.590
$$

### 5.2 Input gate

$$
i_2 = \sigma(0.6 \cdot 0.5 + 0.3 \cdot 0.281 + 0.1)
$$

$$
= \sigma(0.3 + 0.0843 + 0.1)
$$

$$
= \sigma(0.4843) \approx 0.619
$$

### 5.3 Candidate memory

$$
\tilde{c}_2 = \tanh(0.9 \cdot 0.5 + 0.2 \cdot 0.281)
$$

$$
= \tanh(0.45 + 0.0562)
$$

$$
= \tanh(0.5062) \approx 0.467
$$

### 5.4 Cell state

$$
c_2 = f_2 c_1 + i_2 \tilde{c}_2
$$

$$
= 0.590 \cdot 0.478 + 0.619 \cdot 0.467
$$

$$
= 0.2820 + 0.2891 = 0.5711
$$

### 5.5 Output gate

$$
o_2 = \sigma(0.5 \cdot 0.5 + 0.1 \cdot 0.281 + 0.05)
$$

$$
= \sigma(0.25 + 0.0281 + 0.05)
$$

$$
= \sigma(0.3281) \approx 0.581
$$

### 5.6 Hidden state

因為：

$$
\tanh(c_2) = \tanh(0.5711) \approx 0.516
$$

所以：

$$
h_2 = o_2 \tanh(c_2) = 0.581 \cdot 0.516 \approx 0.300
$$

---

## 6. 輸出與 Loss

假設最後分類層是：

$$
z_1 = 0.2h_2 + 0.1
$$

$$
z_2 = 0.7h_2 - 0.05
$$

代入 $h_2 \approx 0.300$：

$$
z_1 = 0.2(0.300) + 0.1 = 0.160
$$

$$
z_2 = 0.7(0.300) - 0.05 = 0.160
$$

兩個 logit 一樣，所以 softmax 後：

$$
\hat{y} = [0.5, 0.5]
$$

而正確答案是第 2 類：

$$
y^* = [0,1]
$$

cross entropy loss：

$$
\mathcal{L} = -\log(0.5) \approx 0.693
$$

---

## 7. Backward 的核心想法

普通 RNN 反向傳播時，常常是 hidden 一層一層往前乘回去。

LSTM 比較特別，因為它有一條 cell state 的主幹：

$$
c_t = f_t c_{t-1} + i_t \tilde{c}_t
$$

所以梯度傳到 $c_t$ 之後，會拆成兩部分：

1. 傳到前一個 cell state：$c_{t-1}$
2. 傳到這一步的 gate 與 candidate

這也是它比較能保留長期資訊的原因。

---

## 8. Step 2 反向傳播數值示範

我們先從輸出層開始。

### 8.1 Softmax + Cross Entropy

$$
\hat{y} = [0.5,0.5], \quad y^* = [0,1]
$$

所以對 logits 的梯度是：

$$
\delta^z = \hat{y} - y^* = [0.5, -0.5]
$$

### 8.2 傳回 hidden state

因為

$$
z_1 = 0.2h_2 + 0.1, \quad z_2 = 0.7h_2 - 0.05
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial h_2} = 0.2(0.5) + 0.7(-0.5)
$$

$$
= 0.1 - 0.35 = -0.25
$$

這表示：

- 如果把 $h_2$ 增大
- loss 反而會下降
- 因為第 2 類的權重比較大

---

### 8.3 傳到 output gate 與 cell state

因為

$$
h_2 = o_2 \tanh(c_2)
$$

所以先對 $o_2$ 求導：

$$
\frac{\partial \mathcal{L}}{\partial o_2} = \frac{\partial \mathcal{L}}{\partial h_2} \cdot \tanh(c_2)
$$

$$
= -0.25 \cdot 0.516 = -0.129
$$

再對 $c_2$ 求導（先只看從 $h_2$ 這條路來的部分）：

$$
\frac{\partial \mathcal{L}}{\partial c_2} = \frac{\partial \mathcal{L}}{\partial h_2} \cdot o_2 \cdot (1 - \tanh^2(c_2))
$$

$$
= -0.25 \cdot 0.581 \cdot (1 - 0.516^2)
$$

$$
= -0.25 \cdot 0.581 \cdot (1 - 0.2663)
$$

$$
= -0.25 \cdot 0.581 \cdot 0.7337 \approx -0.1065
$$

---

### 8.4 output gate pre-activation 的梯度

設 output gate 的 pre-activation 為 $a_{o,2}$，因為

$$
o_2 = \sigma(a_{o,2})
$$

所以：

$$
\delta_{o,2} = \frac{\partial \mathcal{L}}{\partial a_{o,2}} = \frac{\partial \mathcal{L}}{\partial o_2} \cdot o_2(1-o_2)
$$

$$
= -0.129 \cdot 0.581 \cdot 0.419
$$

$$
\approx -0.0314
$$

---

### 8.5 cell state 拆到 forget / input / candidate

因為

$$
c_2 = f_2 c_1 + i_2 \tilde{c}_2
$$

所以：

#### 對 forget gate 的梯度

$$
\frac{\partial \mathcal{L}}{\partial f_2} = \frac{\partial \mathcal{L}}{\partial c_2} \cdot c_1
$$

$$
= -0.1065 \cdot 0.478 \approx -0.0509
$$

#### 對 input gate 的梯度

$$
\frac{\partial \mathcal{L}}{\partial i_2} = \frac{\partial \mathcal{L}}{\partial c_2} \cdot \tilde{c}_2
$$

$$
= -0.1065 \cdot 0.467 \approx -0.0497
$$

#### 對 candidate 的梯度

$$
\frac{\partial \mathcal{L}}{\partial \tilde{c}_2} = \frac{\partial \mathcal{L}}{\partial c_2} \cdot i_2
$$

$$
= -0.1065 \cdot 0.619 \approx -0.0659
$$

---

### 8.6 轉成 pre-activation 梯度

forget gate：

$$
\delta_{f,2} = \frac{\partial \mathcal{L}}{\partial f_2} \cdot f_2(1-f_2)
$$

$$
= -0.0509 \cdot 0.590 \cdot 0.410 \approx -0.0123
$$

input gate：

$$
\delta_{i,2} = -0.0497 \cdot 0.619 \cdot 0.381 \approx -0.0117
$$

candidate：

$$
\delta_{c,2} = \frac{\partial \mathcal{L}}{\partial \tilde{c}_2} \cdot (1-\tilde{c}_2^2)
$$

$$
= -0.0659 \cdot (1 - 0.467^2)
$$

$$
= -0.0659 \cdot 0.782 \approx -0.0515
$$

---

### 8.7 一個權重梯度示例

例如 forget gate 的 $w_f$：

$$
a_{f,2} = w_f x_2 + u_f h_1 + b_f
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial w_f}^{(t=2)} = \delta_{f,2} \cdot x_2
$$

$$
= -0.0123 \cdot 0.5 = -0.00615
$$

同理，

$$
\frac{\partial \mathcal{L}}{\partial u_f}^{(t=2)} = \delta_{f,2} \cdot h_1 = -0.0123 \cdot 0.281 \approx -0.00346
$$

---

## 9. Step 1 反向傳播數值示範

現在看 LSTM 最重要的地方：梯度怎麼從第 2 步流到第 1 步。

### 9.1 cell state 主幹往前傳

因為

$$
c_2 = f_2 c_1 + i_2 \tilde{c}_2
$$

所以對 $c_1$ 的梯度會有：

$$
\frac{\partial \mathcal{L}}{\partial c_1} = \frac{\partial \mathcal{L}}{\partial c_2} \cdot f_2
$$

$$
= -0.1065 \cdot 0.590 \approx -0.0628
$$

這一步非常關鍵。

它表示：

- 第 2 步的誤差
- 可以透過 forget gate
- 直接傳到第 1 步的 cell state

如果 $f_2$ 接近 1，記憶與梯度就能保留得比較多。

---

### 9.2 Step 1 的 output gate

由於

$$
h_1 = o_1 \tanh(c_1)
$$

而第 2 步的 gate 也依賴 $h_1$，完整反向其實會有很多項。

這裡先展示 LSTM 最核心的一條：從 $c_1$ 這條主幹回傳。

先算：

$$
\frac{\partial \mathcal{L}}{\partial o_1} \approx 0
$$

在這個簡化示例中，我們先忽略「經由 $h_1$ 影響 step 2 各個 gate」的細項，只突出 cell state 主幹。

所以先從 $c_1$ 往回拆。

### 9.3 Step 1 對 forget / input / candidate 的影響

因為

$$
c_1 = f_1 c_0 + i_1 \tilde{c}_1
$$

而 $c_0 = 0$，所以：

#### forget gate 幾乎沒貢獻

$$
\frac{\partial \mathcal{L}}{\partial f_1} = \frac{\partial \mathcal{L}}{\partial c_1} \cdot c_0 = -0.0628 \cdot 0 = 0
$$

#### input gate

$$
\frac{\partial \mathcal{L}}{\partial i_1} = \frac{\partial \mathcal{L}}{\partial c_1} \cdot \tilde{c}_1
$$

$$
= -0.0628 \cdot 0.716 \approx -0.0450
$$

#### candidate

$$
\frac{\partial \mathcal{L}}{\partial \tilde{c}_1} = \frac{\partial \mathcal{L}}{\partial c_1} \cdot i_1
$$

$$
= -0.0628 \cdot 0.668 \approx -0.0419
$$

---

### 9.4 轉成 pre-activation 梯度

input gate：

$$
\delta_{i,1} = -0.0450 \cdot 0.668 \cdot (1-0.668)
$$

$$
= -0.0450 \cdot 0.668 \cdot 0.332 \approx -0.0100
$$

candidate：

$$
\delta_{c,1} = -0.0419 \cdot (1 - 0.716^2)
$$

$$
= -0.0419 \cdot (1 - 0.5127)
$$

$$
= -0.0419 \cdot 0.4873 \approx -0.0204
$$

例如 candidate 權重 $w_c$ 在 step 1 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial w_c}^{(t=1)} = \delta_{c,1} \cdot x_1
$$

$$
= -0.0204 \cdot 1 = -0.0204
$$

---

## 10. LSTM 為什麼比較不容易忘記？

請直接看這一條：

$$
c_t = f_t c_{t-1} + i_t \tilde{c}_t
$$

如果某一步：

$$
f_t \approx 1, \quad i_t \approx 0
$$

那就會變成：

$$
c_t \approx c_{t-1}
$$

也就是：

- 幾乎不忘記舊內容
- 幾乎不加入新內容

這表示 cell state 可以像「記憶輸送帶」一樣，讓資訊穩定流下去。

而普通 RNN 沒有這麼明確的控制開關，所以比較容易在很多步之後把訊息沖淡。

---

## 11. 一句話總結

$$
\text{LSTM 的本質，就是用 gate 來決定：忘多少、記多少、輸出多少。}
$$

再更白話一點：

> **LSTM 不是比較神祕，而是比較會管理記憶。**

---

## 最後整理：這份數值例子想讓你看到什麼？

1. forget gate 決定保留多少舊記憶
2. input gate 決定加入多少新內容
3. candidate memory 提供候選新資訊
4. cell state 是真正的長期記憶主幹
5. output gate 決定輸出多少給 hidden state
6. backward 時，梯度可以沿著 $c_t$ 這條主幹穩定往前傳

如果只記一條式子，請記這條：

$$
\boxed{c_t = f_t c_{t-1} + i_t \tilde{c}_t}
$$

因為它就是 LSTM 最核心的設計。
