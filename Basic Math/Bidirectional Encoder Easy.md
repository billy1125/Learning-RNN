# Bidirectional Encoder + Attention 完整數值推導 — 高中生版

> 目標：用高中程度的數學，完整推導一個「雙向 Encoder + Attention + Decoder + Softmax」模型。
>
> 本文刻意不使用矩陣，只用「一個數字、一個數字地算」。  
> 每個前向傳播值、每個反向傳播的 $\delta$、每條梯度路徑都會算到數字。

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [為什麼需要 Attention](#2-為什麼需要-attention)
3. [核心數學工具：tanh 與 softmax](#3-核心數學工具tanh-與-softmax)
4. [Encoder 的前向傳播](#4-encoder-的前向傳播)
5. [Attention 的前向傳播](#5-attention-的前向傳播)
6. [Decoder 的前向傳播](#6-decoder-的前向傳播)
7. [輸出層與機率分佈](#7-輸出層與機率分佈)
8. [損失函數（Cross-Entropy）](#8-損失函數cross-entropy)
9. [反向傳播：梯度完整數值推導](#9-反向傳播梯度完整數值推導)
10. [參數更新：梯度下降法](#10-參數更新梯度下降法)
11. [完整數值模擬總表](#11-完整數值模擬總表)
12. [完整流程整理](#12-完整流程整理)

---

## 1. 基本符號定義

假設輸入句子只有三個 token：

```text
x1 = "I"
x2 = "love"
x3 = "cats"
```

為了讓數學可以手算，我們把每個 token 先簡化成一個數字：

$$
x_1 = 0.5,\quad x_2 = 1.0,\quad x_3 = -0.5
$$

雙向 Encoder 會從兩個方向讀句子：

- 正向 Encoder：從左到右，得到 $\overrightarrow{h_1}, \overrightarrow{h_2}, \overrightarrow{h_3}$
- 反向 Encoder：從右到左，得到 $\overleftarrow{h_1}, \overleftarrow{h_2}, \overleftarrow{h_3}$

最後把兩個方向加起來，作為每個位置的 Encoder 表示：

$$
h_i = \overrightarrow{h_i} + \overleftarrow{h_i}
$$

本文只模擬 Decoder 的第一步輸出，因此只會有一個 Decoder state：

$$
s_1
$$

Attention 會替 Decoder 算出一個 context：

$$
c_1 = \sum_{i=1}^{3} \alpha_i h_i
$$

---

## 2. 為什麼需要 Attention

如果沒有 Attention，Decoder 只能使用 Encoder 最後一個 hidden state：

$$
c = h_3
$$

這會有一個問題：一句話的資訊全部被壓縮到一個數字，長句容易丟失細節。

Attention 的做法是：不要只看最後一個位置，而是讓 Decoder 自己決定每個輸入位置的重要程度。

$$
c_1 = \alpha_1 h_1 + \alpha_2 h_2 + \alpha_3 h_3
$$

其中：

$$
\alpha_1 + \alpha_2 + \alpha_3 = 1
$$

可以把 $\alpha_i$ 想成注意力比例。例如：

$$
\alpha_2 = 0.443
$$

代表模型在這一步大約有 $44.3\%$ 的注意力放在第 2 個 token 上。

---

## 3. 核心數學工具：tanh 與 softmax

### 3.1 tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

tanh 會把任何數字壓到 $-1$ 到 $1$ 之間。

反向傳播時會用到它的導數：

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

如果：

$$
h = \tanh(a)
$$

那麼：

$$
\frac{dh}{da} = 1 - h^2
$$

### 3.2 softmax

softmax 把一組分數轉成機率：

$$
\alpha_i = \frac{e^{e_i}}{e^{e_1}+e^{e_2}+e^{e_3}}
$$

它保證：

$$
\alpha_1 + \alpha_2 + \alpha_3 = 1
$$

### 3.3 Softmax + Cross-Entropy 的重要結果

如果輸出機率是 $\hat{y}$，正確答案是 $y$，那麼：

$$
\delta^o = \frac{\partial L}{\partial o} = \hat{y} - y
$$

這個結果會讓輸出層的反向傳播變得很簡單。

---

## 4. Encoder 的前向傳播

### 4.1 參數設定

正向 Encoder 參數：

$$
w_f = 0.8,\quad u_f = 0.5,\quad b_f = 0.1
$$

反向 Encoder 參數：

$$
w_b = 0.8,\quad u_b = 0.5,\quad b_b = 0.1
$$

初始狀態：

$$
\overrightarrow{h_0} = 0,\quad \overleftarrow{h_4}=0
$$

---

### 4.2 正向 Encoder：從左到右

公式：

$$
\overrightarrow{a_t} = w_f x_t + u_f \overrightarrow{h_{t-1}} + b_f
$$

$$
\overrightarrow{h_t} = \tanh(\overrightarrow{a_t})
$$

#### 第 1 步

$$
\overrightarrow{a_1}
= 0.8(0.5)+0.5(0)+0.1
= 0.500
$$

$$
\overrightarrow{h_1} = \tanh(0.500)=0.462
$$

#### 第 2 步

$$
\overrightarrow{a_2}
=0.8(1.0)+0.5(0.462)+0.1
=0.800+0.231+0.100
=1.131
$$

$$
\overrightarrow{h_2}=\tanh(1.131)=0.812
$$

#### 第 3 步

$$
\overrightarrow{a_3}
=0.8(-0.5)+0.5(0.812)+0.1
=-0.400+0.406+0.100
=0.106
$$

$$
\overrightarrow{h_3}=\tanh(0.106)=0.106
$$

### 4.3 正向 Encoder 表格

| 位置 | $x_t$ | $\overrightarrow{a_t}$ | $\overrightarrow{h_t}$ |
|---:|---:|---:|---:|
| 1 | $0.5$ | $0.500$ | $0.462$ |
| 2 | $1.0$ | $1.131$ | $0.812$ |
| 3 | $-0.5$ | $0.106$ | $0.106$ |

---

### 4.4 反向 Encoder：從右到左

公式：

$$
\overleftarrow{a_t} = w_b x_t + u_b \overleftarrow{h_{t+1}} + b_b
$$

$$
\overleftarrow{h_t} = \tanh(\overleftarrow{a_t})
$$

#### 第 3 步

$$
\overleftarrow{a_3}
=0.8(-0.5)+0.5(0)+0.1
=-0.300
$$

$$
\overleftarrow{h_3}
=\tanh(-0.300)
=-0.291
$$

#### 第 2 步

$$
\overleftarrow{a_2}
=0.8(1.0)+0.5(-0.291)+0.1
=0.800-0.146+0.100
=0.754
$$

$$
\overleftarrow{h_2}
=\tanh(0.754)
=0.638
$$

#### 第 1 步

$$
\overleftarrow{a_1}
=0.8(0.5)+0.5(0.638)+0.1
=0.400+0.319+0.100
=0.819
$$

$$
\overleftarrow{h_1}
=\tanh(0.819)
=0.674
$$

### 4.5 反向 Encoder 表格

| 位置 | $x_t$ | $\overleftarrow{a_t}$ | $\overleftarrow{h_t}$ |
|---:|---:|---:|---:|
| 3 | $-0.5$ | $-0.300$ | $-0.291$ |
| 2 | $1.0$ | $0.754$ | $0.638$ |
| 1 | $0.5$ | $0.819$ | $0.674$ |

---

### 4.6 合併雙向 Encoder

$$
h_i = \overrightarrow{h_i} + \overleftarrow{h_i}
$$

$$
h_1 = 0.462+0.674=1.136
$$

$$
h_2 = 0.812+0.638=1.450
$$

$$
h_3 = 0.106+(-0.291)=-0.185
$$

| 位置 | $\overrightarrow{h_i}$ | $\overleftarrow{h_i}$ | $h_i$ |
|---:|---:|---:|---:|
| 1 | $0.462$ | $0.674$ | $1.136$ |
| 2 | $0.812$ | $0.638$ | $1.450$ |
| 3 | $0.106$ | $-0.291$ | $-0.185$ |

---

## 5. Attention 的前向傳播

### 5.1 Attention 參數

為了簡化，Decoder 初始狀態設為：

$$
s_0 = 0
$$

Attention score 使用：

$$
e_i = v \cdot z_i
$$

其中：

$$
z_i = \tanh(w_a s_0 + u_a h_i)
$$

設定：

$$
w_a=1.0,\quad u_a=1.2,\quad v=1.0
$$

因為 $s_0=0$，所以：

$$
z_i = \tanh(1.2h_i)
$$

---

### 5.2 計算 Attention score

#### 第 1 個位置

$$
q_1 = 1.0(0)+1.2(1.136)=1.363
$$

$$
z_1 = \tanh(1.363)=0.877
$$

$$
e_1 = 1.0(0.877)=0.877
$$

#### 第 2 個位置

$$
q_2 = 1.0(0)+1.2(1.450)=1.740
$$

$$
z_2 = \tanh(1.740)=0.940
$$

$$
e_2 = 1.0(0.940)=0.940
$$

#### 第 3 個位置

$$
q_3 = 1.0(0)+1.2(-0.185)=-0.222
$$

$$
z_3 = \tanh(-0.222)=-0.218
$$

$$
e_3 = 1.0(-0.218)=-0.218
$$

| 位置 | $h_i$ | $q_i=1.2h_i$ | $z_i=\tanh(q_i)$ | $e_i$ |
|---:|---:|---:|---:|---:|
| 1 | $1.136$ | $1.363$ | $0.877$ | $0.877$ |
| 2 | $1.450$ | $1.740$ | $0.940$ | $0.940$ |
| 3 | $-0.185$ | $-0.222$ | $-0.218$ | $-0.218$ |

---

### 5.3 用 softmax 算 Attention 權重

$$
e^{0.877}=2.404
$$

$$
e^{0.940}=2.560
$$

$$
e^{-0.218}=0.804
$$

總和：

$$
2.404+2.560+0.804=5.768
$$

所以：

$$
\alpha_1=\frac{2.404}{5.768}=0.417
$$

$$
\alpha_2=\frac{2.560}{5.768}=0.444
$$

$$
\alpha_3=\frac{0.804}{5.768}=0.139
$$

驗算：

$$
0.417+0.444+0.139=1.000
$$

| 位置 | $e_i$ | $e^{e_i}$ | $\alpha_i$ |
|---:|---:|---:|---:|
| 1 | $0.877$ | $2.404$ | $0.417$ |
| 2 | $0.940$ | $2.560$ | $0.444$ |
| 3 | $-0.218$ | $0.804$ | $0.139$ |

---

### 5.4 計算 context vector

$$
c_1 = \alpha_1h_1+\alpha_2h_2+\alpha_3h_3
$$

$$
c_1
=0.417(1.136)+0.444(1.450)+0.139(-0.185)
$$

$$
c_1
=0.474+0.644-0.026
=1.092
$$

---

## 6. Decoder 的前向傳播

Decoder 使用：

$$
a^{dec}_1 = w_c c_1 + w_s s_0 + b_s
$$

設定：

$$
w_c=0.9,\quad w_s=0.7,\quad b_s=0,\quad s_0=0
$$

所以：

$$
a^{dec}_1
=0.9(1.092)+0.7(0)+0
=0.983
$$

$$
s_1=\tanh(0.983)=0.754
$$

---

## 7. 輸出層與機率分佈

假設輸出詞彙表只有兩個字：

```text
A, B
```

輸出層參數：

$$
w_A=1.5,\quad w_B=-0.5
$$

偏差都設為 $0$。

### 7.1 算 logits

$$
o_A = 1.5(0.754)=1.131
$$

$$
o_B = -0.5(0.754)=-0.377
$$

### 7.2 softmax 成機率

$$
e^{1.131}=3.099
$$

$$
e^{-0.377}=0.686
$$

總和：

$$
3.099+0.686=3.785
$$

$$
\hat{y}_A=\frac{3.099}{3.785}=0.819
$$

$$
\hat{y}_B=\frac{0.686}{3.785}=0.181
$$

| 類別 | logit | 指數值 | 機率 |
|---:|---:|---:|---:|
| A | $1.131$ | $3.099$ | $0.819$ |
| B | $-0.377$ | $0.686$ | $0.181$ |

---

## 8. 損失函數（Cross-Entropy）

假設正確答案是 A：

$$
y_A=1,\quad y_B=0
$$

交叉熵：

$$
L = -\log(\hat{y}_A)
$$

$$
L=-\log(0.819)=0.200
$$

---

## 9. 反向傳播：梯度完整數值推導

反向傳播的方向是：

```text
Loss
→ output logits
→ decoder state
→ decoder tanh
→ context
→ attention weights
→ attention scores
→ encoder hidden states
→ 正向 Encoder BPTT
→ 反向 Encoder BPTT
```

---

### 9.1 輸出層梯度

Softmax + Cross-Entropy 的結果：

$$
\delta_A^o = \hat{y}_A-y_A
$$

$$
\delta_B^o = \hat{y}_B-y_B
$$

代入數字：

$$
\delta_A^o = 0.819-1=-0.181
$$

$$
\delta_B^o = 0.181-0=0.181
$$

| 類別 | $\hat{y}$ | $y$ | $\delta^o=\hat{y}-y$ |
|---:|---:|---:|---:|
| A | $0.819$ | $1$ | $-0.181$ |
| B | $0.181$ | $0$ | $0.181$ |

意思是：

- A 是正確答案，但機率還不夠高，所以 A 的分數要上升。
- B 不是正確答案，但機率有 $18.1\%$，所以 B 的分數要下降。

---

### 9.2 輸出層參數梯度

因為：

$$
o_A = w_A s_1
$$

所以：

$$
\frac{\partial L}{\partial w_A}=\delta_A^o s_1
$$

$$
\frac{\partial L}{\partial w_B}=\delta_B^o s_1
$$

代入：

$$
\frac{\partial L}{\partial w_A}
=(-0.181)(0.754)
=-0.136
$$

$$
\frac{\partial L}{\partial w_B}
=(0.181)(0.754)
=0.136
$$

| 參數 | 梯度 |
|---:|---:|
| $\frac{\partial L}{\partial w_A}$ | $-0.136$ |
| $\frac{\partial L}{\partial w_B}$ | $0.136$ |

---

### 9.3 從輸出層傳回 Decoder state

因為：

$$
o_A=w_A s_1,\quad o_B=w_B s_1
$$

所以 $s_1$ 收到的梯度是兩個類別加總：

$$
g_s
= w_A\delta_A^o+w_B\delta_B^o
$$

代入：

$$
g_s
=1.5(-0.181)+(-0.5)(0.181)
$$

$$
g_s
=-0.272-0.091
=-0.363
$$

所以：

$$
\frac{\partial L}{\partial s_1}=-0.363
$$

---

### 9.4 穿過 Decoder 的 tanh

Decoder：

$$
s_1=\tanh(a^{dec}_1)
$$

tanh 導數：

$$
\frac{\partial s_1}{\partial a^{dec}_1}=1-s_1^2
$$

代入：

$$
1-s_1^2
=1-(0.754)^2
=1-0.569
=0.431
$$

所以：

$$
\delta^{dec}_1
=\frac{\partial L}{\partial a^{dec}_1}
=g_s(1-s_1^2)
$$

$$
\delta^{dec}_1
=(-0.363)(0.431)
=-0.156
$$

---

### 9.5 Decoder 參數梯度

Decoder 預激活：

$$
a^{dec}_1 = w_c c_1+w_s s_0+b_s
$$

所以：

$$
\frac{\partial L}{\partial w_c}
=\delta^{dec}_1 c_1
$$

$$
\frac{\partial L}{\partial w_s}
=\delta^{dec}_1 s_0
$$

$$
\frac{\partial L}{\partial b_s}
=\delta^{dec}_1
$$

代入：

$$
\frac{\partial L}{\partial w_c}
=(-0.156)(1.092)
=-0.170
$$

$$
\frac{\partial L}{\partial w_s}
=(-0.156)(0)
=0
$$

$$
\frac{\partial L}{\partial b_s}
=-0.156
$$

| Decoder 參數 | 梯度 |
|---:|---:|
| $\frac{\partial L}{\partial w_c}$ | $-0.170$ |
| $\frac{\partial L}{\partial w_s}$ | $0.000$ |
| $\frac{\partial L}{\partial b_s}$ | $-0.156$ |

---

### 9.6 傳回 context vector

因為：

$$
a^{dec}_1 = w_c c_1+w_s s_0+b_s
$$

所以：

$$
g_c=\frac{\partial L}{\partial c_1}
=w_c\delta^{dec}_1
$$

代入：

$$
g_c
=0.9(-0.156)
=-0.140
$$

這代表：

$$
\frac{\partial L}{\partial c_1}=-0.140
$$

---

## 9.7 Attention 第一條路：context 直接傳回 Encoder

context 是：

$$
c_1 = \alpha_1h_1+\alpha_2h_2+\alpha_3h_3
$$

所以每個 $h_i$ 會直接收到一條梯度：

$$
\frac{\partial L}{\partial h_i}^{(context)}
=\alpha_i g_c
$$

代入：

$$
\frac{\partial L}{\partial h_1}^{(context)}
=0.417(-0.140)
=-0.058
$$

$$
\frac{\partial L}{\partial h_2}^{(context)}
=0.444(-0.140)
=-0.062
$$

$$
\frac{\partial L}{\partial h_3}^{(context)}
=0.139(-0.140)
=-0.019
$$

| 位置 | $\alpha_i$ | $g_c$ | context 路梯度 |
|---:|---:|---:|---:|
| 1 | $0.417$ | $-0.140$ | $-0.058$ |
| 2 | $0.444$ | $-0.140$ | $-0.062$ |
| 3 | $0.139$ | $-0.140$ | $-0.019$ |

這是 Attention 的第一條路：

```text
c1 → hi
```

意思是：某個位置的 attention 權重越大，它直接收到的梯度越大。

---

## 9.8 Attention 第二條路：經過 attention score 傳回 Encoder

Attention 還有第二條路：

```text
c1 → αi → ei → zi → qi → hi
```

這條路表示：模型不只要調整 Encoder hidden state 本身，也要調整「怎麼算注意力分數」。

---

### 9.8.1 從 context 傳到 attention weight

因為：

$$
c_1 = \sum_i \alpha_i h_i
$$

所以：

$$
\frac{\partial L}{\partial \alpha_i}
=g_c h_i
$$

代入：

$$
\frac{\partial L}{\partial \alpha_1}
=(-0.140)(1.136)
=-0.159
$$

$$
\frac{\partial L}{\partial \alpha_2}
=(-0.140)(1.450)
=-0.203
$$

$$
\frac{\partial L}{\partial \alpha_3}
=(-0.140)(-0.185)
=0.026
$$

| 位置 | $h_i$ | $\frac{\partial L}{\partial \alpha_i}$ |
|---:|---:|---:|
| 1 | $1.136$ | $-0.159$ |
| 2 | $1.450$ | $-0.203$ |
| 3 | $-0.185$ | $0.026$ |

---

### 9.8.2 softmax 反傳：從 $\alpha_i$ 傳到 $e_i$

softmax 的反傳公式：

$$
\delta_i^e
=
\alpha_i
\left(
\frac{\partial L}{\partial \alpha_i}
-
\sum_j \alpha_j \frac{\partial L}{\partial \alpha_j}
\right)
$$

先算平均項：

$$
M=\sum_j \alpha_j \frac{\partial L}{\partial \alpha_j}
$$

$$
M
=0.417(-0.159)+0.444(-0.203)+0.139(0.026)
$$

$$
M
=-0.066-0.090+0.004
=-0.152
$$

接著算每一個 $\delta_i^e$。

#### 位置 1

$$
\delta_1^e
=0.417((-0.159)-(-0.152))
$$

$$
=0.417(-0.007)
=-0.003
$$

#### 位置 2

$$
\delta_2^e
=0.444((-0.203)-(-0.152))
$$

$$
=0.444(-0.051)
=-0.023
$$

#### 位置 3

$$
\delta_3^e
=0.139((0.026)-(-0.152))
$$

$$
=0.139(0.178)
=0.025
$$

| 位置 | $\alpha_i$ | $\frac{\partial L}{\partial \alpha_i}$ | $M$ | $\delta_i^e$ |
|---:|---:|---:|---:|---:|
| 1 | $0.417$ | $-0.159$ | $-0.152$ | $-0.003$ |
| 2 | $0.444$ | $-0.203$ | $-0.152$ | $-0.023$ |
| 3 | $0.139$ | $0.026$ | $-0.152$ | $0.025$ |

驗算：

$$
\delta_1^e+\delta_2^e+\delta_3^e
=-0.003-0.023+0.025
\approx -0.001 \approx 0
$$

softmax 梯度通常加總接近 $0$，因為 softmax 的機率總和固定是 $1$。

---

### 9.8.3 從 $e_i$ 傳到 $z_i$

因為：

$$
e_i = v z_i
$$

而：

$$
v=1.0
$$

所以：

$$
\frac{\partial L}{\partial z_i}
=
\delta_i^e v
=
\delta_i^e
$$

| 位置 | $\delta_i^e$ | $\frac{\partial L}{\partial z_i}$ |
|---:|---:|---:|
| 1 | $-0.003$ | $-0.003$ |
| 2 | $-0.023$ | $-0.023$ |
| 3 | $0.025$ | $0.025$ |

---

### 9.8.4 穿過 attention 的 tanh

Attention 內部：

$$
z_i = \tanh(q_i)
$$

所以：

$$
\delta_i^q
=
\frac{\partial L}{\partial q_i}
=
\frac{\partial L}{\partial z_i}(1-z_i^2)
$$

#### 位置 1

$$
1-z_1^2
=
1-(0.877)^2
=1-0.769
=0.231
$$

$$
\delta_1^q
=
(-0.003)(0.231)
=-0.001
$$

#### 位置 2

$$
1-z_2^2
=
1-(0.940)^2
=1-0.884
=0.116
$$

$$
\delta_2^q
=
(-0.023)(0.116)
=-0.003
$$

#### 位置 3

$$
1-z_3^2
=
1-(-0.218)^2
=1-0.048
=0.952
$$

$$
\delta_3^q
=
0.025(0.952)
=0.024
$$

| 位置 | $z_i$ | $1-z_i^2$ | $\delta_i^q$ |
|---:|---:|---:|---:|
| 1 | $0.877$ | $0.231$ | $-0.001$ |
| 2 | $0.940$ | $0.116$ | $-0.003$ |
| 3 | $-0.218$ | $0.952$ | $0.024$ |

---

### 9.8.5 Attention score 路傳回 Encoder

因為：

$$
q_i = w_a s_0 + u_a h_i
$$

所以：

$$
\frac{\partial L}{\partial h_i}^{(score)}
=
u_a \delta_i^q
$$

其中：

$$
u_a=1.2
$$

代入：

$$
\frac{\partial L}{\partial h_1}^{(score)}
=1.2(-0.001)
=-0.001
$$

$$
\frac{\partial L}{\partial h_2}^{(score)}
=1.2(-0.003)
=-0.004
$$

$$
\frac{\partial L}{\partial h_3}^{(score)}
=1.2(0.024)
=0.029
$$

| 位置 | $\delta_i^q$ | score 路梯度 |
|---:|---:|---:|
| 1 | $-0.001$ | $-0.001$ |
| 2 | $-0.003$ | $-0.004$ |
| 3 | $0.024$ | $0.029$ |

---

## 9.9 Encoder hidden state 的總梯度

每個 Encoder 表示 $h_i$ 收到兩條梯度：

1. context 直接路
2. attention score 路

所以：

$$
g_i^h
=
\frac{\partial L}{\partial h_i}^{(context)}
+
\frac{\partial L}{\partial h_i}^{(score)}
$$

代入：

$$
g_1^h
=
-0.058+(-0.001)
=-0.059
$$

$$
g_2^h
=
-0.062+(-0.004)
=-0.066
$$

$$
g_3^h
=
-0.019+0.029
=0.010
$$

| 位置 | context 路 | score 路 | 總梯度 $g_i^h$ |
|---:|---:|---:|---:|
| 1 | $-0.058$ | $-0.001$ | $-0.059$ |
| 2 | $-0.062$ | $-0.004$ | $-0.066$ |
| 3 | $-0.019$ | $0.029$ | $0.010$ |

因為：

$$
h_i = \overrightarrow{h_i}+\overleftarrow{h_i}
$$

所以正向與反向 hidden state 都會收到同樣的外部梯度：

$$
\frac{\partial L}{\partial \overrightarrow{h_i}} = g_i^h
$$

$$
\frac{\partial L}{\partial \overleftarrow{h_i}} = g_i^h
$$

---

## 9.10 正向 Encoder 的 BPTT

正向 Encoder：

$$
\overrightarrow{a_t}
=
w_f x_t + u_f \overrightarrow{h_{t-1}} + b_f
$$

$$
\overrightarrow{h_t}
=
\tanh(\overrightarrow{a_t})
$$

正向 BPTT 要從後往前算，因為 $\overrightarrow{h_t}$ 會影響 $\overrightarrow{h_{t+1}}$。

公式：

$$
\overrightarrow{\delta_t}
=
\left(
g_t^h + u_f\overrightarrow{\delta_{t+1}}
\right)
(1-\overrightarrow{h_t}^2)
$$

邊界：

$$
\overrightarrow{\delta_4}=0
$$

---

### 9.10.1 正向第 3 步

$$
g_3^h=0.010
$$

$$
\overrightarrow{\delta_4}=0
$$

$$
1-\overrightarrow{h_3}^2
=
1-(0.106)^2
=
0.989
$$

$$
\overrightarrow{\delta_3}
=
(0.010+0.5(0))(0.989)
=
0.010
$$

---

### 9.10.2 正向第 2 步

$$
g_2^h=-0.066
$$

$$
\overrightarrow{\delta_3}=0.010
$$

$$
1-\overrightarrow{h_2}^2
=
1-(0.812)^2
=
1-0.659
=
0.341
$$

$$
\overrightarrow{\delta_2}
=
(-0.066+0.5(0.010))(0.341)
$$

$$
=
(-0.066+0.005)(0.341)
=
(-0.061)(0.341)
=
-0.021
$$

---

### 9.10.3 正向第 1 步

$$
g_1^h=-0.059
$$

$$
\overrightarrow{\delta_2}=-0.021
$$

$$
1-\overrightarrow{h_1}^2
=
1-(0.462)^2
=
1-0.213
=
0.787
$$

$$
\overrightarrow{\delta_1}
=
(-0.059+0.5(-0.021))(0.787)
$$

$$
=
(-0.059-0.011)(0.787)
=
(-0.070)(0.787)
=
-0.055
$$

### 9.10.4 正向 Encoder BPTT 表格

| 位置 | $g_t^h$ | 下一步傳回 $u_f\overrightarrow{\delta_{t+1}}$ | $1-\overrightarrow{h_t}^2$ | $\overrightarrow{\delta_t}$ |
|---:|---:|---:|---:|---:|
| 3 | $0.010$ | $0.000$ | $0.989$ | $0.010$ |
| 2 | $-0.066$ | $0.005$ | $0.341$ | $-0.021$ |
| 1 | $-0.059$ | $-0.011$ | $0.787$ | $-0.055$ |

---

## 9.11 正向 Encoder 參數梯度

公式：

$$
\frac{\partial L}{\partial w_f}
=
\sum_{t=1}^{3}
\overrightarrow{\delta_t}x_t
$$

$$
\frac{\partial L}{\partial u_f}
=
\sum_{t=1}^{3}
\overrightarrow{\delta_t}\overrightarrow{h_{t-1}}
$$

$$
\frac{\partial L}{\partial b_f}
=
\sum_{t=1}^{3}
\overrightarrow{\delta_t}
$$

### 9.11.1 正向 $w_f$ 梯度

$$
\frac{\partial L}{\partial w_f}
=
(-0.055)(0.5)+(-0.021)(1.0)+(0.010)(-0.5)
$$

$$
=
-0.028-0.021-0.005
=
-0.054
$$

### 9.11.2 正向 $u_f$ 梯度

注意：

$$
\overrightarrow{h_0}=0,\quad
\overrightarrow{h_1}=0.462,\quad
\overrightarrow{h_2}=0.812
$$

$$
\frac{\partial L}{\partial u_f}
=
(-0.055)(0)+(-0.021)(0.462)+(0.010)(0.812)
$$

$$
=
0-0.010+0.008
=
-0.002
$$

### 9.11.3 正向 $b_f$ 梯度

$$
\frac{\partial L}{\partial b_f}
=
-0.055-0.021+0.010
=
-0.066
$$

| 正向參數 | 梯度 |
|---:|---:|
| $\frac{\partial L}{\partial w_f}$ | $-0.054$ |
| $\frac{\partial L}{\partial u_f}$ | $-0.002$ |
| $\frac{\partial L}{\partial b_f}$ | $-0.066$ |

---

## 9.12 反向 Encoder 的 BPTT

反向 Encoder：

$$
\overleftarrow{a_t}
=
w_b x_t + u_b\overleftarrow{h_{t+1}}+b_b
$$

$$
\overleftarrow{h_t}
=
\tanh(\overleftarrow{a_t})
$$

注意反向 Encoder 是從右讀到左，所以在 BPTT 時，梯度會從左往右傳。

公式：

$$
\overleftarrow{\delta_t}
=
\left(
g_t^h + u_b\overleftarrow{\delta_{t-1}}
\right)
(1-\overleftarrow{h_t}^2)
$$

邊界：

$$
\overleftarrow{\delta_0}=0
$$

---

### 9.12.1 反向第 1 步

$$
g_1^h=-0.059
$$

$$
\overleftarrow{\delta_0}=0
$$

$$
1-\overleftarrow{h_1}^2
=
1-(0.674)^2
=
1-0.454
=
0.546
$$

$$
\overleftarrow{\delta_1}
=
(-0.059+0.5(0))(0.546)
=
-0.032
$$

---

### 9.12.2 反向第 2 步

$$
g_2^h=-0.066
$$

$$
\overleftarrow{\delta_1}=-0.032
$$

$$
1-\overleftarrow{h_2}^2
=
1-(0.638)^2
=
1-0.407
=
0.593
$$

$$
\overleftarrow{\delta_2}
=
(-0.066+0.5(-0.032))(0.593)
$$

$$
=
(-0.066-0.016)(0.593)
=
(-0.082)(0.593)
=
-0.049
$$

---

### 9.12.3 反向第 3 步

$$
g_3^h=0.010
$$

$$
\overleftarrow{\delta_2}=-0.049
$$

$$
1-\overleftarrow{h_3}^2
=
1-(-0.291)^2
=
1-0.085
=
0.915
$$

$$
\overleftarrow{\delta_3}
=
(0.010+0.5(-0.049))(0.915)
$$

$$
=
(0.010-0.025)(0.915)
=
(-0.015)(0.915)
=
-0.014
$$

### 9.12.4 反向 Encoder BPTT 表格

| 位置 | $g_t^h$ | 前一步傳回 $u_b\overleftarrow{\delta_{t-1}}$ | $1-\overleftarrow{h_t}^2$ | $\overleftarrow{\delta_t}$ |
|---:|---:|---:|---:|---:|
| 1 | $-0.059$ | $0.000$ | $0.546$ | $-0.032$ |
| 2 | $-0.066$ | $-0.016$ | $0.593$ | $-0.049$ |
| 3 | $0.010$ | $-0.025$ | $0.915$ | $-0.014$ |

---

## 9.13 反向 Encoder 參數梯度

公式：

$$
\frac{\partial L}{\partial w_b}
=
\sum_{t=1}^{3}
\overleftarrow{\delta_t}x_t
$$

$$
\frac{\partial L}{\partial u_b}
=
\sum_{t=1}^{3}
\overleftarrow{\delta_t}\overleftarrow{h_{t+1}}
$$

$$
\frac{\partial L}{\partial b_b}
=
\sum_{t=1}^{3}
\overleftarrow{\delta_t}
$$

### 9.13.1 反向 $w_b$ 梯度

$$
\frac{\partial L}{\partial w_b}
=
(-0.032)(0.5)+(-0.049)(1.0)+(-0.014)(-0.5)
$$

$$
=
-0.016-0.049+0.007
=
-0.058
$$

### 9.13.2 反向 $u_b$ 梯度

注意：

$$
\overleftarrow{h_4}=0,\quad
\overleftarrow{h_3}=-0.291,\quad
\overleftarrow{h_2}=0.638
$$

$$
\frac{\partial L}{\partial u_b}
=
(-0.032)(0.638)+(-0.049)(-0.291)+(-0.014)(0)
$$

$$
=
-0.020+0.014+0
=
-0.006
$$

### 9.13.3 反向 $b_b$ 梯度

$$
\frac{\partial L}{\partial b_b}
=
-0.032-0.049-0.014
=
-0.095
$$

| 反向參數 | 梯度 |
|---:|---:|
| $\frac{\partial L}{\partial w_b}$ | $-0.058$ |
| $\frac{\partial L}{\partial u_b}$ | $-0.006$ |
| $\frac{\partial L}{\partial b_b}$ | $-0.095$ |

---

## 9.14 Attention 參數梯度

Attention：

$$
q_i=w_as_0+u_ah_i
$$

$$
z_i=\tanh(q_i)
$$

$$
e_i=vz_i
$$

### 9.14.1 對 $v$ 的梯度

因為：

$$
e_i=vz_i
$$

所以：

$$
\frac{\partial L}{\partial v}
=
\sum_i \delta_i^e z_i
$$

代入：

$$
\frac{\partial L}{\partial v}
=
(-0.003)(0.877)+(-0.023)(0.940)+(0.025)(-0.218)
$$

$$
=
-0.003-0.022-0.005
=
-0.030
$$

### 9.14.2 對 $w_a$ 的梯度

$$
\frac{\partial L}{\partial w_a}
=
\sum_i \delta_i^q s_0
$$

因為 $s_0=0$：

$$
\frac{\partial L}{\partial w_a}
=0
$$

### 9.14.3 對 $u_a$ 的梯度

$$
\frac{\partial L}{\partial u_a}
=
\sum_i \delta_i^q h_i
$$

代入：

$$
\frac{\partial L}{\partial u_a}
=
(-0.001)(1.136)+(-0.003)(1.450)+(0.024)(-0.185)
$$

$$
=
-0.001-0.004-0.004
=
-0.009
$$

| Attention 參數 | 梯度 |
|---:|---:|
| $\frac{\partial L}{\partial v}$ | $-0.030$ |
| $\frac{\partial L}{\partial w_a}$ | $0.000$ |
| $\frac{\partial L}{\partial u_a}$ | $-0.009$ |

---

## 10. 參數更新：梯度下降法

梯度下降公式：

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

設定學習率：

$$
\eta=0.1
$$

---

### 10.1 輸出層更新

$$
w_A \leftarrow 1.5 - 0.1(-0.136)
=1.514
$$

$$
w_B \leftarrow -0.5 - 0.1(0.136)
=-0.514
$$

---

### 10.2 Decoder 更新

$$
w_c \leftarrow 0.9 - 0.1(-0.170)
=0.917
$$

$$
w_s \leftarrow 0.7 - 0.1(0)
=0.700
$$

$$
b_s \leftarrow 0 - 0.1(-0.156)
=0.016
$$

---

### 10.3 Attention 更新

$$
v \leftarrow 1.0 - 0.1(-0.030)
=1.003
$$

$$
w_a \leftarrow 1.0 - 0.1(0)
=1.000
$$

$$
u_a \leftarrow 1.2 - 0.1(-0.009)
=1.201
$$

---

### 10.4 正向 Encoder 更新

$$
w_f \leftarrow 0.8 - 0.1(-0.054)
=0.805
$$

$$
u_f \leftarrow 0.5 - 0.1(-0.002)
=0.500
$$

$$
b_f \leftarrow 0.1 - 0.1(-0.066)
=0.107
$$

---

### 10.5 反向 Encoder 更新

$$
w_b \leftarrow 0.8 - 0.1(-0.058)
=0.806
$$

$$
u_b \leftarrow 0.5 - 0.1(-0.006)
=0.501
$$

$$
b_b \leftarrow 0.1 - 0.1(-0.095)
=0.110
$$

---

## 11. 完整數值模擬總表

### 11.1 前向傳播總表

| 階段 | 數值 |
|---|---|
| $\overrightarrow{h_1},\overrightarrow{h_2},\overrightarrow{h_3}$ | $0.462,\ 0.812,\ 0.106$ |
| $\overleftarrow{h_1},\overleftarrow{h_2},\overleftarrow{h_3}$ | $0.674,\ 0.638,\ -0.291$ |
| $h_1,h_2,h_3$ | $1.136,\ 1.450,\ -0.185$ |
| $e_1,e_2,e_3$ | $0.877,\ 0.940,\ -0.218$ |
| $\alpha_1,\alpha_2,\alpha_3$ | $0.417,\ 0.444,\ 0.139$ |
| $c_1$ | $1.092$ |
| $s_1$ | $0.754$ |
| $\hat{y}_A,\hat{y}_B$ | $0.819,\ 0.181$ |
| $L$ | $0.200$ |

---

### 11.2 反向傳播總表

| 梯度項目 | 數值 |
|---|---:|
| $\delta_A^o$ | $-0.181$ |
| $\delta_B^o$ | $0.181$ |
| $g_s=\partial L/\partial s_1$ | $-0.363$ |
| $\delta_1^{dec}$ | $-0.156$ |
| $g_c=\partial L/\partial c_1$ | $-0.140$ |
| $\delta_1^e$ | $-0.003$ |
| $\delta_2^e$ | $-0.023$ |
| $\delta_3^e$ | $0.025$ |
| $\delta_1^q$ | $-0.001$ |
| $\delta_2^q$ | $-0.003$ |
| $\delta_3^q$ | $0.024$ |
| $g_1^h$ | $-0.059$ |
| $g_2^h$ | $-0.066$ |
| $g_3^h$ | $0.010$ |
| $\overrightarrow{\delta_1}$ | $-0.055$ |
| $\overrightarrow{\delta_2}$ | $-0.021$ |
| $\overrightarrow{\delta_3}$ | $0.010$ |
| $\overleftarrow{\delta_1}$ | $-0.032$ |
| $\overleftarrow{\delta_2}$ | $-0.049$ |
| $\overleftarrow{\delta_3}$ | $-0.014$ |

---

### 11.3 參數更新總表

| 參數 | 更新前 | 梯度 | 更新後 |
|---:|---:|---:|---:|
| $w_A$ | $1.500$ | $-0.136$ | $1.514$ |
| $w_B$ | $-0.500$ | $0.136$ | $-0.514$ |
| $w_c$ | $0.900$ | $-0.170$ | $0.917$ |
| $w_s$ | $0.700$ | $0.000$ | $0.700$ |
| $b_s$ | $0.000$ | $-0.156$ | $0.016$ |
| $v$ | $1.000$ | $-0.030$ | $1.003$ |
| $w_a$ | $1.000$ | $0.000$ | $1.000$ |
| $u_a$ | $1.200$ | $-0.009$ | $1.201$ |
| $w_f$ | $0.800$ | $-0.054$ | $0.805$ |
| $u_f$ | $0.500$ | $-0.002$ | $0.500$ |
| $b_f$ | $0.100$ | $-0.066$ | $0.107$ |
| $w_b$ | $0.800$ | $-0.058$ | $0.806$ |
| $u_b$ | $0.500$ | $-0.006$ | $0.501$ |
| $b_b$ | $0.100$ | $-0.095$ | $0.110$ |

---

## 12. 完整流程整理

### 12.1 前向傳播

```text
x1, x2, x3
   ↓
正向 Encoder：左 → 右
   ↓
反向 Encoder：右 → 左
   ↓
合併 hi
   ↓
Attention score ei
   ↓
softmax 得到 αi
   ↓
context c1
   ↓
Decoder state s1
   ↓
輸出 logits
   ↓
softmax 機率
   ↓
Cross-Entropy loss
```

---

### 12.2 反向傳播

```text
Loss
  ↓
δo = ŷ - y
  ↓
輸出層梯度
  ↓
Decoder state 梯度
  ↓
穿過 tanh：δdec
  ↓
context 梯度 gc
  ↓
兩條 Attention 梯度路：

第一條：
c1 → hi

第二條：
c1 → αi → ei → zi → qi → hi

  ↓
兩條路加總成 gh
  ↓
正向 Encoder BPTT
  ↓
反向 Encoder BPTT
  ↓
所有參數更新
```

---

### 12.3 最重要的五個觀念

#### 觀念 1：Attention 的 context 是加權平均

$$
c_1=\sum_i\alpha_i h_i
$$

權重越大，代表該位置越重要。

---

#### 觀念 2：Attention 有兩條梯度路

第一條是直接路：

$$
c_1 \rightarrow h_i
$$

第二條是分數路：

$$
c_1 \rightarrow \alpha_i \rightarrow e_i \rightarrow h_i
$$

所以 Encoder 不只學「要提供什麼資訊」，也學「如何被注意到」。

---

#### 觀念 3：tanh 的反傳只要乘上 $1-h^2$

$$
\delta_{\text{前}}=\delta_{\text{後}}(1-h^2)
$$

這是整個 RNN / Encoder / Decoder 都會反覆用到的公式。

---

#### 觀念 4：softmax + cross-entropy 的輸出梯度很簡單

$$
\delta^o=\hat{y}-y
$$

這就是模型知道「預測和正確答案差多少」的地方。

---

#### 觀念 5：BPTT 是把時間上的影響也算回去

正向 Encoder：

$$
\overrightarrow{\delta_t}
=
(g_t^h+u_f\overrightarrow{\delta_{t+1}})
(1-\overrightarrow{h_t}^2)
$$

反向 Encoder：

$$
\overleftarrow{\delta_t}
=
(g_t^h+u_b\overleftarrow{\delta_{t-1}})
(1-\overleftarrow{h_t}^2)
$$

差別只在時間方向不同。

---

## 最後總結

這個模型做了一件事：

1. 雙向 Encoder 先從左右兩邊理解輸入。
2. Attention 根據 Decoder 的需求，決定每個位置的重要程度。
3. Decoder 用 context 產生輸出。
4. Softmax 把分數變成機率。
5. Cross-Entropy 衡量錯誤。
6. 反向傳播把錯誤一路分配回每個參數。
7. 梯度下降讓參數往更好的方向移動。

如果只記一句話：

> Bidirectional Encoder 負責「看完整句」，Attention 負責「選重要位置」，Backpropagation 負責「把錯誤精準傳回該修正的地方」。

---

*數學工具：tanh、softmax、cross-entropy、鏈鎖律、加權平均、BPTT、梯度下降*
