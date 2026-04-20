# Basic Seq2Seq Showcase

> 參考一般基礎神經網路講義的寫法，改成 **高中生可讀** 的 Seq2Seq 數值模擬。
> 重點不是追求工程上完整，而是看懂：**encoder 怎麼把輸入壓成一個狀態，decoder 又怎麼一步一步產生輸出，最後 loss 如何一路傳回 encoder。**

---

## 目錄

1. [Seq2Seq 是什麼？](#1-seq2seq-是什麼)
2. [本例的任務設定](#2-本例的任務設定)
3. [基本公式](#3-基本公式)
4. [Encoder 前向傳播](#4-encoder-前向傳播)
5. [Decoder 前向傳播](#5-decoder-前向傳播)
6. [Loss 計算](#6-loss-計算)
7. [Output Layer 反向傳播](#7-output-layer-反向傳播)
8. [Decoder Hidden 反向傳播](#8-decoder-hidden-反向傳播)
9. [梯度如何傳回 Encoder](#9-梯度如何傳回-encoder)
10. [參數更新示意](#10-參數更新示意)
11. [一句話總結](#11-一句話總結)

---

## 1. Seq2Seq 是什麼？

Seq2Seq 的全名是 **Sequence to Sequence**，意思是：

- 輸入是一串資料
- 輸出也是一串資料

例如：

- 中文句子 → 英文句子
- 語音訊號 → 文字
- 一串數字 → 另一串數字

最基本的 Seq2Seq 可以拆成兩段：

1. **Encoder**：把輸入序列讀完，整理成一個最後狀態
2. **Decoder**：根據這個最後狀態，逐步產生輸出序列

你可以把它想成：

- Encoder 像「看完整題目的人」
- Decoder 像「根據理解開始作答的人」

---

## 2. 本例的任務設定

我們做一個很小的玩具例子：

- 輸入序列長度：2 步
- 輸出序列長度：2 步
- hidden state 只有 **1 個數值**，這樣比較容易算
- encoder 與 decoder 都用最基本的 RNN
- hidden activation 使用 $\tanh$
- 輸出層用 softmax

### 2.1 字彙表

我們只用三個符號：

- A
- B
- C

對應 one-hot：

$$
A = [1,0,0], \quad B = [0,1,0], \quad C = [0,0,1]
$$

### 2.2 任務

輸入：

$$
[A, B]
$$

希望 decoder 輸出：

$$
[B, C]
$$

也就是：

- 第 1 步目標是 $B$
- 第 2 步目標是 $C$

---

## 3. 基本公式

因為 hidden state 只有 1 維，所以很多公式都會變成普通的四則運算。

### 3.1 Encoder

$$
a_t^{enc} = w_{xh}^{enc} x_t + w_{hh}^{enc} h_{t-1}^{enc} + b_h^{enc}
$$

$$
h_t^{enc} = \tanh(a_t^{enc})
$$

### 3.2 Decoder

$$
a_t^{dec} = w_{xh}^{dec} x_t^{dec} + w_{hh}^{dec} h_{t-1}^{dec} + b_h^{dec}
$$

$$
h_t^{dec} = \tanh(a_t^{dec})
$$

### 3.3 Output layer

因為 hidden 只有 1 個值，所以每個類別的 logit 都只是：

$$
o_{t,k} = w_k \cdot h_t^{dec} + b_k
$$

再用 softmax 變成機率：

$$
\hat{y}_{t,k} = \frac{e^{o_{t,k}}}{e^{o_{t,A}} + e^{o_{t,B}} + e^{o_{t,C}}}
$$

### 3.4 Loss

每一步用 cross entropy：

$$
\mathcal{L}_t = -\log \hat{y}_t[y_t^*]
$$

總 loss 取平均：

$$
\mathcal{L} = \frac{1}{2}(\mathcal{L}_1 + \mathcal{L}_2)
$$

---

## 4. Encoder 前向傳播

### 4.1 參數設定

我們故意選簡單數字：

$$
w_{xh}^{enc} = 0.8, \quad w_{hh}^{enc} = 0.5, \quad b_h^{enc} = 0.1
$$

因為輸入是 one-hot，我們把 A、B、C 先對應成簡單數值：

$$
A \rightarrow 1, \quad B \rightarrow -1, \quad C \rightarrow 0.5
$$

初始 hidden：

$$
h_0^{enc} = 0
$$

### 4.2 Step 1：讀入 A

A 對應數值是 $1$。

$$
a_1^{enc} = 0.8(1) + 0.5(0) + 0.1 = 0.9
$$

$$
h_1^{enc} = \tanh(0.9) \approx 0.716
$$

### 4.3 Step 2：讀入 B

B 對應數值是 $-1$。

$$
a_2^{enc} = 0.8(-1) + 0.5(0.716) + 0.1
$$

$$
= -0.8 + 0.358 + 0.1 = -0.342
$$

$$
h_2^{enc} = \tanh(-0.342) \approx -0.329
$$

這個最後 hidden state 會交給 decoder：

$$
h_0^{dec} = h_2^{enc} \approx -0.329
$$

> 這就是 Seq2Seq 的橋梁：**encoder 的最後狀態，當作 decoder 的起點。**

---

## 5. Decoder 前向傳播

### 5.1 Decoder 參數

$$
w_{xh}^{dec} = 0.7, \quad w_{hh}^{dec} = 0.6, \quad b_h^{dec} = 0.05
$$

decoder 的第一步輸入先餵一個起始符號 `<SOS>`，把它對應成數值 $0.5$。

另外，輸出層三個類別的參數設定為：

$$
(w_A, b_A) = (0.2, 0.0)
$$

$$
(w_B, b_B) = (0.6, 0.1)
$$

$$
(w_C, b_C) = (-0.4, 0.05)
$$

---

### 5.2 Decoder Step 1

輸入 `<SOS>`，也就是 $x_1^{dec} = 0.5$。

$$
a_1^{dec} = 0.7(0.5) + 0.6(-0.329) + 0.05
$$

$$
= 0.35 - 0.1974 + 0.05 = 0.2026
$$

$$
h_1^{dec} = \tanh(0.2026) \approx 0.200
$$

接著算三個類別的 logit：

$$
o_{1,A} = 0.2(0.200) + 0 = 0.040
$$

$$
o_{1,B} = 0.6(0.200) + 0.1 = 0.220
$$

$$
o_{1,C} = -0.4(0.200) + 0.05 = -0.030
$$

做 softmax（此處取近似值）：

$$
\hat{y}_1 \approx [0.319, 0.382, 0.299]
$$

也就是：

- 預測 A 的機率：0.319
- 預測 B 的機率：0.382
- 預測 C 的機率：0.299

而第 1 步正確答案是 **B**。

---

### 5.3 Decoder Step 2

這裡用 teacher forcing，餵入正確答案 B。B 對應數值是 $-1$。

$$
a_2^{dec} = 0.7(-1) + 0.6(0.200) + 0.05
$$

$$
= -0.7 + 0.12 + 0.05 = -0.53
$$

$$
h_2^{dec} = \tanh(-0.53) \approx -0.485
$$

三個類別的 logit：

$$
o_{2,A} = 0.2(-0.485) = -0.097
$$

$$
o_{2,B} = 0.6(-0.485) + 0.1 = -0.191
$$

$$
o_{2,C} = -0.4(-0.485) + 0.05 = 0.244
$$

softmax 近似得到：

$$
\hat{y}_2 \approx [0.302, 0.275, 0.423]
$$

第 2 步正確答案是 **C**。

---

## 6. Loss 計算

### 6.1 第 1 步

目標是 B，所以用預測中的第 2 個機率：

$$
\mathcal{L}_1 = -\log(0.382) \approx 0.962
$$

### 6.2 第 2 步

目標是 C，所以用第 3 個機率：

$$
\mathcal{L}_2 = -\log(0.423) \approx 0.861
$$

### 6.3 平均 loss

$$
\mathcal{L} = \frac{1}{2}(0.962 + 0.861) = 0.9115
$$

---

## 7. Output Layer 反向傳播

這一段最重要的公式是：

$$
\delta_t^o = \frac{1}{2}(\hat{y}_t - y_t^*)
$$

前面的 $\frac{1}{2}$ 來自總 loss 是兩步平均。

---

### 7.1 Step 1 的輸出誤差

第 1 步：

$$
\hat{y}_1 \approx [0.319, 0.382, 0.299]
$$

正確 one-hot 是：

$$
y_1^* = [0,1,0]
$$

所以：

$$
\delta_1^o = \frac{1}{2}([0.319,0.382,0.299] - [0,1,0])
$$

$$
= [0.1595, -0.309, 0.1495]
$$

---

### 7.2 Step 2 的輸出誤差

第 2 步：

$$
\hat{y}_2 \approx [0.302, 0.275, 0.423]
$$

正確 one-hot 是：

$$
y_2^* = [0,0,1]
$$

所以：

$$
\delta_2^o = \frac{1}{2}([0.302,0.275,0.423] - [0,0,1])
$$

$$
= [0.151, 0.1375, -0.2885]
$$

---

### 7.3 對輸出層權重的梯度

因為每個類別的 logit 是

$$
o_{t,k} = w_k h_t^{dec} + b_k
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial w_k}^{(t)} = \delta_{t,k}^o \cdot h_t^{dec}
$$

#### Step 1（$h_1^{dec} \approx 0.200$）

$$
\frac{\partial \mathcal{L}}{\partial w_A}^{(1)} = 0.1595 \times 0.200 = 0.0319
$$

$$
\frac{\partial \mathcal{L}}{\partial w_B}^{(1)} = -0.309 \times 0.200 = -0.0618
$$

$$
\frac{\partial \mathcal{L}}{\partial w_C}^{(1)} = 0.1495 \times 0.200 = 0.0299
$$

#### Step 2（$h_2^{dec} \approx -0.485$）

$$
\frac{\partial \mathcal{L}}{\partial w_A}^{(2)} = 0.151 \times (-0.485) \approx -0.0732
$$

$$
\frac{\partial \mathcal{L}}{\partial w_B}^{(2)} = 0.1375 \times (-0.485) \approx -0.0667
$$

$$
\frac{\partial \mathcal{L}}{\partial w_C}^{(2)} = -0.2885 \times (-0.485) \approx 0.1399
$$

#### Total

$$
\frac{\partial \mathcal{L}}{\partial w_A} \approx 0.0319 - 0.0732 = -0.0413
$$

$$
\frac{\partial \mathcal{L}}{\partial w_B} \approx -0.0618 - 0.0667 = -0.1285
$$

$$
\frac{\partial \mathcal{L}}{\partial w_C} \approx 0.0299 + 0.1399 = 0.1698
$$

偏差梯度更簡單，就是把兩步的對應誤差加起來。

例如：

$$
\frac{\partial \mathcal{L}}{\partial b_B} = -0.309 + 0.1375 = -0.1715
$$

---

## 8. Decoder Hidden 反向傳播

輸出層的誤差，會先傳回 decoder hidden。

因為每個 logit 都依賴 $h_t^{dec}$，所以：

$$
\delta_t^h = w_A \delta_{t,A}^o + w_B \delta_{t,B}^o + w_C \delta_{t,C}^o
$$

這裡用目前的輸出權重：

$$
w_A = 0.2, \quad w_B = 0.6, \quad w_C = -0.4
$$

### 8.1 Step 2 先往回傳

$$
\delta_2^h = 0.2(0.151) + 0.6(0.1375) + (-0.4)(-0.2885)
$$

$$
= 0.0302 + 0.0825 + 0.1154 = 0.2281
$$

因為 hidden 經過 $\tanh$，所以還要乘導數：

$$
\delta_2^a = \delta_2^h (1 - (h_2^{dec})^2)
$$

$$
= 0.2281(1 - 0.485^2)
$$

$$
= 0.2281(1 - 0.2352) = 0.2281(0.7648) \approx 0.1745
$$

---

### 8.2 Step 1 要加上未來時間的影響

第 1 步不只影響自己那一步的輸出，還會透過 recurrent connection 影響第 2 步。

先算它自己這一步從輸出層收到的誤差：

$$
\delta_{1,\text{local}}^h = 0.2(0.1595) + 0.6(-0.309) + (-0.4)(0.1495)
$$

$$
= 0.0319 - 0.1854 - 0.0598 = -0.2133
$$

再加上來自未來 step 2 的誤差：

$$
\delta_{1,\text{future}}^h = w_{hh}^{dec} \cdot \delta_2^a = 0.6 \times 0.1745 = 0.1047
$$

所以 step 1 hidden 的總誤差是：

$$
\delta_1^h = -0.2133 + 0.1047 = -0.1086
$$

接著經過 $\tanh$ 的導數：

$$
\delta_1^a = \delta_1^h (1 - (h_1^{dec})^2)
$$

$$
= -0.1086(1 - 0.200^2)
$$

$$
= -0.1086(0.96) \approx -0.1043
$$

---

### 8.3 Decoder recurrent 權重梯度

decoder 的 recurrent 權重是 $w_{hh}^{dec}$，其梯度來自每一步：

$$
\frac{\partial \mathcal{L}}{\partial w_{hh}^{dec}}^{(t)} = \delta_t^a \cdot h_{t-1}^{dec}
$$

#### Step 2

$$
\frac{\partial \mathcal{L}}{\partial w_{hh}^{dec}}^{(2)} = 0.1745 \times 0.200 = 0.0349
$$

#### Step 1

因為 $h_0^{dec} = h_2^{enc} \approx -0.329$

$$
\frac{\partial \mathcal{L}}{\partial w_{hh}^{dec}}^{(1)} = -0.1043 \times (-0.329) \approx 0.0343
$$

#### Total

$$
\frac{\partial \mathcal{L}}{\partial w_{hh}^{dec}} \approx 0.0349 + 0.0343 = 0.0692
$$

---

## 9. 梯度如何傳回 Encoder

這是 Seq2Seq 最重要的一句話：

$$
h_0^{dec} = h_2^{enc}
$$

所以 decoder 第 1 步對初始 hidden 的梯度，會直接傳給 encoder 最後一步。

### 9.1 先看 decoder 傳回來多少

由於

$$
a_1^{dec} = w_{xh}^{dec}x_1^{dec} + w_{hh}^{dec}h_0^{dec} + b_h^{dec}
$$

所以：

$$
\delta_{enc,2}^h = w_{hh}^{dec} \cdot \delta_1^a = 0.6 \times (-0.1043) = -0.0626
$$

這就是傳到 encoder 最後 hidden 的誤差。

---

### 9.2 Encoder Step 2

encoder 第 2 步也有 $\tanh$，所以：

$$
\delta_2^{a,enc} = \delta_{enc,2}^h (1 - (h_2^{enc})^2)
$$

$$
= -0.0626(1 - 0.329^2)
$$

$$
= -0.0626(1 - 0.1082)
$$

$$
= -0.0626(0.8918) \approx -0.0558
$$

---

### 9.3 Encoder Step 1

再往前傳一格：

$$
\delta_1^{h,enc} = w_{hh}^{enc} \cdot \delta_2^{a,enc}
$$

$$
= 0.5 \times (-0.0558) = -0.0279
$$

再乘上 $\tanh$ 導數：

$$
\delta_1^{a,enc} = -0.0279(1 - 0.716^2)
$$

$$
= -0.0279(1 - 0.5127)
$$

$$
= -0.0279(0.4873) \approx -0.0136
$$

---

### 9.4 一個 encoder 權重梯度示意

例如對 $w_{hh}^{enc}$ 而言：

$$
\frac{\partial \mathcal{L}}{\partial w_{hh}^{enc}}^{(2)} = \delta_2^{a,enc} \cdot h_1^{enc}
$$

$$
= -0.0558 \times 0.716 \approx -0.0399
$$

這表示：

- loss 明明只發生在 decoder 的輸出
- 但梯度仍然會透過橋梁傳回 encoder

這就是 Seq2Seq 可訓練的根本原因。

---

## 10. 參數更新示意

假設學習率：

$$
\alpha = 0.1
$$

以前面算出的輸出層某個權重為例：

$$
\frac{\partial \mathcal{L}}{\partial w_B} \approx -0.1285
$$

原本：

$$
w_B = 0.6
$$

更新後：

$$
w_B \leftarrow 0.6 - 0.1(-0.1285) = 0.61285
$$

因為梯度是負的，所以更新後 $w_B$ 變大了。

這樣做的直覺是：

- 模型在第 1 步應該更傾向輸出 B
- 所以把有助於 B 的權重稍微拉高

---

## 11. 一句話總結

$$
\text{loss 雖然出現在 decoder，但梯度會穿過 decoder，跨過橋梁，再回到 encoder。}
$$

更口語地說就是：

> **Seq2Seq 的學習，不是只有輸出端在學；整個 encoder-decoder 都一起被同一個 loss 推著調整。**

---

## 最後整理：這個數值例子想讓你看到什麼？

1. Encoder 先把輸入序列壓成最後狀態
2. Decoder 以這個最後狀態作為起點開始輸出
3. 每一步輸出都會產生 loss
4. loss 先傳回 output layer
5. 再傳回 decoder hidden
6. 再沿著時間往前傳
7. 最後跨回 encoder

如果只記一件事，請記住：

$$
\boxed{h_0^{dec} = h_{T_x}^{enc}}
$$

這一條等號，就是 Seq2Seq 裡 encoder 和 decoder 真正接起來的地方。
