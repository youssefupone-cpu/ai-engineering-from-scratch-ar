# الاهتمام بالذات من الصفر
> الانتباه عبارة عن جدول بحث حيث تسأل كل كلمة "من يهمني؟" - ويتعلم الجواب.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 3 (التعلم العميق الأساسي)، المرحلة 5 الدرس 10 (تسلسل إلى تسلسل)
**الوقت:** ~90 دقيقة
## أهداف التعلم
- تنفيذ الاهتمام الذاتي بالمنتج النقطي من البداية باستخدام NumPy فقط، بما في ذلك توقعات الاستعلام/المفتاح/القيمة والمجموع المرجح softmax
- قم ببناء طبقة انتباه متعددة الرؤوس تعمل على تقسيم الرؤوس، وحساب الاهتمام الموازي، وتسلسل النتائج
- تتبع كيفية التقاط مصفوفة الاهتمام للعلاقات الرمزية وشرح سبب منع القياس بواسطة sqrt(d_k) من تشبع softmax
- تطبيق الإخفاء السببي لتحويل الانتباه ثنائي الاتجاه إلى انتباه انحداري ذاتي (نمط وحدة فك التشفير).
## المشكلة
تعالج RNNs تسلسلات رمزية واحدة في كل مرة. بحلول الوقت الذي تصل فيه إلى الرمز المميز 50، يكون قد تم ضغط المعلومات من الرمز المميز 1 من خلال 50 خطوة ضغط. يتم سحق التبعيات طويلة المدى إلى حالة مخفية ذات حجم ثابت - وهو عنق الزجاجة الذي لا يمكن لأي قدر من LSTM حله بالكامل.
أظهرت ورقة اهتمام Bahdanau لعام 2014 الحل: دع وحدة فك التشفير تنظر إلى كل موضع من أوضاع التشفير وتقرر أي منها يهم في الخطوة الحالية. ولكن لا يزال مثبتًا على RNN. طرحت ورقة "الانتباه هو كل ما تحتاجه" لعام 2017 سؤالاً أكثر وضوحًا: ماذا لو كان الاهتمام هو الآلية *الوحيدة*؟ لا تكرار. لا التواء. مجرد الاهتمام.
يتيح الاهتمام الذاتي لكل موضع في التسلسل أن يواكب كل موضع آخر في خطوة واحدة متوازية. هذا هو ما تتميز به محولات makes السريعة والقابلة للتطوير والمهيمنة.
##المفهوم
### تشبيه البحث في قاعدة البيانات
فكر في الاهتمام باعتباره بحثًا بسيطًا في قاعدة البيانات:
```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

يولد كل رمز ثلاثة نواقل:
- **استعلام (س)**: "ما الذي أبحث عنه؟"
- **المفتاح (ك)**: "ماذا أحتوي؟"
- **القيمة (V)**: "ما هي المعلومات التي سأقدمها إذا تم تحديدها؟"
المنتج النقطي بين الاستعلام وجميع المفاتيح ينتج عنه درجات الاهتمام. الدرجة العالية تعني أن "هذا المفتاح يطابق استعلامي". تلك الدرجات تزن القيم. الناتج هو مجموع مرجح للقيم.
### حساب Q، K، V
يتم عرض كل تضمين للرمز المميز من خلال ثلاث مصفوفات للوزن المستفادة:
```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

بصريا، لرمز واحد:
```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### مصفوفة الاهتمام
بمجرد حصولك على Q وK وV لجميع الرموز المميزة، تشكل درجات الانتباه مصفوفة:
```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### لماذا المقياس؟
تنمو المنتجات النقطية بالبعد dk. إذا كان dk = 64، يمكن أن تكون المنتجات النقطية في نطاق العشرات، مما يدفع softmax إلى المناطق التي تختفي فيها التدرجات. الإصلاح: القسمة على sqrt(dk).
```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

يؤدي هذا إلى الاحتفاظ بالقيم في نطاق ينتج فيه softmax تدرجات مفيدة.
### Softmax يحول النتائج إلى أوزان
تقوم Softmax بتحويل الدرجات الأولية إلى توزيع احتمالي عبر كل صف:
```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

الآن يحتوي كل رمز مميز على مجموعة من الأوزان توضح مقدار الاهتمام بكل رمز مميز آخر.
### المجموع المرجح للقيم
الناتج النهائي لكل رمز هو مجموع مرجح لجميع متجهات القيمة:
```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### خط أنابيب كامل
```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

الصيغة في سطر واحد:
```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## بنائها
### الخطوة 1: سوفت ماكس من الصفر
يقوم Softmax بتحويل logits الخام إلى احتمالات. اطرح الحد الأقصى لتحقيق الاستقرار العددي.
```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### الخطوة الثانية: زيادة الاهتمام بالمنتج النقطي
الوظيفة الأساسية. يأخذ مصفوفات Q وK وV ويعيد مخرجات الانتباه بالإضافة إلى مصفوفة الوزن.
```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### الخطوة 3: فصل الاهتمام الذاتي مع التوقعات المستفادة
وحدة اهتمام ذاتي كاملة مع مصفوفات وزن Wq وWk وWv تمت تهيئتها باستخدام مقياس يشبه Xavier.
```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### الخطوة 4: قم بتشغيل الجملة
قم بإنشاء تضمينات وهمية لجملة وشاهد أوزان الانتباه.
```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### الخطوة 5: تصور الانتباه باستخدام الخريطة الحرارية ASCII
قم بتعيين أوزان الانتباه إلى الشخصيات للحصول على رؤية سريعة.
```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## استخدمه
يقوم `nn.MultiheadAttention` الخاص بـ PyTorch بتنفيذ ما أنشأناه بالضبط، بالإضافة إلى التقسيم متعدد الرؤوس وإسقاط الإخراج:
```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

الفرق الرئيسي: يقوم الاهتمام متعدد الرؤوس بتشغيل وظائف انتباه متعددة بالتوازي، ولكل منها إسقاطات Q وK وV الخاصة بها بالحجم dk = d_model / n_heads، ثم يقوم بتسلسل النتائج. يتيح ذلك للنموذج الاهتمام بأنواع العلاقات المختلفة في وقت واحد.
## اشحنها
ينتج هذا الدرس:
- `outputs/prompt-attention-explainer.md` - مطالبة لشرح الاهتمام من خلال تشبيه البحث في قاعدة البيانات
## تمارين
1. قم بتعديل `scaled_dot_product_attention` لقبول مصفوفة قناع اختيارية تقوم بتعيين مواضع معينة على اللانهاية السالبة قبل softmax (هذه هي الطريقة التي يعمل بها الإخفاء السببي/وحدة فك التشفير)
2. تنفيذ الانتباه متعدد الرؤوس من البداية: قم بتقسيم Q وK وV إلى أجزاء `n_heads`، وقم بتشغيل الانتباه على كل منها، ثم قم بتسلسلها وعرضها من خلال مصفوفة الوزن النهائية Wo
3. خذ جملتين مختلفتين بنفس الطول، وقم بتغذيتهما من خلال نفس نموذج الانتباه الذاتي، وقارن بين أنماط انتباههما. ما التغييرات؟ ما الذي يبقى على حاله؟
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| استعلام (س) | "ناقل السؤال" | إسقاط مكتسب للمدخلات التي تمثل المعلومات التي يبحث عنها هذا الرمز المميز |
| المفتاح (ك) | "ناقل التسمية" | إسقاط مكتسب يمثل المعلومات التي يحتوي عليها هذا الرمز المميز، ومطابقتها للاستعلامات |
| القيمة (الخامس) | "ناقل المحتوى" | إسقاط مكتسب يحمل المعلومات الفعلية التي يتم تجميعها بناءً على درجات الاهتمام |
| اهتمام متدرج بالمنتج النقطي | "صيغة الاهتمام" | softmax(QK^T / sqrt(dk)) @ V - يمنع القياس تشبع softmax في الأبعاد العالية |
| الاهتمام بالنفس | "الرمز ينظر إلى نفسه وإلى غيره" | انتبه إلى أن Q وK وV كلها تأتي من نفس التسلسل، مما يسمح لكل موضع بالحضور إلى كل موضع آخر |
| اوزان الانتباه | "كم التركيز" | توزيع احتمالي على المواضع، تم إنتاجه بواسطة softmax على منتجات نقطية متدرجة |
| اهتمام متعدد الرؤوس | "الاهتمام الموازي" | تشغيل وظائف انتباه متعددة بإسقاطات مختلفة، ثم تسلسل النتائج للحصول على تمثيلات أكثر ثراءً |
## مزيد من القراءة
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - ورق المحولات الأصلي
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - أفضل جولة مرئية للبنية الكاملة
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - تنفيذ PyTorch سطرًا بسطر مع التوضيحات