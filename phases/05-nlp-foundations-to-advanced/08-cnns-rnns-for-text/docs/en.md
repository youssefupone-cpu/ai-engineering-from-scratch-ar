# CNNs وRNNs للنص
> تتعلم التلافيفات n-gram. تذكر التكرارات. كلاهما يحل محله الاهتمام. كلاهما لا يزال مهمًا على الأجهزة المقيدة.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 3 · 11 (PyTorch مقدمة)، المرحلة 5 · 03 (تضمين الكلمات)، المرحلة 4 · 02 (الالتفافات من الصفر)
**الوقت:** ~75 دقيقة
## المشكلة
TF-IDF وأنتج Word2Vec متجهات مسطحة تتجاهل ترتيب الكلمات. لم يتمكن المصنف المبني عليها من التمييز بين `dog bites man` و`man bites dog`. ترتيب الكلمات يحمل الإشارة في بعض الأحيان.
قامت عائلتين من الهندسة المعمارية بملء هذه الفجوة قبل وصول المحولات.
** شبكات تلافيفية للنص (TextCNN). ** تطبيق لفات أحادية الأبعاد على تسلسلات تضمين الكلمات. يعد مرشح العرض 3 بمثابة كاشف ثلاثي الأبعاد قابل للتعلم: فهو يمتد إلى ثلاث كلمات ويخرج النتيجة. قم بتجميع عروض مختلفة (2، 3، 4، 5) لاكتشاف الأنماط متعددة المقاييس. الحد الأقصى للتجميع إلى تمثيل ذو حجم ثابت. مسطحة، موازية، سريعة.
**الشبكات المتكررة (RNN، LSTM، GRU).** معالجة الرموز المميزة واحدًا تلو الآخر، مع الحفاظ على الحالة المخفية التي تنقل المعلومات إلى الأمام. أطوال إدخال متسلسلة ومرنة وحاملة للذاكرة. سيطرت نمذجة التسلسل من عام 2014 إلى عام 2017، ثم حدث الاهتمام.
يبني هذا الدرس كليهما، ثم يسمي الفشل الذي حفز الاهتمام.
##المفهوم
**TextCNN** (كيم، 2014). يتم تضمين الرموز المميزة. يعمل الالتفاف ذو العرض `k` 1D على تحريك المرشح عبر `k`-جرام متتالية من التضمينات، مما يؤدي إلى إنتاج خريطة المعالم. يختار التجميع الأقصى العالمي على تلك الخريطة أقوى عملية تنشيط. قم بتسلسل المخرجات المجمعة القصوى من عروض مرشح متعددة. تغذية لرأس المصنف.
لماذا يعمل. المرشح هو n-gram قابل للتعلم. الحد الأقصى للتجميع لا يتغير في الموضع، لذا فإن "ليس جيدًا" يطلق نفس الميزة في بداية المراجعة أو في منتصفها. ثلاثة عروض للمرشحات مع 100 مرشح لكل منها تمنحك 300 كاشف n-gram متعلم. التدريب موازي. لا التبعية التسلسلية.
**RNN.** في كل خطوة زمنية `t`، الحالة المخفية `h_t = f(W * x_t + U * h_{t-1} + b)`. شارك `W`، `U`، `b` عبر الزمن. الحالة المخفية في الوقت `T` هي ملخص للبادئة بأكملها. بالنسبة للتصنيف، قم بالتجميع عبر `h_1 ... h_T` (الحد الأقصى أو المتوسط ​​أو الأخير).
تعاني شبكات RNN البسيطة من تدرجات التلاشي. يضيف **LSTM** بوابات تقرر ما يجب نسيانه، وما يجب تخزينه، وما يجب إخراجه، مما يؤدي إلى تثبيت التدرجات من خلال تسلسلات طويلة. **GRU** يبسط LSTM إلى بوابتين؛ يؤدي بالمثل مع عدد أقل من المعلمات.
**شبكات RNN ثنائية الاتجاه** تعمل على تشغيل RNN للأمام وأخرى للخلف، وتسلسل الحالات المخفية. يرى تمثيل كل رمز مميز السياق الأيسر والأيمن. ضروري لوضع العلامات على المهام.
## بنائها
### الخطوة 1: إرسال رسالة نصية إلى CNN في PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

يقوم `transpose(1, 2)` بإعادة تشكيل `[batch, seq_len, embed_dim]` إلى `[batch, embed_dim, seq_len]` لأن `nn.Conv1d` يتعامل مع المحور الأوسط كقنوات. يكون حجم الإخراج المجمع ثابتًا بغض النظر عن طول الإدخال.
### الخطوة الثانية: المصنف LSTM
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

الحد الأقصى للتجميع فوق التسلسل، وليس تجمع الحالة الأخيرة. بالنسبة للتصنيف، عادةً ما يتفوق التجميع الأقصى على الحالة المخفية الأخيرة لأن المعلومات الموجودة في نهاية التسلسل الطويل تميل إلى السيطرة على الحالة الأخيرة.
### الخطوة 3: عرض التدرج المتلاشي (الحدس)
لا يمكن لـ RNN العادي بدون بوابة أن يتعلم التبعيات طويلة المدى. فكر في مهمة لعبة: توقع ما إذا كان الرمز المميز `A` قد ظهر في أي مكان بالتسلسل. إذا كان `A` في الموضع 1 وكان طول التسلسل 100 رمزًا مميزًا، فيجب أن يتدفق التدرج من الخسارة مرة أخرى خلال 99 ضعفًا للوزن المتكرر. إذا كان الوزن أقل من 1، يختفي التدرج. إذا كان أكثر من 1، فإنه ينفجر.
```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

تعمل LSTMs على إصلاح ذلك من خلال **حالة الخلية** التي تعمل عبر الشبكة مع التفاعلات الإضافية فقط (تقوم بوابة النسيان بقياسها بشكل مضاعف، لكن التدرجات لا تزال تتدفق على طول "الطريق السريع"). تفعل وحدات GRU شيئًا مشابهًا مع عدد أقل من المعلمات. كلاهما يمنحك تدريبًا ثابتًا من خلال أكثر من 100 خطوة متسلسلة.
### الخطوة 4: لماذا لا يزال هذا غير كاف
استمرت ثلاث مشاكل حتى مع LSTMs.
1. **عنق الزجاجة المتسلسل.** يتطلب تدريب RNN على تسلسل بطول 1000 1000 خطوة تسلسلية للأمام/للخلف. لا يمكن موازاة عبر الزمن.
2. **ناقل سياق ذو حجم ثابت في إعدادات برنامج التشفير وفك التشفير.** يرى جهاز فك التشفير فقط الحالة المخفية النهائية لجهاز التشفير، مضغوطًا على الإدخال بأكمله. المدخلات الطويلة تفقد التفاصيل. الدرس 09 يغطي هذا مباشرة.
3. **سقف دقة التبعية البعيدة.** تتفوق LSTMs على شبكات RNN العادية ولكنها لا تزال تواجه صعوبة في نشر معلومات محددة عبر أكثر من 200 خطوة.
الاهتمام حل كل ثلاثة. أسقطت المحولات التكرار تمامًا. الدرس 10 هو المحور.
## استخدمه
PyTorch's `nn.LSTM` و`nn.GRU` و`nn.Conv1d` جاهزة للإنتاج. رمز التدريب هو المعيار.
Hugging Face يشحن التضمينات المدربة مسبقًا التي تقوم بتوصيلها كطبقة الإدخال:
```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

استخدم قائمة التحقق عندما تناسب القيد.
- **استدلال الحافة / على الجهاز.** TextCNN مع تضمينات GloVe أصغر بمقدار 10-100 مرة من المحول. إذا كان هدف النشر الخاص بك هو الهاتف، فهذا هو المكدس.
- **البث / التصنيف عبر الإنترنت.** يعالج RNN رمزًا مميزًا واحدًا في كل مرة؛ المحولات تحتاج إلى التسلسل الكامل. بالنسبة للنص الوارد في الوقت الفعلي، لا تزال LSTMs تفوز.
- **نماذج صغيرة لخطوط الأساس.** تكرار سريع لمهمة جديدة. قم بتدريب TextCNN في 5 دقائق على CPU.
- **تصنيف التسلسل ببيانات محدودة.** لا يزال BiLSTM-CRF (الدرس 06) عبارة عن بنية NER على مستوى الإنتاج للجمل ذات التصنيف 1k-10k.
كل شيء آخر يذهب إلى المحولات.
## اشحنها
حفظ باسم `outputs/prompt-text-encoder-picker.md`:
```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## تمارين
1. **سهل.** قم بتدريب TextCNN على مجموعة بيانات لعبة من 3 فئات (أنت تخترع البيانات). تحقق من أن عروض المرشح (2، 3، 4) تتفوق على العرض الفردي (3) في المتوسط ​​F1.
2. **متوسط.** قم بتنفيذ تجميع الحد الأقصى، وتجميع المتوسط، وتجميع الحالة الأخيرة للمصنف LSTM. قارن على مجموعة بيانات صغيرة؛ قم بتوثيق أي تجميع يفوز وافترض السبب.
3. **صعب.** أنشئ علامة تمييز BiLSTM-CRF NER (اجمع بين الدرس 06 وهذا الدرس). تدريب على CoNLL-2003. قارن مع خط الأساس CRF وحده من الدرس 06 ومع BERT المضبوط. قم بالإبلاغ عن وقت التدريب والذاكرة وF1.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| نص سي ان ان | CNN للنص | كومة من التلافيفات أحادية الأبعاد عبر تضمينات الكلمات مع الحد الأقصى العالمي. كيم (2014). |
| __المصطلح_2__ | صافي المتكررة | يتم تحديث الحالة المخفية في كل خطوة زمنية: `h_t = f(W x_t + U h_{t-1})`. |
| __المصطلح_3__ | بوابات RNN | يضيف بوابات الإدخال/النسيان/الإخراج + حالة الخلية. القطارات بثبات من خلال تسلسلات طويلة. |
| __المصطلح_5__ | أبسط LSTM | بوابتين بدلا من ثلاثة. دقة مماثلة، معلمات أقل. |
| ثنائي الاتجاه | كلا الاتجاهين | للأمام + للخلف RNN متسلسل. كل رمز يرى كلا الجانبين من سياقه. |
| التلاشي التدرج | تموت إشارة التدريب | الضرب المتكرر بأوزان أقل من 1 في تدرجات الخطوة المبكرة لـ RNNs makes هي صفر فعليًا. |
## مزيد من القراءة
- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — ورقة TextCNN. ثماني صفحات. قابلة للقراءة.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — الورقة LSTM. واضح بشكل غير متوقع.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — المخططات التي جعلت LSTMs في متناول الجميع.