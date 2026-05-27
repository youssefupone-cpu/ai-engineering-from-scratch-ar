# Sequence-to-Sequence Models

> اثنان من RNN يتظاهران بأنهما مترجمان. إن عنق الزجاجة الذي وصلوا إليه هو سبب وجود الاهتمام.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 08 (شبكات CNN + RNNs للنص)، المرحلة 3 · 11 (PyTorch مقدمة)
**الوقت:** ~75 دقيقة

## The Problem

يقوم التصنيف بتعيين تسلسل متغير الطول إلى تسمية واحدة. تقوم الترجمة بتعيين تسلسل متغير الطول إلى تسلسل آخر متغير الطول. المدخلات والمخرجات تعيش في مفردات مختلفة، وربما لغات مختلفة، مع عدم وجود ضمان لتكافؤ الطول.

قامت الهندسة المعمارية seq2seq (Sutskever، Vinyals، Le، 2014) بحل هذه المشكلة بوصفة بسيطة متعمدة. اثنان من شبكات RNN. يقرأ المرء الجملة المصدر وينتج ناقل سياق ذي حجم ثابت. يقرأ الآخر هذا المتجه وينشئ رمز الجملة المستهدف برمز مميز. نفس الكود الذي كتبته للدرس 08، تم لصقه معًا بشكل مختلف.

وهذا يستحق الدراسة لسببين. أولاً، يعتبر عنق الزجاجة بين ناقل السياق هو الفشل الأكثر فائدة من الناحية التربوية في NLP. إنه يحفز كل ما يجيده الاهتمام والمحولات. ثانيًا، لا تزال وصفة التدريب (إجبار المعلم، أخذ العينات المجدولة، بحث الشعاع عند الاستدلال) تنطبق على كل نظام جيل حديث بما في ذلك LLMs.

## The Concept

**التشفير.** RNN يقرأ الجملة المصدر. الحالة المخفية النهائية هي **ناقل السياق** — وهو ملخص ذو حجم ثابت للإدخال بأكمله. من المفترض أن لا تفقد شيئًا سوى المصدر.

** وحدة فك الترميز. ** تمت تهيئة RNN آخر من ناقل السياق. في كل خطوة، يأخذ الرمز المميز الذي تم إنشاؤه مسبقًا كمدخل وينتج توزيعًا على المفردات المستهدفة. عينة أو argmax لاختيار الرمز المميز التالي. قم بإعادته مرة أخرى. كرر ذلك حتى يتم إنتاج الرمز المميز `<EOS>` أو الوصول إلى الحد الأقصى للطول.

**التدريب:** فقدان الإنتروبيا المتقاطعة في كل خطوة من خطوات وحدة فك التشفير، ويتم تلخيصها عبر التسلسل. دعامة خلفية قياسية عبر الزمن من خلال كلا الشبكتين.

**إجبار المعلم.** أثناء التدريب، يكون إدخال وحدة فك التشفير في الخطوة `t` هو رمز *الحقيقة الأرضية* في الموضع `t-1`، وليس التنبؤ السابق لجهاز فك التشفير. وهذا يؤدي إلى استقرار التدريب. وبدون ذلك، تتوالى الأخطاء المبكرة ولا يتعلم النموذج أبدًا. عند الاستدلال، يجب عليك استخدام التنبؤات الخاصة بالنموذج، لذلك توجد دائمًا فجوة توزيع القطار/الاستدلال. تسمى هذه الفجوة **تحيز التعرض**.

**عنق الزجاجة.** يجب ضغط كل ما تعلمه برنامج التشفير عن المصدر في ناقل السياق الواحد هذا. الجمل الطويلة تفقد التفاصيل. الكلمات النادرة تصبح غير واضحة. يجب حفظ عملية إعادة الترتيب (chat noir vs. black cat)، وليس حسابها.

انتبه (الدرس 10) إلى إصلاح هذه المشكلة من خلال السماح لوحدة فك التشفير بالاطلاع على *كل* حالة مخفية لجهاز التشفير، وليس فقط الحالة الأخيرة. هذا هو الملعب كله.

## Build It

### Step 1: an encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` له شكل `[batch, seq_len, hidden_dim]` — حالة مخفية واحدة لكل موضع إدخال. `hidden` له شكل `[1, batch, hidden_dim]` — الخطوة الأخيرة. قال الدرس 08 "تجميع المخرجات للتصنيف." هنا نحتفظ بالحالة المخفية الأخيرة كمتجه السياق، ونتجاهل مخرجات كل خطوة.

### Step 2: a decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

يُطلق على وحدة فك التشفير خطوة واحدة في كل مرة. الإدخال: مجموعة من الرموز الفردية والحالة المخفية الحالية. الإخراج: المفردات logits للرمز التالي والحالة المخفية المحدثة.

### Step 3: training loop with teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

مقبضان يستحقان التسمية. `ignore_index=0` يتخطى الخسارة في رموز الحشو. `teacher_forcing_ratio` هو احتمال استخدام الرمز الحقيقي مقابل تنبؤ النموذج في كل خطوة. ابدأ عند 1.0 (إجبار المعلم بالكامل) ثم واصل الانخفاض إلى ~0.5 خلال التدريب لسد فجوة التعرض والتحيز.

### Step 4: inference loop (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

يقوم فك التشفير الجشع باختيار الرمز المميز ذو الاحتمالية الأعلى في كل خطوة. يمكن أن تتجول: بمجرد الالتزام برمز مميز، لا يمكنك التراجع عنه. **بحث الشعاع** يحافظ على التسلسلات الجزئية العلوية `k` حية ويختار التسلسل الكامل الحاصل على أعلى الدرجات في النهاية. عرض الشعاع 3-5 قياسي.

### Step 5: the bottleneck, demonstrated

تدريب النموذج على مهمة نسخ لعبة: المصدر `[a, b, c, d, e]`، الهدف `[a, b, c, d, e]`. زيادة طول التسلسل. مراقبة الدقة.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

لا يمكن لحالة مخفية GRU واحدة أن تحفظ إدخالاً مكونًا من 40 رمزًا دون خسارة. المعلومات موجودة في كل خطوة من خطوات التشفير، لكن وحدة فك التشفير ترى الحالة الأخيرة فقط. الاهتمام بإصلاح هذا مباشرة.

## Use It

يحتوي PyTorch على قوالب seq2seq المستندة إلى `nn.Transformer` و `nn.LSTM`. توفر مكتبة Hugging Face `transformers` نماذج كاملة لوحدة فك التشفير (BART، T5، mBART، NLLB) مدربة على مليارات الرموز المميزة.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

قامت أجهزة فك التشفير والتشفير الحديثة بإسقاط RNNs للمحولات. الشكل عالي المستوى (جهاز التشفير، وحدة فك التشفير، إنشاء رمز مميز بواسطة رمز مميز) مطابق لورقة seq2seq لعام 2014. الآلية داخل كل كتلة مختلفة.

### When to still reach for RNN-based seq2seq

تقريبا أبدا، للمشاريع الجديدة. استثناءات محددة:

- الترجمة المتدفقة حيث تستهلك رمزًا مميزًا واحدًا في كل مرة مع ذاكرة محدودة.
- إنشاء نص على الجهاز حيث تكون تكلفة ذاكرة المحول باهظة.
- أصول التدريس. إن فهم عنق الزجاجة بين التشفير وفك التشفير هو أسرع طريق لفهم سبب فوز المحولات.

### Exposure bias and its mitigations

- **أخذ العينات المجدولة.** يصلب المعلم نسبة التأثير أثناء التدريب حتى يتعلم النموذج التعافي من أخطائه.
- **التدريب على الحد الأدنى من المخاطر.** تدرب على درجة BLEU على مستوى الجملة بدلاً من الإنتروبيا المتقاطعة على مستوى الرمز المميز. أقرب إلى ما تريد فعلا.
- ** تعزيز التعلم والضبط الدقيق. ** قم بمكافأة مولد التسلسل بمقياس. يستخدم في الحديث LLM RLHF.

لا تزال الثلاثة تنطبق على التوليد المعتمد على المحولات.

## Ship It

حفظ باسم `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Exercises

1. **سهل.** تنفيذ مهمة نسخ اللعبة. تدريب GRU seq2seq على أزواج المدخلات والمخرجات حيث يكون الهدف يساوي المصدر. قم بقياس الدقة عند الأطوال 5، 10، 20. قم بإعادة إنتاج عنق الزجاجة.
2. **متوسط.** أضف فك تشفير بحث الشعاع بعرض الشعاع 3. قم بالقياس BLEU على جسم متوازي صغير ضد الجشع. قم بتوثيق المكان الذي يفوز فيه البحث عن الشعاع (عادةً ما تكون الرموز الأخيرة) وحيث لا يوجد فرق.
3. **صعب.** الضبط الدقيق `facebook/bart-base` على مجموعة بيانات إعادة صياغة مكونة من 10 آلاف زوج. قارن مخرجات الشعاع 4 للنموذج المضبوط بدقة مع مخرجات النموذج الأساسي عند المدخلات المعلقة. أبلغ عن BLEU واختر 10 أمثلة نوعية.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| التشفير | الإدخال RNN | يقرأ المصدر. يُنتج حالات مخفية لكل خطوة ومتجه سياق نهائي. |
| فك | الإخراج RNN | تمت التهيئة من ناقل السياق. يولد الرموز المستهدفة واحدا تلو الآخر. |
| ناقل السياق | الملخص | الحالة المخفية النهائية لبرنامج التشفير. حجم ثابت. الاهتمام عنق الزجاجة يحل. |
| إجبار المعلم | استخدم الرموز الحقيقية | قم بتغذية رمز الحقيقة الأرضية السابق في وقت التدريب. يستقر التعلم. |
| تحيز التعرض | فجوة التدريب/الاختبار | النموذج الذي تم تدريبه على الرموز الحقيقية لم يمارس أبدًا التعافي من أخطائه. |
| بحث الشعاع | فك أفضل | احتفظ بالتسلسلات الجزئية top-k حية في كل خطوة بدلاً من الالتزام بالجشع. |

## Further Reading

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — the original seq2seq paper. Four pages.
- [Cho et al. (2014). تعلم تمثيل العبارات باستخدام RNN Encoder-Decoder للترجمة الآلية الإحصائية](https://arxiv.org/abs/1406.1078) - قدم GRU وتأطير التشفير-فك التشفير.
- [بهداناو، تشو، بينجيو (2014). الترجمة الآلية العصبية من خلال التعلم المشترك للمحاذاة والترجمة](https://arxiv.org/abs/1409.0473) — ورقة الاهتمام. اقرأ مباشرة بعد هذا الدرس.
- [PyTorch NLP من برنامج Scratch التعليمي](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — seq2seq القابل للبناء + كود الانتباه.
