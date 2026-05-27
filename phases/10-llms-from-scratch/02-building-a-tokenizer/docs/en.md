# بناء رمز مميز من الصفر
> الدرس 01 أعطاك لعبة. هذا الدرس يعطيك سلاحا.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 10، الدرس 01 (الرموز المميزة: BPE، WordPiece، SentencePiece)
**الوقت:** ~90 دقيقة
## أهداف التعلم
- أنشئ رمزًا مميزًا BPE على مستوى الإنتاج يتعامل مع Unicode وتسوية المسافات البيضاء والرموز المميزة
- تنفيذ احتياطي على مستوى البايت حتى يتمكن برنامج الرمز المميز من تشفير أي إدخال (بما في ذلك الرموز التعبيرية وCJK والكود) بدون رموز مميزة غير معروفة
- أضف أنماط التعبير العادي للترميز المسبق التي تقسم النص عند حدود الكلمات قبل تطبيق عمليات الدمج BPE
- تدريب رمز مميز مخصص على المجموعة وتقييم نسبة ضغطه مقابل tiktoken على نص متعدد اللغات
## المشكلة
يعمل رمز BPE الخاص بك من الدرس 01 على النص الإنجليزي. الآن رمي اليابانية في ذلك. أو الرموز التعبيرية. أو كود Python مع علامات التبويب والمسافات المختلطة.
ينكسر.
ليس لأن BPE خطأ -- لأن التنفيذ غير مكتمل. يتعامل مُرمز الإنتاج مع البايتات الأولية في أي ترميز، ويقوم بتطبيع Unicode قبل التقسيم، ويدير الرموز المميزة الخاصة التي لا يتم دمجها أبدًا، وتسلسل الترميز المسبق مع تقسيم الكلمات الفرعية، ويفعل كل هذا بسرعة كافية لعدم اختناق تدريب pipeline لمعالجة 15 تريليون رمز مميز.
يحتوي رمز GPT-2 المميز على 50,257 رمزًا. اللاما 3 لديه 128,256. GPT-4 يحتوي على 100000 تقريبًا. هذه ليست أرقام لعبة. تم تدريب جداول الدمج خلف تلك المفردات على مئات الجيجابايت من النص، والآلات المحيطة - التطبيع، والترميز المسبق، وحقن الرمز المميز الخاص، وتنسيق قالب الدردشة - هي ما يفصل بين أداة الرمز المميز التي تتعامل مع "hello World" وبين تلك التي تتعامل مع الإنترنت بالكامل.
أنت ذاهب لبناء تلك الآلات.
##المفهوم
### خط الأنابيب الكامل
رمز الإنتاج ليس خوارزمية واحدة. وهو عبارة عن خط pipمن خمس مراحل، كل منها تحل مشكلة مختلفة.
```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

ولكل مرحلة وظيفة محددة:
| المرحلة | ماذا يفعل | لماذا يهم |
|-------|------------|----------------|
| تطبيع | NFKC Unicode، أحرف صغيرة اختيارية، علامات التمييز اختيارية | يصبح حرف "fi" (U+FB01) "fi" (حرفين). وبدون ذلك، تحصل الكلمة نفسها على رموز مختلفة. |
| الرمز المميز مسبقًا | قم بتقسيم النص إلى أجزاء قبل BPE | يمنع BPE من الدمج عبر حدود الكلمات. يجب ألا تنتج "القطة" أبدًا رمزًا مميزًا "e c". |
| BPE دمج | تطبيق قواعد الدمج المستفادة على تسلسلات البايت | الضغط الأساسي. يحول البايتات الخام إلى رموز الكلمات الفرعية. |
| الرموز الخاصة | أدخل [BOS]، [EOS]، [PAD]، علامات قالب الدردشة | هذه الرموز لها معرفات ثابتة. ولن يشاركوا مطلقًا في عمليات الدمج BPE. النموذج يحتاجهم للهيكل. |
| ID رسم الخرائط | تحويل سلاسل الرمز المميز إلى معرفات عدد صحيح | يرى النموذج الأعداد الصحيحة، وليس السلاسل. |
### مستوى البايت BPE
يعمل رمز الدرس 01 على UTF-8 بايت. كانت تلك هي الدعوة الصحيحة. لكننا تخطينا شيئًا مهمًا: ماذا يحدث عندما تكون تلك البايتات غير صالحة UTF-8؟
مستوى البايت BPE يحل هذه المشكلة عن طريق التعامل مع كل قيمة بايت محتملة (0-255) كرمز مميز صالح. المفردات الأساسية الخاصة بك هي بالضبط 256 إدخالاً. يمكن ترميز أي ملف - نص، أو ثنائي، أو تالف - دون إنتاج رمز مميز غير معروف.
أضاف GPT-2 خدعة: قم بتعيين كل بايت إلى حرف Unicode قابل للطباعة بحيث تظل المفردات قابلة للقراءة بواسطة الإنسان. البايت 0x20 (مسافة) يصبح الحرف "G" في التعيين الخاص بهم. هذا تجميلي بحت. الخوارزمية لا تهتم.
القوة الحقيقية: مستوى البايت BPE يتعامل مع كل لغة على وجه الأرض. الأحرف الصينية هي 3 UTF-8 بايت لكل منها. يمكن أن يكون حجم اللغة اليابانية 3-4 بايت. العربية والديفاناغاري والرموز التعبيرية - كلها مجرد تسلسلات بايت. تبحث خوارزمية BPE عن الأنماط في تسلسلات البايت هذه تمامًا بنفس الطريقة التي تبحث بها عن الأنماط باللغة الإنجليزية ASCII بايت.
### الترميز المسبق
قبل أن يمس BPE النص الخاص بك، تحتاج إلى تقسيمه إلى أجزاء. يمنع هذا خوارزمية الدمج من إنشاء الرموز المميزة التي تتجاوز حدود الكلمات.
يستخدم GPT-2 نمط regex لتقسيم النص:
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

ينقسم هذا النمط إلى الاختصارات ("لا" تصبح "لا" + "'t")، والكلمات ذات المسافات البادئة الاختيارية، والأرقام، وعلامات الترقيم، والمسافات البيضاء. تظل المسافة البادئة متصلة بالكلمة - لذلك تصبح "القط" ["the"، "cat"]، وليس ["the"، ""، "cat"].
يستخدم Llama SentencePiece، الذي يتخطى التعبير العادي بالكامل. فهو يتعامل مع تدفق البايتات الخام كتسلسل طويل واحد ويتيح لخوارزمية BPE معرفة الحدود. هذا أبسط ولكنه يمنح BPE المزيد من الحرية لإنشاء رموز الكلمات المتقاطعة.
الاختيار مهم. يمنع التعبير العادي لـ GPT-2 المُرمز من معرفة أنه يجب دمج "the" في نهاية كلمة واحدة و"the" في بداية الكلمة التالية. يسمح SentencePiece بذلك، مما ينتج عنه في بعض الأحيان ضغطًا أكثر كفاءة ولكن رموزًا أقل قابلية للتفسير.
### الرموز الخاصة
يحتفظ كل رمز مميز للإنتاج بمعرفات رمزية للعلامات الهيكلية:
| الرمز المميز | الغرض | يستخدم بواسطة |
|-------|---------|---------|
| `[BOS]` / `<s>` | بداية التسلسل | اللاما 3، GPT |
| `[EOS]` / `</s>` | نهاية التسلسل | جميع الموديلات |
| __الكود_4__ | الحشو لمحاذاة الدفعة | BERT، T5 |
| __الكود_5__ | رمز مميز غير معروف (مستوى البايت BPE يلغي هذا) | BERT، قطعة Word |
| __الكود_6__ | بداية حدود رسالة الدردشة | شات جي بي تي، كوين |
| __الكود_7__ | نهاية حدود رسالة الدردشة | شات جي بي تي، كوين |
| __الكود_8__ | علامة دوران المستخدم | اللاما 3 |
| __الكود_9__ | مساعد بدوره علامة | اللاما 3 |
لا يتم أبدًا تقسيم الرموز المميزة حسب BPE. وتتم مطابقتها تمامًا قبل تشغيل خوارزمية الدمج، واستبدالها بـ ID الثابت، ويتم ترميز النص المحيط بشكل طبيعي.
### قوالب الدردشة
هذا هو المكان الذي يشعر فيه معظم الناس بالارتباك وتتعطل معظم عمليات التنفيذ.
عندما تقوم بإرسال رسائل إلى نموذج دردشة، يقبل API قائمة من الرسائل:
```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

النموذج لا يرى JSON. يرى تسلسل رمزي مسطح. يقوم قالب الدردشة بتحويل الرسائل إلى هذا التسلسل الثابت باستخدام رموز خاصة. كل نموذج يفعل ذلك بشكل مختلف:
```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

أخطأ في القالب وسينتج النموذج بيانات غير صحيحة. تم تدريبه على تنسيق واحد محدد. أي انحراف - سطر جديد مفقود، أو رمز مميز متبادل، أو مساحة إضافية - يضع المدخلات خارج توزيع التدريب.
### سرعة
بايثون بطيئة جدًا في ترميز الإنتاج.
tiktoken (OpenAI) مكتوب باللغة Rust باستخدام روابط Python. تعتبر رموز HuggingFace أيضًا Rust. SentencePiece هو C++. تحقق هذه عمليات تسريع بمعدل 10 إلى 100 مرة مقارنة ببايثون النقي.
من أجل المنظور: ترميز 15 تريليون رمز للتدريب المسبق على Llama 3 بمعدل مليون رمز في الثانية (Fast Python) سيستغرق 174 يومًا. عند 100 مليون رمز في الثانية (Rust)، يستغرق الأمر 1.7 يومًا.
أنت تقوم بالبناء باستخدام لغة بايثون لفهم الخوارزمية. في الإنتاج، ستستخدم تطبيقًا مُجمَّعًا وتلمس فقط غلاف Python.
## بنائها
### الخطوة 1: التشفير على مستوى البايت
الأساس. تحويل أي سلسلة إلى سلسلة من البايتات، وتعيين كل بايت إلى حرف قابل للطباعة للعرض، وعكس العملية.
```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

اختبار على نص متعدد اللغات لمعرفة عدد البايتات:
```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"مرحبا" هو 5 بايت. "你好" هو 6 بايت (3 لكل حرف). حجم التعبير الناري هو 4 بايت. لا يهتم مُرمز مستوى البايت باللغة التي هي عليها. البايتات هي بايت.
### الخطوة 2: إنشاء الرمز المميز باستخدام Regex
قم بتقسيم النص إلى أجزاء باستخدام نمط regex GPT-2. يتم ترميز كل قطعة بشكل مستقل بواسطة BPE.
```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

تدعم الوحدة `regex` هروب خاصية Unicode (`\p{L}` للأحرف، `\p{N}` للأرقام). وحدة المكتبة القياسية `re` لا تفعل ذلك، لذلك نعود إلى فئات الأحرف ASCII. بالنسبة لإنتاج الرموز المميزة متعددة اللغات، قم بتثبيت `regex`.
جربه:
```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

تظل المساحة البادئة مرتبطة بالكلمة. انقسمت التقلصات عند الفاصلة العليا. علامات الترقيم تصبح قطعة خاصة بها. BPE لن يقوم أبدًا بدمج الرموز المميزة عبر هذه الحدود.
### الخطوة 3: BPE على تسلسلات البايت
الخوارزمية الأساسية من الدرس 01، ولكنها تعمل الآن على أجزاء تم ترميزها مسبقًا بشكل مستقل.
```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### الخطوة 4: التعامل مع الرموز المميزة
تحتاج الرموز المميزة إلى مطابقة تامة ومعرفات ثابتة. إنهم يتجاوزون BPE تمامًا.
```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### الخطوة 5: فئة Tokenizer الكاملة
قم بربط كل شيء معًا: التطبيع، والتقسيم على الرموز الخاصة، والترميز المسبق، ودمج BPE، والتخطيط للمعرفات.
```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### الخطوة 6: اختبار متعدد اللغات
الاختبار الحقيقي. استخدم اللغة الإنجليزية والصينية والرموز التعبيرية والرموز عليها.
```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

تنتج الأحرف الصينية 3 بايت لكل منها. تنتج الرموز التعبيرية 4 بايت. لا شيء من هذا يعطل الرمز المميز. لا شيء ينتج رموزًا غير معروفة. هذه هي قوة مستوى البايت BPE.
## استخدمه
### مقارنة الرموز الحقيقية
قم بتحميل الرموز المميزة الفعلية من Llama 3 وGPT-4 وMistral. انظر كيف يتعامل كل منهم مع نفس الفقرة متعددة اللغات.
```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

سترى أعدادًا مختلفة من الرموز المميزة لنفس النص. يعتبر Llama 3 الذي يحتوي على 128 ألف مفردة أكثر عدوانية في دمج الأنماط الشائعة. GPT-4 مع وجود 100 ألف في المنتصف. تنتج ميسترال ذات 32 كيلو بايت المزيد من الرموز المميزة ولكنها تحتوي على طبقة تضمين أصغر.
إن المقايضة هي نفسها دائمًا: المفردات الأكبر تعني تسلسلات أقصر ولكن المزيد من المعلمات.
## اشحنها
يُنتج هذا الدرس مطالبة بإنشاء رموز مميزة للإنتاج وتصحيح أخطاءها. انظر `outputs/prompt-tokenizer-builder.md`.
## تمارين
1. **سهل:** أضف طريقة `get_token_bytes(id)` التي تعرض وحدات البايت الأولية لأي رمز مميز ID. استخدمه لفحص ما تمثله الرموز المميزة المدمجة الأكثر شيوعًا بالفعل.
2. **متوسط:** قم بتنفيذ الرمز المميز المسبق بنمط اللاما الذي ينقسم على مسافات بيضاء وdigits ولكنه يحتفظ بالمسافات البادئة. قارن مفرداتها مع أسلوب GPT-2 regex في نفس المجموعة.
3. **صعب:** أضف طريقة قالب الدردشة التي تأخذ قائمة من رسائل `{"role": ..., "content": ...}` وتنتج التسلسل المميز الصحيح لتنسيق دردشة Llama 3. اختبره مقابل تطبيق HuggingFace.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| مستوى البايت BPE | "الرمز المميز الذي يعمل على وحدات البايت" | BPE بمفردات أساسية تبلغ 256 بايت - يتعامل مع أي إدخال بدون رموز مميزة غير معروفة |
| الترميز المسبق | "التقسيم قبل BPE" | Regex أو التقسيم المستند إلى القواعد الذي يمنع BPE من الدمج عبر حدود الكلمات |
| NFKC التطبيع | "تنظيف يونيكود" | التحليل الأساسي متبوعًا بتكوين التوافق - يصبح حرف "fi" "fi"، ويصبح العرض الكامل "A" "A" |
| قالب الدردشة | "كيف تصبح الرسائل رموزًا" | التنسيق الدقيق لتحويل قائمة رسائل الدور/المحتوى إلى تسلسل رمزي مسطح - خاص بالنموذج ويجب أن يتطابق مع تنسيق التدريب |
| الرموز الخاصة | "رموز التحكم" | معرفات الرموز المميزة المحجوزة التي تتجاوز BPE -- [BOS]، [EOS]، [PAD]، علامات الدردشة -- متطابقة تمامًا قبل الدمج |
| الخصوبة | "الرموز لكل كلمة" | نسبة الرموز المميزة للإخراج إلى الكلمات المدخلة - 1.3 للغة الإنجليزية في GPT-4، 2-3 للغة الكورية، أعلى يعني السياق الضائع |
| تيك توكين | "OpenAI الرمز المميز" | Rust BPE التنفيذ باستخدام روابط Python - أسرع بمقدار 10 إلى 100 مرة من Python النقي |
| دمج الجدول | "المفردات" | قائمة مرتبة لدمج أزواج البايتات التي تم تعلمها أثناء التدريب - هذا IS المعرفة المكتسبة لأداة الرمز المميز |
## مزيد من القراءة
- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Rust BPE التنفيذ المستخدم بواسطة GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Rust مكتبة الرموز المميزة التي تدعم BPE، WordPiece، Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- تفاصيل حول 128 ألف مفردة وتدريب على الرموز المميزة
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- ترميز لا يعرف اللغة
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) - التعيين الأصلي للبايت إلى Unicode