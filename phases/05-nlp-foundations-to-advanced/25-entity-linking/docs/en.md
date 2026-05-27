# ربط الكيان وتوضيحه
> NER تم العثور على "باريس". ربط الكيان يقرر: باريس، فرنسا؟ باريس هيلتون؟ باريس، تكساس؟ باريس (أمير طروادة)؟ بدون الربط، يظل الرسم البياني المعرفي الخاص بك غامضًا.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 06 (NER)، المرحلة 5 · 24 (القرار المرجعي)
**الوقت:** ~60 دقيقة
## المشكلة
وجاء في الجملة: "الأردن فاز على الصحافة". قام NER بوضع علامات على "الأردن" كـ PERSON. جيد. لكن *أي* الأردن؟
- مايكل جوردان (كرة السلة)؟
- مايكل ب. جوردان (ممثل)؟
- مايكل آي جوردان (أستاذ بيركلي ML - نعم، هذا الالتباس حقيقي في أوراق ML)؟
- الأردن (البلد)؟
- جوردن (الاسم الأول العبري)؟
ربط الكيان (EL) يحل كل إشارة إلى إدخال فريد في قاعدة المعرفة: Wikidata، Wikipedia، DBpedia، أو المجال الخاص بك KB. مهمتان فرعيتان:
1. **جيل المرشحين.** بالنظر إلى "الأردن"، ما هي الإدخالات KB المعقولة؟
2. **توضيح.** بالنظر إلى السياق، من هو المرشح المناسب؟
كلتا الخطوتين قابلة للتعلم. كلاهما يتم قياسهما. لقد ظل خط pipe المدمج مستقرًا لمدة عقد من الزمن - ما يتغير هو جودة أداة توضيح الغموض.
##المفهوم
![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)
**جيل المرشح.** بالنظر إلى صيغة الإشارة ("الأردن")، ابحث عن المرشحين في فهرس الأسماء المستعارة. تغطي قواميس ويكيبيديا المستعارة معظم الكيانات المسماة: "JFK" → جون إف كينيدي، جاكلين كينيدي، JFK المطار، JFK (فيلم). يُرجع الفهرس النموذجي 10-30 مرشحًا لكل ذكر.
**توضيح: ثلاثة مقاربات.**
1. ** السياق السابق + (ميلن وويتن، 2008).** `P(entity | mention) × context-similarity(entity, text)`. يعمل بشكل جيد، سريع، لا يوجد تدريب.
2. **استنادًا إلى التضمين (ESS / REL / Blink).** قم بتشفير الإشارة + السياق. قم بتشفير وصف كل مرشح. اختر الحد الأقصى لجيب التمام. الافتراضي 2020-2024.
3. **توليدي (GENRE، 2021؛ LLM، 2023+).** فك تشفير الاسم المتعارف عليه للكيان رمزًا تلو الآخر. يقتصر على تجربة أسماء الكيانات الصالحة، لذلك يتم ضمان أن يكون الإخراج معرفًا صالحًا KB.
**من النهاية إلى النهاية مقابل pipeline.** تعمل النماذج الحديثة (ELQ، BLINK، ExtEnD، GENRE) على تشغيل NER + إنشاء المرشح + توضيح الغموض في مسار واحد. لا تزال أنظمة خطوط الأنابيب تهيمن على الإنتاج لأنه يمكنك تبديل المكونات.
### القياسين
- **ذكر الاستدعاء (جيل المرشح).** يشير جزء من الذهب إلى المكان الذي يظهر فيه الإدخال الصحيح KB في قائمة المرشحين. الكلمة لكامل pipeline.
- **دقة التوضيح / F1.** في ضوء المرشحين الصحيحين، كم مرة يكون صاحب المركز الأول على حق.
أبلغ دائمًا عن كليهما. النظام الذي يحتوي على توضيح بنسبة 99% و80% من استدعاء المرشحين هو 80% pipeline.
## بنائها
### الخطوة 1: إنشاء فهرس مستعار من عمليات إعادة التوجيه في ويكيبيديا
```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

بيانات الاسم المستعار لويكيبيديا: ~18 مليون زوج (اسم مستعار، كيان). التنزيل من مقالب ويكي بيانات. تخزين كمؤشر مقلوب.
### الخطوة الثانية: توضيح على أساس السياق
```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

تداخل Jaccard هو لعبة. استبدل بتشابه جيب التمام على التضمينات (راجع `code/main.py` الخطوة 2 لإصدار المحول).
### الخطوة 3: القائم على التضمين (نمط BLINK)
```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

في وقت الفهرس، قم بتضمين كل كيان KB مرة واحدة. في وقت الاستعلام، قم بتضمين سياق الإشارة + مرة واحدة، ثم المنتج النقطي مقابل مجموعة المرشحين، واختر الحد الأقصى.
### الخطوة 4: ربط الكيان التوليدي (المفهوم)
GENRE يقوم بفك تشفير عنوان ويكيبيديا الخاص بالكيان حرفًا بحرف. يضمن فك التشفير المقيد (راجع الدرس 20) إمكانية إخراج العناوين الصالحة فقط. تكامل محكم مع محاولة مدعومة بـ KB. السليل الحديث هو REL-GEN وLLM-مطالب به EL بمخرجات منظمة.
```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

بالدمج مع القائمة البيضاء (المخططات التفصيلية `choice`)، يعد هذا هو أبسط EL pipeline ليتم شحنه في عام 2026.
### الخطوة 5: التقييم على AIDA-CoNLL
AIDA-CoNLL هو المعيار EL المعياري: 1,393 مقالة من رويترز، و34 ألف إشارة، وكيانات ويكيبيديا. قم بالإبلاغ عن الدقة في KB (`P@1`) ومعدل الاكتشاف خارج KB NIL.
## مطبات
- **NIL التعامل.** بعض الإشارات ليست في KB (الكيانات الناشئة، والأشخاص الغامضون). يجب أن تتنبأ الأنظمة بـ NIL بدلاً من تخمين الكيان الخاطئ. تقاس بشكل منفصل.
- **أذكر أخطاء الحدود.** يفتقد المنبع NER امتدادات جزئية (تم وضع علامة "Bank of America" ​​على أنها "Bank" فقط). EL يسقط الاستدعاء.
- **التحيز الشعبي.** تبالغ الأنظمة المدربة في التنبؤ بالكيانات المتكررة. غالبًا ما يرتبط ذكر "مايكل آي جوردان" في ورقة ML بكرة السلة بالأردن.
- ** متعدد اللغات EL.** تعيين الإشارات في النص الصيني إلى كيانات ويكيبيديا الإنجليزية. يتطلب برنامج تشفير متعدد اللغات أو خطوة ترجمة.
- **KB الجمود.** الشركات والأحداث والأشخاص الجدد ليسوا في تفريغ ويكيبيديا العام الماضي. تحتاج خطوط الإنتاج pip إلى حلقة تحديث.
## استخدمه
مكدس 2026:
| الوضع | اختر |
|-----------|------|
| الإنجليزية للأغراض العامة + ويكيبيديا | BLINK أو REL |
| متعدد اللغات، KB = ويكيبيديا | النوع |
| LLM-مناسب، عدد قليل من الإشارات/اليوم | موجه Claude/GPT-4 مع قائمة المرشحين + مقيدة JSON |
| خاص بالمجال KB (طبي، قانوني) | مخصص BERT مع استرجاع مدرك لـ KB + ضبط دقيق على مجموعة أنماط المجال AIDA |
| الكمون منخفض للغاية | التطابق التام السابق فقط (خط الأساس لميلن-ويتن) |
| بحث SOTA | GENRE / ممتد / مولدي LLM-EL |
نمط الإنتاج الذي سيتم طرحه في عام 2026: NER → coref → EL عند كل ذكر → انهيار المجموعات إلى كيان أساسي واحد لكل مجموعة. الإخراج: معرف KB واحد لكل كيان في المستند، وليس معرفًا واحدًا لكل إشارة.
## اشحنها
حفظ باسم `outputs/skill-entity-linker.md`:
```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## تمارين
1. **سهل.** قم بتنفيذ أداة توضيح السياق + السابقة في `code/main.py` على 10 إشارات غامضة (باريس، الأردن، Apple). قم بتسمية الكيان الصحيح يدويًا. دقة القياس.
2. **متوسط.** قم بتشفير 50 ​​إشارة غامضة باستخدام محول الجملة. تضمين وصف كل مرشح. قارن التوضيح القائم على التضمين بتداخل سياق Jaccard.
3. **صعب.** أنشئ نطاق كيان مكون من 1 ألف KB (على سبيل المثال، الموظفين + المنتجات في شركتك). قم بتنفيذ NER + EL من البداية إلى النهاية. قم بقياس الدقة واسترجاع 100 جملة معلقة.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| ربط الكيان (EL) | رابط ويكيبيديا | قم بتعيين الإشارة إلى إدخال KB فريد. |
| جيل المرشح | من يمكن أن يكون؟ | قم بإرجاع قائمة مختصرة من إدخالات KB المعقولة للإشارة إليها. |
| توضيح | اختر الخيار الصحيح | سجل المرشحين باستخدام السياق، واختيار الفائز. |
| فهرس الاسم المستعار | جدول البحث | خريطة من النموذج السطحي → الكيانات المرشحة. |
| __المصطلح_3__ | ليس في KB | توقع صريح بعدم تطابق أي إدخال KB. |
| __المصطلح_6__ | قاعدة المعرفة | ويكي بيانات، ويكيبيديا، DBpedia، أو المجال الخاص بك KB. |
| AIDA-CoNLL | المعيار | 1,393 مقالة من رويترز بروابط كيان ذهبي. |
## مزيد من القراءة
- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — المنهج التأسيسي السابق+السياق.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — العمود الفقري القائم على التضمين.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — EL توليدي مع فك تشفير مقيد.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — الورقة المرجعية.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — مكدس الإنتاج المفتوح.