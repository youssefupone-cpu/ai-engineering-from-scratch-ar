# Vision Transformers (ViT)

> قم بقص الصورة إلى بقع، وتعامل مع كل رقعة على أنها كلمة، وقم بتشغيل محول قياسي. لا تنظر إلى الوراء.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة السابعة الدرس 02 (الانتباه الذاتي)، المرحلة 4 الدرس 04 (تصنيف الصور)
**الوقت:** ~45 دقيقة

## Learning Objectives

- تنفيذ تضمين التصحيح، والتضمين الموضعي المكتسب، ورمز الفئة، وكتل تشفير المحولات من البداية لبناء الحد الأدنى من ViT
- اشرح لماذا كان يُعتقد أن ViT يحتاج إلى بيانات ضخمة للتدريب المسبق حتى أثبت DeiT وMAE عكس ذلك
- قارن بين ViT وSwin وConvNeXt في سابقاتهم المعمارية (لا شيء، انتباه النافذة المحلية، العمود الفقري للتحويل)
- ضبط ViT المُدرب مسبقًا على مجموعة بيانات صغيرة باستخدام `timm` ووصفة المسبار الخطي القياسي / الضبط الدقيق

## The Problem

لمدة عقد من الزمن، كان الإلتواء مرادفًا للرؤية الحاسوبية. كان لشبكات CNN تحيزات استقرائية قوية - المحلية، وتكافؤ الترجمة - والتي لم يعتقد أحد أنه يمكنك استبدالها. ثم دوسوفيتسكي وآخرون. (2020) أظهر أن المحول العادي المطبق على بقع الصور المسطحة، مع عدم وجود آلات تلافيفية على الإطلاق، يمكن أن يطابق أو يتفوق على أفضل شبكات CNN على نطاق واسع.

وكان المصيد "على نطاق واسع". خسر ViT على ImageNet-1k أمام ResNet. تم تدريب ViT مسبقًا على ImageNet-21k أو JFT-300M ثم تم ضبطه على ImageNet-1k للتغلب عليه. وكان الاستنتاج هو أن المحولات كانت تفتقر إلى مقدمات مفيدة ولكن يمكنها تعلمها من البيانات الكافية. أظهر العمل اللاحق (DeiT، MAE، DINO) أنه مع وصفات التدريب الصحيحة - التعزيز القوي، والتدريب المسبق تحت الإشراف الذاتي، والتقطير - تتدرب ViTs جيدًا على البيانات الصغيرة أيضًا.

بحلول عام 2026، لا تزال شبكات CNN النقية قادرة على المنافسة على الأجهزة المتطورة (ConvNeXt هو الأقوى)، لكن المحولات تهيمن على كل شيء آخر: التجزئة (Mask2Former، SegFormer)، والكشف (DETR، RT-DETR)، متعدد الوسائط (CLIP، SigLIP)، الفيديو (VideoMAE، VJEPA). هيكل كتلة ViT هو الذي يجب معرفته.

## The Concept

### The pipeline

```mermaid
flowchart LR
    IMG["Image<br/>(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["Flatten to<br/>(196, 768) tokens"]
    FLAT --> CAT["Prepend<br/>[CLS] token"]
    CAT --> POS["Add learned<br/>positional embed"]
    POS --> ENC["N transformer<br/>encoder blocks"]
    ENC --> CLS["Take [CLS]<br/>token output"]
    CLS --> HEAD["MLP classifier"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

سبع خطوات. التصحيحات -> الرموز المميزة -> الاهتمام -> المصنف. كل متغير (DeiT، Swin، ConvNeXt، MAE التدريب المسبق) يغير واحدًا أو اثنين من السبعة ويترك الباقي بمفرده.

### Patch embedding

التحويل الأول هو السر. حجم النواة 16، والخطوة 16، لذا تصبح الصورة مقاس 224 × 224 شبكة مقاس 14 × 14 مكونة من تصحيحات مقاس 16 × 16، يتم عرض كل منها على تضمين 768 خافتًا. يقوم هذا التحويل الفردي بالتصحيح والمشاريع الخطية.

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 رقعة = 196 رمزًا. البعد المميز لكل رمز هو 768 (ViT-B)، أو 1024 (ViT-L)، أو 1280 (ViT-H).

### Class token

متجه واحد متعلم مُلحق بالتسلسل:

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

بعد كتل المحولات N، يكون الإخراج `[CLS]` هو تمثيل الصورة العامة. يقرأ رأس التصنيف هذا المتجه الواحد فقط.

### Positional embedding

المحولات ليس لديها فكرة مدمجة عن الموقع المكاني. أضف متجهًا مكتسبًا إلى كل رمز مميز:

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

التضمين هو معلمة للنموذج؛ يقوم التدريب القائم على التدرج بتكييفه مع بنية الصورة ثنائية الأبعاد. توجد بدائل جيبية ثنائية الأبعاد ولكن نادرًا ما يتم استخدامها عمليًا.

### Transformer encoder block

معيار. الاهتمام الذاتي متعدد الرؤوس، MLP، الاتصالات المتبقية، ما قبل الطبقة.

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

يقوم ViT-B/16 بتجميع 12 من هذه الكتل، كل منها يحتوي على 12 رأس انتباه، بإجمالي 86 مليون معلمة.

### Why pre-LN

استخدمت المحولات المبكرة ما بعد LN (`x = LN(x + sublayer(x))`) وواجهت صعوبة في التدريب بعد 6-8 طبقات دون إحماء. يقوم ما قبل LN (`x = x + sublayer(LN(x))`) بتدريب شبكات أعمق بثبات دون إحماء. كل ViT وكل LLM حديث يستخدم ما قبل LN.

### Patch size trade-off

- 16 × 16 تصحيحًا -> 196 رمزًا قياسيًا.
- 32 × 32 تصحيحًا -> 49 رمزًا، دقة أسرع ولكن أقل.
- تصحيحات 8 × 8 -> 784 رمزًا مميزًا، وهي أفضل ولكن مقاييس تكلفة الاهتمام O(n^2) سيئة.

تصحيحات أكبر = عدد أقل من الرموز = تفاصيل مكانية أسرع ولكن أقل. يستخدم SwinV2 تصحيحات 4x4 في النوافذ ذات التسلسل الهرمي.

### DeiT's recipe for training ViT on ImageNet-1k

احتاج ViT الأصلي إلى JFT-300M للتغلب على شبكات CNN. قام DeiT (Touvron et al., 2020) بتدريب ViT-B على الوصول إلى المركز الأول بنسبة 81.8% على ImageNet-1k وحده مع أربعة تغييرات:

1. التكبير الثقيل: RandAugment، Mixup، CutMix، المسح العشوائي.
2. العمق العشوائي (إسقاط كتل كاملة بشكل عشوائي أثناء التدريب).
3. التكبير المتكرر (نفس الصورة 3 مرات لكل دفعة).
4. التقطير من معلم CNN (اختياري، يرفع الدقة أكثر).

كل وصفة تدريب ViT حديثة تنحدر من DeiT.

### Swin vs ConvNeXt

- **سوين** (ليو وآخرون، 2021) — الاهتمام القائم على النافذة. تحضر كل كتلة ضمن نافذة محلية؛ تعمل الكتل المتناوبة على تحريك النافذة لخلط المعلومات عبر النوافذ. يعيد مكانًا يشبه CNN سابقًا مع الحفاظ على عامل الانتباه.
- **ConvNeXt** (Liu et al., 2022) - أعيد تصميم CNN ليتوافق مع خيارات بنية Swin (التحويلات العميقة، LayerNorm، GELU، عنق الزجاجة المقلوب). أظهر أن الفجوة ليست "الانتباه مقابل الإلتواء" بل "وصفة التدريب الحديثة + الهندسة المعمارية".

في عام 2026، سيكون كل من ConvNeXt-V2 وSwin-V2 كلاهما من فئة الإنتاج؛ يعتمد الاختيار الصحيح على مكدس الاستدلال الخاص بك (يجمع ConvNeXt بشكل أفضل للحافة) ومجموعة التدريب المسبق.

### MAE pretraining

جهاز التشفير التلقائي المقنع (He et al., 2022): قناع 75% من التصحيحات بشكل عشوائي، وتدريب جهاز التشفير على معالجة 25% المرئية فقط، وتدريب وحدة فك ترميز صغيرة لإعادة بناء التصحيحات المقنعة من مخرجات جهاز التشفير. بعد التدريب المسبق، تخلص من وحدة فك التشفير وقم بضبط جهاز التشفير.

MAE makes ViT قابل للتدريب على ImageNet-1k وحده، ويصل إلى SOTA، وهي الوصفة الافتراضية الحالية الخاضعة للإشراف الذاتي.

## Build It

### Step 1: Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

تحويل واحد، وتسوية واحدة، وتبديل واحد. هذه هي خطوة تحويل الصورة إلى الرموز بأكملها.

### Step 2: Transformer block

ما قبل LN، الاهتمام الذاتي متعدد الرؤوس، MLP مع GELU، الاتصالات المتبقية.

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` يتعامل مع التقسيم إلى رؤوس، والمنتج النقطي المقيس، وإسقاط المخرجات. `batch_first=True` إذن الأشكال هي `(N, seq, dim)`.

### Step 3: The ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

حوالي 2.8 مليون معلمة - ViT صغير يمكن تتبعه على CPU. الحقيقي ViT-B هو 86M؛ نفس تعريف الفئة مع `dim=768, depth=12, num_heads=12`.

### Step 4: Sanity check — single image inference

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

يجب أن تعمل دون خطأ. مجموع الاحتمالات هو 1.

## Use It

`timm` يشحن كل متغير ViT مع أوزان ImageNet المدربة مسبقًا. سطر واحد:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` هو الإعداد الافتراضي للإنتاج لمحولات الرؤية في عام 2026. يدعم ViT وDeiT وSwin وSwin-V2 وConvNeXt وConvNeXt-V2 وMaxViT وMViT وEfficientFormer وعشرات الآخرين تحت نفس API.

للعمل متعدد الوسائط (صورة + نص)، `transformers` السفن CLIP، SigLIP، BLIP-2، LLaVA. برنامج تشفير الصور في كل هذه العناصر هو متغير ViT.

## Ship It

ينتج هذا الدرس:

- `outputs/prompt-vit-vs-cnn-picker.md` — مطالبة تختار بين ViT أو ConvNeXt أو Swin بناءً على حجم مجموعة البيانات والحوسبة ومكدس الاستدلال.
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — مهارة تتحقق من مطابقة أشكال التضمين الموضعي والتضمين التصحيحي لـ ViT مع طول التسلسل المتوقع للنموذج، مما يؤدي إلى اكتشاف أخطاء النقل الأكثر شيوعًا.

## Exercises

1. **(سهل)** اطبع أشكال كل موتر متوسط ​​لتمرير للأمام عبر ViT الصغير أعلاه. أكد: الإدخال `(N, 3, 64, 64)` -> التصحيحات `(N, 16, 192)` -> مع CLS `(N, 17, 192)` -> إدخال المصنف `(N, 192)` -> الإخراج `(N, num_classes)`.
2. **(متوسط)** قم بضبط `timm` ViT-S/16 المُدرب مسبقًا على مجموعة البيانات الاصطناعية CIFAR من الدرس 4. قارن مع الضبط الدقيق لـ ResNet-18 على نفس البيانات. الإبلاغ عن وقت التدريب والدقة النهائية.
3. **(صعب)** تنفيذ MAE التدريب المسبق على ViT الصغير: قناع 75% من التصحيحات، وتدريب جهاز التشفير + وحدة فك ترميز صغيرة لإعادة بناء البقع المقنعة. تقييم دقة المسبار الخطي على البيانات الاصطناعية قبل وبعد التدريب المسبق.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| تضمين التصحيح | "التحويل الأول" | تحويل بحجم النواة = الخطوة = حجم التصحيح؛ يحول الصورة إلى شبكة من التضمينات الرمزية |
| رمز الفئة | "[CLS]" | المتجه المكتسب المُسبق لتسلسل الرمز المميز؛ الناتج النهائي هو تمثيل الصورة العالمية |
| التضمين الموضعي | "تعلمت نقاط البيع" | يتم إضافة ناقل تم تعلمه إلى كل رمز حتى يعرف المحول من أين جاء كل تصحيح |
| ما قبل LN | "LayerNorm قبل الطبقة الفرعية" | متغير المحول المستقر: `x + sublayer(LN(x))` بدلاً من `LN(x + sublayer(x))` |
| اهتمام متعدد الرؤوس | "الاهتمام الموازي" | يتم تقسيم انتباه المحول القياسي إلى مساحات فرعية مستقلة num_heads، متسلسلة بعد ذلك |
| فيت-ب/16 | "القاعدة، التصحيح 16" | الحجم القانوني: خافت = 768، العمق = 12، الرؤوس = 12، حجم التصحيح = 16، الصورة = 224؛ ~86 مليون معلمة |
| ديت | "ViT الموفر للبيانات" | تم تدريب ViT على ImageNet-1k وحده مع تعزيز قوي؛ مجموعات البيانات الكبيرة التي تم إثباتها قبل التدريب ليست مطلوبة بشكل صارم |
| MAE | "جهاز التشفير التلقائي المقنع" | التدريب المسبق تحت الإشراف الذاتي: قناع 75% من الرقع، وإعادة بنائها؛ وصفة التدريب المسبق لـ ViT المهيمنة |

## Further Reading

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — the ViT paper
- [DeiT: Data-efficient Image Transformers (ToTouvronron et al., 2020)](https://arxiv.org/abs/2012.12877) — كيفية تدريب ViT على ImageNet-1k وحده
- [أجهزة التشفير التلقائي المقنعة هي متعلمون ذوو رؤية قابلة للتطوير (هو وآخرون، 2022)](https://arxiv.org/abs/2111.06377) — MAE تدريب مسبق
- [توثيق تيم](https://huggingface.co/docs/timm) — المرجع لكل محول رؤية ستستخدمه في الإنتاج
