# تقييم النموذج
> جودة النموذج لا تقل عن جودة الطريقة التي تقيسه بها.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة الأولى (الاحتمالات والتوزيعات، إحصائيات ML)، المرحلة الثانية الدروس 1-8
**الوقت:** ~90 دقيقة
## أهداف التعلم
- تنفيذ التحقق من صحة K-fold والطبقية K-fold من الصفر وشرح سبب أهمية التقسيم الطبقي للبيانات غير المتوازنة
- دقة الحساب، والتذكير، F1، AUC-ROC، ومقاييس الانحدار (MSE، RMSE، MAE، R-squared) من البداية
- تفسير منحنيات التعلم لتشخيص ما إذا كان النموذج يعاني من التحيز العالي أو التباين العالي
- تحديد أخطاء التقييم الشائعة بما في ذلك تسرب البيانات، واختيار المقاييس الخاطئة، وتلوث مجموعة الاختبار
## المشكلة
لقد قمت بتدريب نموذج. تحصل على دقة تصل إلى 95% في بياناتك. هل هو جيد؟
ربما. ربما لا. إذا كانت 95% من بياناتك تنتمي إلى فئة واحدة، فإن النموذج الذي يتوقع دائمًا تلك الفئة يحصل على دقة بنسبة 95% بينما يكون عديم الفائدة تمامًا. إذا قمت بالتقييم على نفس البيانات التي تدربت عليها، فإن رقم 95% لا معنى له لأن النموذج يحفظ الإجابات فقط. إذا كانت مجموعة البيانات الخاصة بك تحتوي على مكون زمني وقمت بخلطها بشكل عشوائي قبل التقسيم، فقد يستخدم نموذجك البيانات المستقبلية للتنبؤ بالماضي.
تقييم النموذج هو المكان الذي تخطئ فيه معظم مشاريع ML. المقياس الخاطئ make هو نموذج سيء يبدو جيدًا. يتيح التقسيم الخاطئ للعارضة الغش. المقارنة الخاطئة makes تختار النموذج الأسوأ. الحصول على التقييم الصحيح ليس أمرًا اختياريًا. إنه الفرق بين النموذج الذي ينجح في الإنتاج والنموذج الذي يفشل في اللحظة التي يرى فيها بيانات حقيقية.
##المفهوم
### التدريب والتحقق والاختبار
```mermaid
flowchart LR
    A[Full Dataset] --> B[Train Set 60-70%]
    A --> C[Validation Set 15-20%]
    A --> D[Test Set 15-20%]
    B --> E[Fit Model]
    E --> C
    C --> F[Tune Hyperparameters]
    F --> E
    F --> G[Final Model]
    G --> D
    D --> H[Report Performance]
```

ثلاثة انقسامات وثلاثة أغراض:
- **مجموعة التدريب**: يتعلم النموذج من هذه البيانات. ويرى هذه الأمثلة أثناء التدريب.
- **مجموعة التحقق من الصحة**: تستخدم لضبط المعلمات الفائقة والاختيار بين النماذج. لا يتدرب النموذج أبدًا على هذه البيانات، لكن قراراتك تتأثر بها.
- **مجموعة الاختبار**: تم لمسها مرة واحدة بالضبط، في النهاية، للإبلاغ عن الأداء النهائي. إذا نظرت إلى أداء الاختبار ثم رجعت لتغيير النموذج الخاص بك، فستجد أنه لم يعد مجموعة اختبار. لقد أصبحت مجموعة التحقق الثانية.
مجموعة الاختبار هي ضمانك الراسخ بأن الأداء المبلغ عنه يعكس كيفية عمل النموذج مع البيانات غير المرئية حقًا.
### التحقق من صحة الطية K
باستخدام مجموعات البيانات الصغيرة، يؤدي قطار واحد/عملية التحقق إلى تقسيم البيانات وإعطاء تقديرات مشوشة. يستخدم التحقق المتقاطع K-fold جميع البيانات لكل من التدريب والتحقق من الصحة:
```mermaid
flowchart TB
    subgraph Fold1["Fold 1"]
        direction LR
        V1["Val"] --- T1a["Train"] --- T1b["Train"] --- T1c["Train"] --- T1d["Train"]
    end
    subgraph Fold2["Fold 2"]
        direction LR
        T2a["Train"] --- V2["Val"] --- T2b["Train"] --- T2c["Train"] --- T2d["Train"]
    end
    subgraph Fold3["Fold 3"]
        direction LR
        T3a["Train"] --- T3b["Train"] --- V3["Val"] --- T3c["Train"] --- T3d["Train"]
    end
    subgraph Fold4["Fold 4"]
        direction LR
        T4a["Train"] --- T4b["Train"] --- T4c["Train"] --- V4["Val"] --- T4d["Train"]
    end
    subgraph Fold5["Fold 5"]
        direction LR
        T5a["Train"] --- T5b["Train"] --- T5c["Train"] --- T5d["Train"] --- V5["Val"]
    end
    Fold1 --> R["Average scores"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. قم بتقسيم البيانات إلى طيات متساوية الحجم
2. بالنسبة لكل طية، قم بالتدريب على طيات K-1 والتحقق من صحة الطية المتبقية
3. متوسط درجات التحقق من صحة K
K=5 أو K=10 هي اختيارات قياسية. يتم استخدام كل نقطة بيانات للتحقق من الصحة مرة واحدة بالضبط. متوسط ​​النتيجة هو تقدير أكثر استقرارًا من أي تقسيم فردي.
** طية K الطبقية **: تحافظ على توزيع الفئة في كل طية. إذا كانت مجموعة البيانات الخاصة بك هي 70% من الفئة (أ) و30% من الفئة (ب)، فسيكون لكل طية نفس النسبة تقريبًا. يعد هذا أمرًا مهمًا بالنسبة لمجموعات البيانات غير المتوازنة حيث قد يؤدي التقسيم العشوائي إلى وضع جميع عينات الأقليات في حظيرة واحدة.
### مقاييس التصنيف
**مصفوفة الارتباك**: الأساس. للتصنيف الثنائي:
|  | توقع إيجابي | توقع سلبي |
|--|---|---|
| في الواقع إيجابي | صحيح إيجابي (TP) | سلبي كاذب (FN) |
| في الواقع سلبي | إيجابية كاذبة (FP) | صحيح سلبي (TN) |
ومن هذه المصفوفة، تتبع جميع المقاييس الأخرى ما يلي:
- **الدقة** = (TP + TN) / (TP + TN + FP + FN). جزء من التوقعات الصحيحة. مضللة عندما تكون الطبقات غير متوازنة.
- **الدقة** = TP / (TP + FP). من بين كل الأشياء المتوقعة الإيجابية، كم كان عددها في الواقع؟ يُستخدم عندما تكون النتائج الإيجابية الخاطئة مكلفة (على سبيل المثال، يقوم مرشح البريد العشوائي بوضع علامة على البريد الإلكتروني الحقيقي كبريد عشوائي).
- **استدعاء** (الحساسية) = TP / (TP + FN). من بين جميع الإيجابيات الفعلية، كم منها حصلنا عليها؟ يُستخدم عندما تكون النتائج السلبية الكاذبة مكلفة (على سبيل المثال، فحص السرطان الذي يفتقد الورم).
- **F1 النتيجة** = 2 * الدقة * الاستدعاء / (الدقة + الاستدعاء). الوسط التوافقي للدقة والاستذكار. يوازن بين الاثنين عندما لا يهيمن أي منهما بشكل واضح.
- **AUC-ROC**: المنطقة الواقعة أسفل منحنى خصائص تشغيل جهاز الاستقبال. يرسم المعدل الإيجابي الحقيقي مقابل المعدل الإيجابي الكاذب عند عتبات التصنيف المختلفة. AUC = 0.5 يعني التخمين العشوائي، AUC = 1.0 يعني الفصل المثالي. مستقل عن العتبة: فهو يقيس مدى جودة النموذج في تصنيف الإيجابيات فوق السلبيات، بغض النظر عن الحد الذي تختاره.
### مقاييس الانحدار
- **MSE** (متوسط ​​الخطأ التربيعي) = المتوسط ​​((y_true - y_pred)^2). يعاقب الأخطاء الكبيرة من الدرجة الثانية. حساسة للقيم المتطرفة.
- **RMSE** (جذر متوسط ​​الخطأ التربيعي) = sqrt(MSE). نفس الوحدات مثل المتغير المستهدف. أسهل في التفسير من MSE.
- **MAE** (متوسط ​​الخطأ المطلق) = المتوسط ​​(|y_true - y_pred|). يعالج كافة الأخطاء خطيا. أكثر قوة للقيم المتطرفة من MSE.
- **R-squared** = 1 - SS_res / SS_tot، حيث SS_res = sum((y_true - y_pred)^2) وSS_tot = sum((y_true - y_mean)^2). جزء التباين الذي يوضحه النموذج. R^2 = 1.0 مثالي. R^2 = 0.0 يعني أن النموذج ليس أفضل من التنبؤ دائمًا بالمتوسط. يمكن أن تكون قيمة R^2 سالبة إذا كان النموذج أسوأ من المتوسط.
### منحنيات التعلم
رسم نقاط التدريب والتحقق من الصحة كدالة لحجم مجموعة التدريب:
- **انحياز عالي (نقص التجهيز)**: يتقارب كلا المنحنيين إلى درجة منخفضة. إضافة المزيد من البيانات لن يساعد. أنت بحاجة إلى نموذج أكثر تعقيدًا.
- **التباين العالي (التركيب الزائد)**: درجة التدريب مرتفعة ولكن درجة التحقق أقل بكثير. والفجوة بينهما كبيرة. من المفترض أن تساعد إضافة المزيد من البيانات.
### منحنيات التحقق من الصحة
التدريب على الرسم ودرجات التحقق من الصحة كوظيفة للمعلمة الفائقة:
- عند التعقيد المنخفض: كلا الدرجات منخفضة (غير مناسبة)
- عند التعقيد الصحيح: كلا الدرجات عالية ومتقاربة من بعضها البعض
- في حالة التعقيد العالي: تظل درجة التدريب مرتفعة ولكن تنخفض درجة التحقق (التجاوز)
قيمة المعلمة الفائقة المثالية هي حيث تبلغ درجة التحقق من الصحة ذروتها.
### أخطاء التقييم الشائعة
**تسرب البيانات**: تتسرب المعلومات من مجموعة الاختبار إلى التدريب. أمثلة: تركيب مقياس على مجموعة البيانات الكاملة قبل التقسيم، بما في ذلك البيانات المستقبلية في تنبؤ السلاسل الزمنية، باستخدام ميزة مشتقة من الهدف. قم دائمًا بالتقسيم أولاً، ثم المعالجة المسبقة.
**اختلال الفئة**: 99% من المعاملات هي legitimate، و1% عبارة عن احتيال. النموذج الذي يتنبأ دائمًا بـ "legitimate" يحصل على دقة تبلغ 99%. استخدم الدقة أو الاستدعاء F1 أو AUC-ROC بدلاً من ذلك.
**مقياس خاطئ**: تحسين الدقة عندما يتعين عليك تحسين الاستدعاء (التشخيص الطبي)، أو تحسين RMSE عندما تحتوي بياناتك على قيم متطرفة ثقيلة (استخدم MAE بدلاً من ذلك).
**عدم استخدام الانقسامات الطبقية**: في حالة البيانات غير المتوازنة، قد يؤدي التقسيم العشوائي إلى وضع عدد قليل جدًا من عينات الأقليات في منطقة التحقق من الصحة، مما يعطي تقديرات غير مستقرة.
**الاختبار في كثير من الأحيان**: في كل مرة تنظر فيها إلى أداء الاختبار وتعدله، فإنك تفرط في التوافق مع مجموعة الاختبار. مجموعة الاختبار للاستخدام الفردي.
## بنائها
### الخطوة 1: التدريب/التحقق/تقسيم الاختبار
```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### الخطوة 2: التحقق من صحة الطية K والطبقية على شكل K
```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### الخطوة 3: مصفوفة الارتباك ومقاييس التصنيف
```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### الخطوة 4: مقاييس الانحدار
```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### الخطوة 5: منحنيات التعلم
```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### الخطوة 6: مصنف بسيط للاختبار، بالإضافة إلى العرض التوضيحي الكامل
```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")
```

## استخدمه
باستخدام scikit-learn، يتم دمج التقييم في سير العمل:
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

تُظهر الإصدارات من البداية بالضبط ما يفعله التحقق المتبادل (لا يوجد سحر، فقط للحلقات وتتبع الفهرس)، وكيف يتم حساب كل مقياس (فقط العد TP/FP/TN/FN)، وسبب أهمية التقسيم الطبقي (الحفاظ على نسب الطبقة في كل طية). تضيف إصدارات المكتبة التوازي والمزيد من خيارات التسجيل والتكامل مع pipelines.
## اشحنها
ينتج هذا الدرس:
- `outputs/skill-evaluation.md` - مهارة تغطي استراتيجية التقييم لنماذج التصنيف والانحدار
## تمارين
1. تنفيذ منحنيات الاسترجاع الدقيق: دقة الرسم مقابل الاستدعاء عند عتبات مختلفة. حساب متوسط ​​الدقة (المساحة تحت المنحنى PR). قارن منحنى PR بمنحنى ROC في مجموعة بيانات غير متوازنة واشرح متى يكون كل منهما أكثر إفادة.
2. إنشاء حلقة تحقق متداخلة: تقوم الحلقة الخارجية بتقييم أداء النموذج، بينما تقوم الحلقة الداخلية بضبط المعلمات الفائقة. استخدمه لمقارنة نموذجين بشكل عادل دون تسريب بيانات التحقق من الصحة إلى التقييم.
3. قم بتنفيذ اختبار التقليب لمقارنة النماذج: قم بخلط التسميات وإعادة التدريب وقياس الأداء. كرر 100 مرة لإنشاء توزيع فارغ. حساب القيمة p لأداء النموذج الملاحظ مقابل هذا التوزيع.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| التجهيز الزائد | "حفظ بيانات التدريب" | يلتقط النموذج الضوضاء في بيانات التدريب، ويقدم أداءً جيدًا في التدريب ولكنه سيئًا في البيانات غير المرئية |
| التحقق المتبادل | "اختبار على مجموعات فرعية مختلفة" | التدوير المنهجي لأي جزء من البيانات يتم استخدامه للتحقق من الصحة، ومتوسط ​​النتائج عبر جميع عمليات التدوير |
| الدقة | "كم عدد النتائج الإيجابية المتوقعة الصحيحة" | TP / (TP + FP): جزء التوقعات الإيجابية الإيجابية بالفعل |
| أذكر | "كم عدد الإيجابيات الفعلية التي وجدناها" | TP / (TP + FN): جزء الإيجابيات الفعلية التي تم تحديدها بشكل صحيح |
| AUC-ROC | "مدى نجاح النموذج في الفصل بين الطبقات" | المنطقة الواقعة تحت منحنى المعدل الإيجابي الحقيقي مقابل المعدل الإيجابي الكاذب عبر جميع العتبات، من 0.5 (عشوائي) إلى 1.0 (مثالي) |
| R-مربع | "ما مقدار التباين الموضح" | 1 - (مجموع المربعات المتبقية / إجمالي مجموع المربعات): جزء التباين المستهدف الذي يلتقطه النموذج |
| تسرب البيانات | "الموديل المغشوش" | استخدام المعلومات أثناء التدريب التي قد لا تكون متاحة في وقت التنبؤ، مما يؤدي إلى تقييم متفائل |
| منحنى التعلم | "كيف يتغير الأداء مع المزيد من البيانات" | مخطط لدرجات التدريب والتحقق من الصحة مقابل حجم مجموعة التدريب، مما يكشف عن عدم الملائمة أو الإفراط في التجهيز |
| الانقسام الطبقي | "المحافظة على توازن النسب الطبقية" | تقسيم البيانات بحيث تحتوي كل مجموعة فرعية على نفس النسبة من كل فئة مثل مجموعة البيانات الكاملة |
## مزيد من القراءة
- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - مرجع شامل حول التحقق من الصحة والمقاييس وضبط المعلمات الفائقة
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - شرح واضح مع الأمثلة التفاعلية
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - معالجة صارمة لمتى ولماذا تنجح استراتيجيات CV المختلفة