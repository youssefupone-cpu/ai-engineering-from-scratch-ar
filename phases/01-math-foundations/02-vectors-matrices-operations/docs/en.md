# Vectors, Matrices & Operations

> كل شبكة عصبية هي مجرد عملية ضرب للمصفوفات بخطوات إضافية.

**النوع:** بناء
** اللغات: ** بايثون، جوليا
**المتطلبات الأساسية:** المرحلة الأولى، الدرس 01 (حدس الجبر الخطي)
**الوقت:** ~60 دقيقة

## Learning Objectives

- بناء فئة مصفوفة مع العمليات الحكيمة للعنصر، وضرب المصفوفة، والتحويل، والمحدد، والعكس
- التمييز بين الضرب حسب العناصر وضرب المصفوفات وشرح متى ينطبق كل منهما
- تنفيذ طبقة شبكة عصبية كثيفة واحدة (`relu(W @ x + b)`) باستخدام فئة Matrix من الصفر فقط
- شرح قواعد البث وكيفية عمل إضافة التحيز في أطر الشبكات العصبية

## The Problem

تريد بناء شبكة عصبية. قرأت الكود ورأيت هذا:

```
output = activation(weights @ input + bias)
```

أن `@` هو ضرب المصفوفة. `weights` عبارة عن مصفوفة. `input` هو ناقل. إذا كنت لا تعرف ما تفعله تلك العمليات، فهذا الخط سحري. إذا كنت تعرف، فهذا هو التمرير الأمامي الكامل للطبقة في ثلاث عمليات.

كل صورة يعالجها نموذجك هي مصفوفة لقيم البكسل. كل تضمين كلمة هو ناقل. كل طبقة من كل شبكة عصبية هي عبارة عن تحويل مصفوفة. لا يمكنك بناء أنظمة AI دون أن تتقن عمليات المصفوفة بنفس الطريقة التي لا يمكنك بها كتابة التعليمات البرمجية دون فهم المتغيرات.

هذا الدرس يبني تلك الطلاقة من الصفر.

## The Concept

### Vectors: ordered lists of numbers

المتجه عبارة عن قائمة من الأرقام ذات الاتجاه والحجم. في AI، تمثل المتجهات نقاط البيانات أو الميزات أو المعلمات.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

يشير المتجه ثنائي الأبعاد `[3, 4]` إلى الإحداثيات (3، 4) على المستوى. طوله (حجمه) هو 5 (المثلث 3-4-5).

### Matrices: grids of numbers

المصفوفة هي شبكة ثنائية الأبعاد. الصفوف والأعمدة. تحتوي المصفوفة m x n على صفوف m وأعمدة n.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

في الشبكات العصبية، تقوم مصفوفات الوزن بتحويل متجهات الإدخال إلى متجهات الإخراج. تستخدم الطبقة التي تحتوي على 784 مدخلاً و128 مخرجًا مصفوفة وزن 128 × 784.

### Why shapes matter

ضرب المصفوفة له قاعدة صارمة: `(m x n) @ (n x p) = (m x p)`. يجب أن تتطابق الأبعاد الداخلية.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

إذا حصلت على خطأ عدم تطابق الشكل في PyTorch، فهذا هو السبب.

### The operations map

| عملية | ماذا يفعل | استخدام الشبكة العصبية |
|-----------|------------|------------------|
| اضافة | الجمع بين العناصر | إضافة التحيز إلى الإخراج |
| الضرب العددي | مقياس كل عنصر | معدل التعلم * التدرجات |
| ضرب المصفوفة | تحويل المتجهات | تمرير طبقة إلى الأمام |
| تبديل | قلب الصفوف والأعمدة | الانتشار العكسي |
| المحدد | ملخص الرقم المفرد | التحقق من الانعكاس |
| معكوس | التراجع عن التحويل | حل الأنظمة الخطية |
| الهوية | مصفوفة عدم القيام بأي شيء | التهيئة، الاتصالات المتبقية |

### Element-wise vs matrix multiplication

هذا التمييز يزعج المبتدئين باستمرار.

من حيث العنصر: مضاعفة المواضع المطابقة. يجب أن تكون كلا المصفوفتين بنفس الشكل.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

ضرب المصفوفة: المنتجات النقطية للصفوف والأعمدة. يجب أن تتطابق الأبعاد الداخلية.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

عمليات مختلفة، نتائج مختلفة، قواعد مختلفة.

### Broadcasting

عند إضافة متجه انحياز إلى مصفوفة من المخرجات، لا تتطابق الأشكال. يقوم البث بتمديد المصفوفة الأصغر لتناسبها.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

كل إطار عمل حديث يقوم بذلك تلقائيًا. إن فهم ذلك يمنع الارتباك عندما تبدو الأشكال خاطئة ولكن الكود يعمل.

## Build It

### Step 1: Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: Matrix class with core operations

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: See it work

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect to neural networks

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

هذه طبقة كثيفة واحدة: `output = relu(W @ x + b)`. كل طبقة كثيفة في كل شبكة عصبية تفعل هذا بالضبط.

## Use It

NumPy يفعل كل شيء أعلاه في عدد أقل من الخطوط وأوامر الحجم بشكل أسرع.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

المشغل `@` في بايثون يستدعي `__matmul__`. NumPy ينفذها من خلال إجراءات BLAS محسنة مكتوبة بلغة C وFortran. نفس الرياضيات، أسرع 100 مرة.

البث في NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy يبث التحيز 1D تلقائيًا عبر كلا الصفين. هذه هي الطريقة التي تعمل بها إضافة التحيز في كل إطار عمل للشبكة العصبية.

## Ship It

يقدم هذا الدرس دافعًا لتدريس عمليات المصفوفة من خلال الحدس الهندسي. انظر `outputs/prompt-matrix-operations.md`.

تعتبر فئة Matrix المبنية هنا الأساس لإطار عمل الشبكة العصبية المصغرة الذي قمنا ببنائه في المرحلة 3، الدرس 10.

## Exercises

1. **تحقق من العكس.** اضرب `A @ A.inverse_2x2()` وتأكد من حصولك على مصفوفة الهوية. جرب ذلك باستخدام ثلاث مصفوفات مختلفة 2x2. ماذا يحدث عندما يكون المحدد صفر؟

2. **تنفيذ معكوس 3x3.** قم بتوسيع فئة المصفوفة لحساب المعكوسات لمصفوفات 3x3 باستخدام الطريقة المرافقة. اختبرها مقابل NumPy `np.linalg.inv`.

3. **قم ببناء شبكة من طبقتين.** باستخدام فئة Matrix الخاصة بك فقط (رقم NumPy)، قم بإنشاء شبكة عصبية من طبقتين: الإدخال (3) -> مخفي (4) -> الإخراج (2). قم بتهيئة الأوزان العشوائية، وقم بتشغيل التمريرة الأمامية، وتحقق من صحة جميع الأشكال.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| ناقل | "السهم" | قائمة مرتبة من الأرقام. في AI: نقطة في الفضاء عالي الأبعاد. |
| مصفوفة | "جدول الأرقام" | التحول الخطي. يقوم بتعيين المتجهات من مساحة إلى أخرى. |
| ضرب المصفوفة | "فقط اضرب الأرقام" | منتجات النقاط بين كل صف من المصفوفة الأولى وكل عمود من الثانية. النظام مهم. |
| تبديل | "اقلبها" | مبادلة الصفوف والأعمدة. يحول مصفوفة m x n إلى n x m. حاسمة في الانتشار العكسي. |
| المحدد | "بعض الأرقام من المصفوفة" | يقيس مدى قياس مساحة المصفوفة (ثنائي الأبعاد) أو الحجم (ثلاثي الأبعاد). الصفر يعني أن التحول يسحق البعد. |
| معكوس | "التراجع عن المصفوفة" | المصفوفة التي تعكس التحول. يوجد فقط عندما لا يكون المحدد صفرًا. |
| مصفوفة الهوية | "المصفوفة المملة" | المصفوفة المكافئة للضرب بـ 1. تستخدم في الاتصالات المتبقية (ResNets). |
| البث | "تثبيت الشكل السحري" | تمديد مصفوفة أصغر لتتناسب مع مصفوفة أكبر من خلال التكرار على طول الأبعاد المفقودة. |
| العنصر الحكيم | "الضرب العادي" | مضاعفة المواقف المطابقة. يجب أن يكون لكلا المصفوفتين نفس الشكل (أو أن يكونا قابلين للبث). |

## Further Reading

- [3Blue1Brown: جوهر الجبر الخطي](https://www.3blue1brown.com/topics/linear-algebra) - الحدس البصري لكل عملية يتم تناولها هنا
- [NumPy توثيق البث](https://numpy.org/doc/stable/user/basics.broadcasting.html) - القواعد الدقيقة NumPy فيما يلي
- [مراجعة الجبر الخطي في جامعة ستانفورد CS229](http://cs229.stanford.edu/section/cs229-linalg.pdf) - مرجع موجز للجبر الخطي الخاص بـ ML
