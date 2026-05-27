"""نموذج الانحدار الذاتي البصري للعبة (VAR): التنبؤ بالمقياس التالي على الهرم. الحد الأدنى من تنفيذ آلية VAR الموضحة في
مستندات/en.md. ثلاث قطع: 1. رمز مميز متبقي متعدد المقاييس VQ على "صور" صغيرة مقاس 8 × 8 (حجم صغير مكتبة الأنماط: الصلبة، المتدرجة، الدائرية، المدقق، المتقاطع). الرموز في يرمز المقياس k إلى المتبقي من المقاييس 1..k-1. جهاز فك التشفير هو مجموع التضمينات على نطاق upsampled.
2. متنبئ النطاق التالي المكيف (لوجستي / softmax mini-LM) على المفردات الصغيرة). يتم تقريب "المحول" حسب المقياس الرسوم البيانية الشرطية. الهندسة التي يعلمها الدرس هي التكييف المرتب على نطاق واسع والتنبؤ الموازي داخل النطاق، ليس اهتماما عميقا.
3. تمر حلقة الجيل التي تقوم بتشغيل محول K (واحدة لكل مقياس) و عينات كل موقف بالمقياس الحالي بالتوازي من مشروط. تعمل المبالغ المفككة من تضمينات المقياس على إعادة بناء الصورة. النقطة المهمة هي ممارسة بيانات التدريب ذات المقياس المتوازي
أخذ العينات ضمن النطاق، وإعادة الإعمار المتبقية-VQ. حقيقي VAR
يقوم بتبديل الرسم البياني لمحول ومكتبة الأنماط لـ
مجموعة بيانات الصورة؛ الحزام من حولهم يبقى كما هو. Stdlib + numpy فقط. تشغيل: بيثون main.py
"""

from __future__ import annotations

import numpy as np


IMG = 8
SCALES = (1, 2, 4, 8)
CODEBOOK = 16


def make_patterns(rng: np.random.Generator, n: int) -> np.ndarray:
    """قم بإرجاع أنماط 8 × 8 ذات تدرج رمادي مأخوذة من مكتبة صغيرة."""
    out = np.zeros((n, IMG, IMG), dtype=np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    for i in range(n):
        kind = int(rng.integers(0, 5))
        if kind == 0:
            out[i] = rng.uniform(0.1, 0.9)
        elif kind == 1:
            out[i] = (xx + yy) / (2 * (IMG - 1))
        elif kind == 2:
            cx, cy = IMG / 2 - 0.5, IMG / 2 - 0.5
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            out[i] = np.clip(1.0 - r / (IMG / 2), 0.0, 1.0)
        elif kind == 3:
            out[i] = ((xx.astype(int) + yy.astype(int)) % 2).astype(np.float32)
        else:
            mid = IMG // 2
            cross = ((xx == mid) | (yy == mid)).astype(np.float32)
            out[i] = cross * 0.9 + 0.05
    return out


def fit_codebook(samples: np.ndarray, k: int, iters: int = 30,
                 seed: int = 0) -> np.ndarray:
    """k-يعني على العينات العددية؛ إرجاع كتاب الشفرات بطول k."""
    rng = np.random.default_rng(seed)
    flat = samples.reshape(-1)
    if flat.size < k:
        raise ValueError(f"need >= {k} samples for codebook init, got {flat.size}")
    idx = rng.choice(flat.size, size=k, replace=False)
    centers = flat[idx].astype(np.float32)
    for _ in range(iters):
        dists = (flat[:, None] - centers[None, :]) ** 2
        assign = dists.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = flat[mask].mean()
    return np.sort(centers)


def encode(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """قم بإطباق كل قيمة على أقرب رمز؛ إرجاع الرموز الصحيحة."""
    dists = (values[..., None] - codebook[None, None, :]) ** 2
    return dists.argmin(axis=-1).astype(np.int32)


def downsample(img: np.ndarray, target: int) -> np.ndarray:
    """قم بتجميع صورة HxW في المتوسط ​​وصولاً إلى الهدف x."""
    h, w = img.shape
    if target == h:
        return img.copy()
    factor = h // target
    return img.reshape(target, factor, target, factor).mean(axis=(1, 3))


def upsample(grid: np.ndarray, target: int) -> np.ndarray:
    """يقوم أقرب جار بتكوين شبكة HxW حتى الهدف x المستهدف."""
    h, w = grid.shape
    if target == h:
        return grid.copy()
    factor = target // h
    return grid.repeat(factor, axis=0).repeat(factor, axis=1)


def tokenize_multiscale(img: np.ndarray, codebooks: list[np.ndarray]
                        ) -> list[np.ndarray]:
    """المتبقي VQ: كل مقياس يرمز إلى ما فاته المقاييس السابقة."""
    residual = img.copy()
    tokens: list[np.ndarray] = []
    for scale, book in zip(SCALES, codebooks):
        coarse = downsample(residual, scale)
        tok = encode(coarse, book)
        recon = book[tok]
        residual = residual - upsample(recon, IMG)
        tokens.append(tok)
    return tokens


def detokenize_multiscale(tokens: list[np.ndarray],
                          codebooks: list[np.ndarray]) -> np.ndarray:
    """وحدة فك الترميز: مجموع التضمينات ذات الحجم المضخم."""
    out = np.zeros((IMG, IMG), dtype=np.float32)
    for tok, book, scale in zip(tokens, codebooks, SCALES):
        out = out + upsample(book[tok], IMG)
    return out


def train_codebooks(images: np.ndarray) -> list[np.ndarray]:
    """قم بملاءمة دفاتر الرموز لكل مقياس على بقايا مجموعة صور صغيرة."""
    residuals = images.copy()
    books: list[np.ndarray] = []
    for scale in SCALES:
        pooled = np.stack([downsample(r, scale) for r in residuals])
        book = fit_codebook(pooled, CODEBOOK)
        books.append(book)
        recon = np.stack([upsample(book[encode(p[None], book)[0]], IMG)
                          for p in pooled])
        residuals = residuals - recon
    return books


def context_key(prev_tokens: list[np.ndarray]) -> tuple:
    """ملخص قابل للتجزئة لجميع الرموز المميزة للمقاييس السابقة."""
    return tuple(int(t.mean() * 1000) for t in prev_tokens) if prev_tokens else ()


def fit_predictor(token_streams: list[list[np.ndarray]]
                  ) -> list[dict[tuple, np.ndarray]]:
    """رسم بياني شرطي واحد لكل مقياس، مرتبط بملخص المقياس السابق. هذا يمثل المحول: في وقت التدريب، قم بإحصاء الرموز المميزة تظهر بمقياس k مشروطًا بالملخص الخشن للمقاييس 1..k-1.
    """
    predictors: list[dict[tuple, np.ndarray]] = [
        {} for _ in SCALES
    ]
    for stream in token_streams:
        for k in range(len(SCALES)):
            ctx = context_key(stream[:k])
            table = predictors[k].setdefault(ctx, np.ones(CODEBOOK,
                                                          dtype=np.float64))
            for tok in stream[k].reshape(-1):
                table[int(tok)] += 1.0
    for table in predictors:
        for key, counts in table.items():
            table[key] = counts / counts.sum()
    return predictors


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def generate(predictors: list[dict[tuple, np.ndarray]],
             codebooks: list[np.ndarray],
             rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """عينة VAR واحدة: تمريرات K، متوازية داخل المقياس، سببية عبر المقاييس."""
    drawn: list[np.ndarray] = []
    for k, scale in enumerate(SCALES):
        ctx = context_key(drawn[:k])
        table = predictors[k]
        probs = table.get(ctx)
        if probs is None:
            probs = np.ones(CODEBOOK) / CODEBOOK
        size = scale * scale
        flat = np.array([sample_categorical(probs, rng) for _ in range(size)],
                        dtype=np.int32)
        drawn.append(flat.reshape(scale, scale))
    image = detokenize_multiscale(drawn, codebooks)
    return image, drawn


def reconstruction_mse(images: np.ndarray,
                       codebooks: list[np.ndarray]) -> float:
    errs = []
    for img in images:
        toks = tokenize_multiscale(img, codebooks)
        recon = detokenize_multiscale(toks, codebooks)
        errs.append(float(np.mean((recon - img) ** 2)))
    return float(np.mean(errs))


def main() -> None:
    rng = np.random.default_rng(0)
    train_imgs = make_patterns(rng, 64)
    val_imgs = make_patterns(rng, 16)

    codebooks = train_codebooks(train_imgs)
    train_token_streams = [tokenize_multiscale(img, codebooks) for img in train_imgs]
    predictors = fit_predictor(train_token_streams)

    print(f"image size: {IMG}x{IMG}")
    print(f"scales: {SCALES}")
    print(f"codebook size per scale: {CODEBOOK}")
    print(f"reconstruction MSE on train: {reconstruction_mse(train_imgs, codebooks):.5f}")
    print(f"reconstruction MSE on val:   {reconstruction_mse(val_imgs, codebooks):.5f}")

    print()
    print("generation: 4 transformer passes, all positions parallel within a scale")
    for trial in range(3):
        img, toks = generate(predictors, codebooks, rng)
        shapes = [t.shape for t in toks]
        print(f"  trial {trial}: scales={shapes}  range=[{img.min():.2f}, {img.max():.2f}]")

    print()
    print("scale-ordered attention check: every scale k only sees scales 1..k-1")
    for k, scale in enumerate(SCALES):
        n_pos = scale * scale
        prior_seen = sum(s * s for s in SCALES[:k])
        print(f"  scale {k} (size {scale}x{scale}, {n_pos} tokens):"
              f" attends to {prior_seen} prior tokens")


if __name__ == "__main__":
    main()
