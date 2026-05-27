"""BERT نمذجة اللغة المقنعة - تم إزالة الغموض عن قواعد الإخفاء. ستدلب خالص. إظهار قاعدة 80/10/10 وإخفاء الكلمة بالكامل و
التحقق من سلامة التوزيع على مجموعة كبيرة من الرموز المميزة.
"""

import random
from collections import Counter


MASK_ID = 0  # reserve id 0 for [MASK] in this toy vocab
CLS_ID = 1
SEP_ID = 2
SPECIAL_IDS = {MASK_ID, CLS_ID, SEP_ID}
IGNORE_INDEX = -100


def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    """تطبيق BERT اخفاء. إرجاع (input_ids، التسميات). التسميات[i] = الرمز الأصلي إذا كان الموضع تم تحديده للتنبؤ، IGNORE_INDEX بخلاف ذلك.
    """
    if rng is None:
        rng = random.Random()
    input_ids = list(tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for i, t in enumerate(tokens):
        if t in SPECIAL_IDS:
            continue
        if rng.random() >= mask_prob:
            continue
        labels[i] = t
        r = rng.random()
        if r < 0.8:
            input_ids[i] = MASK_ID
        elif r < 0.9:
            rand_id = t
            while rand_id in SPECIAL_IDS or rand_id == t:
                rand_id = rng.randrange(vocab_size)
            input_ids[i] = rand_id
    return input_ids, labels


def whole_word_mlm(tokens, word_spans, vocab_size, mask_prob=0.15, rng=None):
    """إخفاء الكلمات بأكملها: إذا تم تحديد أي كلمة فرعية في النطاق، قم بإخفاء الكل. word_spans: قائمة بالنطاقات نصف المفتوحة (البداية والنهاية) في الرموز المميزة.
    """
    if rng is None:
        rng = random.Random()
    input_ids = list(tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for start, end in word_spans:
        if any(tokens[i] in SPECIAL_IDS for i in range(start, end)):
            continue
        if rng.random() >= mask_prob:
            continue
        r = rng.random()
        if r < 0.8:
            for i in range(start, end):
                labels[i] = tokens[i]
                input_ids[i] = MASK_ID
        elif r < 0.9:
            for i in range(start, end):
                labels[i] = tokens[i]
                rand_id = tokens[i]
                while rand_id in SPECIAL_IDS or rand_id == tokens[i]:
                    rand_id = rng.randrange(vocab_size)
                input_ids[i] = rand_id
        else:
            for i in range(start, end):
                labels[i] = tokens[i]
    return input_ids, labels


def distribution_check(n_tokens, vocab_size, mask_prob=0.15, seed=42):
    rng = random.Random(seed)
    tokens = [rng.randrange(3, vocab_size) for _ in range(n_tokens)]
    input_ids, labels = create_mlm_batch(tokens, vocab_size, mask_prob, rng)

    selected = sum(1 for l in labels if l != IGNORE_INDEX)
    masked = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == MASK_ID)
    randomized = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t != MASK_ID and t != l)
    unchanged = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == l)

    return {
        "tokens": n_tokens,
        "selected": selected,
        "selected_pct": 100 * selected / n_tokens,
        "masked_of_selected_pct": 100 * masked / selected if selected else 0.0,
        "random_of_selected_pct": 100 * randomized / selected if selected else 0.0,
        "unchanged_of_selected_pct": 100 * unchanged / selected if selected else 0.0,
    }


def toy_predict(masked_inputs, vocab):
    """التظاهر بـ MLM head: يُرجع توزيعًا موحدًا على المفردات. يستخدم BERT الحقيقي مخرجات برنامج التشفير في كل موضع، ويتم إسقاطها على المفردات.
    """
    V = len(vocab)
    return [[1.0 / V for _ in range(V)] for _ in masked_inputs]


def main():
    vocab_words = [
        "[MASK]", "[CLS]", "[SEP]",
        "the", "quick", "brown", "fox", "jumps", "over",
        "lazy", "dog", "a", "stitch", "in", "time",
        "saves", "nine", "sat", "on", "mat",
    ]
    vocab_size = len(vocab_words)
    id_of = {w: i for i, w in enumerate(vocab_words)}

    sentence = ["[CLS]", "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"]
    tokens = [id_of[w] for w in sentence]
    rng = random.Random(42)
    inp, labels = create_mlm_batch(tokens, vocab_size, mask_prob=0.5, rng=rng)

    print("=== MLM masking demo (prob=0.5 so you can see it) ===")
    print(f"{'idx':>4}  {'word':>9}  {'input_id':>9}  {'input_word':>11}  {'label':>6}")
    for i, (t_in, t_orig, lab) in enumerate(zip(inp, tokens, labels)):
        print(f"{i:>4}  {vocab_words[t_orig]:>9}  {t_in:>9}  {vocab_words[t_in]:>11}  {lab:>6}")

    print()
    print("=== 80/10/10 distribution over 100k random tokens ===")
    stats = distribution_check(n_tokens=100_000, vocab_size=vocab_size, mask_prob=0.15)
    print(f"selected:                   {stats['selected_pct']:.2f}%   (target 15.0%)")
    print(f"  -> replaced with [MASK]:  {stats['masked_of_selected_pct']:.2f}%   (target 80.0%)")
    print(f"  -> replaced with random:  {stats['random_of_selected_pct']:.2f}%   (target 10.0%)")
    print(f"  -> left unchanged:        {stats['unchanged_of_selected_pct']:.2f}%   (target 10.0%)")

    print()
    print("=== whole-word masking demo ===")
    # تعامل مع "Quick Brown" و"Lazy Dog" على أنهما كلمتان فرعيتان للعرض التوضيحي
    tokens2 = [id_of[w] for w in sentence]
    spans = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 11)]
    rng2 = random.Random(7)
    inp2, labels2 = whole_word_mlm(tokens2, spans, vocab_size, mask_prob=0.5, rng=rng2)
    print("spans:        " + " ".join(f"[{s}:{e}]" for s, e in spans))
    print("input words:  " + " ".join(vocab_words[t] for t in inp2))
    print("label mask:   " + " ".join(("P" if l != IGNORE_INDEX else ".") for l in labels2))
    print("P = position has a label, . = ignored")


if __name__ == "__main__":
    main()
