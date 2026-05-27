import math
from collections import Counter


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)


def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    if total == 0:
        return [0.0] * len(vocab)
    return [count.get(word, 0) / total for word in vocab]


def compute_idf(documents, vocab):
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf


def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def vector_search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        total = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            total += idf * numerator / denominator

        return total

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, retrieval_pool=15):
    query_emb = tfidf_embed(query, vocab, idf)
    vec_results = vector_search(query_emb, vector_embeddings, top_k=retrieval_pool)
    bm25_results = bm25_index.search(query, top_k=retrieval_pool)
    fused = reciprocal_rank_fusion([vec_results, bm25_results])
    return fused[:top_k]


def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define the process and requirements.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request for {topic}. Then, the system processes it according to the defined rules and policies.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to {topic} that provide a comprehensive answer to this question."
    }
    query_lower = query.lower().strip()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    filler = {"what", "is", "the", "how", "do", "does", "a", "an", "for", "of",
              "to", "in", "on", "at", "by", "and", "or", "are", "was", "were", "?"}
    topic_words = [w.strip("?.,!") for w in query.lower().split() if w.strip("?.,!") not in filler]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = vector_search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis


def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent


def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results


def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return (
        "Answer the question based ONLY on the following context.\n"
        "If the context doesn't contain enough information, "
        "say \"I don't have enough information to answer that.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


SAMPLE_DOCUMENTS = [
    """سياسة استرداد الأموال لشركة Acme Corp. جميع عملاء الخطة القياسية مؤهلون لاسترداد أموالهم بالكامل خلال 30 يومًا من تاريخ الشراء. يحصل عملاء خطة Enterprise على فترة استرداد ممتدة مدتها 60 يومًا مع عمليات استرداد مقسمة بشكل تناسبي تحسب من تاريخ الإلغاء. تتم معالجة المبالغ المستردة في غضون 5-7 أيام عمل والعودة إلى طريقة الدفع الأصلية. لا توجد مبالغ مستردة متاحة بعد استرداد الأموال تغلق النافذة. يجب على العملاء تقديم طلبات استرداد الأموال من خلال بوابة الدعم أو عن طريق الاتصال بمدير حساباتهم مباشرة. الاشتراكات السنوية التي يتم إلغاؤها في منتصف المدة سيحصل على رصيد متناسب للأشهر المتبقية.""",

    """نظرة عامة على منتج شركة Acme. تقدم شركة Acme Corp ثلاثة مستويات من المنتجات: Starter وProfessional وEnterprise. تتضمن خطة Starter ميزات أساسية للمستخدمين الفرديين بسعر 29 دولارًا شهريًا. تضيف الخطة الاحترافية تعاون الفريق والتحليلات المتقدمة والأولوية دعم بمبلغ 99 دولارًا شهريًا لكل مستخدم. تتضمن خطة Enterprise كل شيء في عمليات تكامل احترافية ومخصصة، وإدارة حساب مخصصة، SSO، سجلات التدقيق، ووقت تشغيل بنسبة 99.99% SLA. تسعير المؤسسة مخصص ويبدأ بسعر 500 دولار شهريًا لما يصل إلى 50 مستخدمًا. تتضمن جميع الخطط نسخة تجريبية مجانية مدتها 14 يومًا مع عدم وجود بطاقة الائتمان المطلوبة.""",

    """الممارسات الأمنية لشركة Acme Corp. تحافظ شركة Acme Corp على امتثال SOC 2 من النوع II وتخضع سنويًا لطرف ثالث عمليات التدقيق الأمني. يتم تشفير جميع البيانات أثناء عدم النشاط باستخدام AES-256 وأثناء النقل باستخدام TLS 1.3. يتم تخزين بيانات العميل في مستأجرين معزولين خلال AWS مناطق شرق الولايات المتحدة 1 وغرب الاتحاد الأوروبي 1. يمكن تكوين إقامة البيانات لكل تنظيم لعملاء المؤسسات. يتم إجراء النسخ الاحتياطية كل 6 ساعات مع احتفاظ لمدة 30 يومًا. لا تقوم شركة Acme Corp ببيع بيانات العملاء أو مشاركتها معها أطراف ثالثة. يمكن لعملاء المؤسسات طلب حذف البيانات خلال 24 ساعة. برنامج مكافأة الأخطاء متاح من خلال HackerOne.""",

    """Acme Corp API الوثائق. يستخدم Acme API REST مع JSON نصوص الطلب والاستجابة. المصادقة يتم عبر الرموز المميزة لحاملها الصادرة من خلال OAuth 2.0. حدود المعدل هي 100 طلب للدقيقة للمبتدئين، و1000 للمحترفين، و10000 للمؤسسات. يتم تضمين رؤوس حدود المعدل في كل استجابة: X-RateLimit-Limit، X-RateLimit المتبقية، وX-RateLimit-Reset. تجاوز الحد الأقصى للسعر تُرجع HTTP 429 برأس "إعادة المحاولة بعد". يدعم API ترقيم الصفحات عبر ترقيم الصفحات المستند إلى المؤشر باستخدام حقل next_cursor. خطافات الويب هي متاح لإشعارات الأحداث في الوقت الفعلي على Professional وEnterprise خطط. يستخدم إصدار API الإصدارات المستندة إلى التاريخ في المسار URL.""",

    """Acme Corp Q3 تقرير أرباح 2025. بلغ إجمالي الإيرادات لعام Q3 لعام 2025 47.2 مليون دولار أمريكي، بزيادة قدرها 23% على أساس سنوي. وساهم قطاع المؤسسات بمبلغ 31.8 مليون دولار، وهو ما يمثل 67% من الإجمالي الإيرادات. أضاف القطاع المهني 12.1 مليون دولار. الجزء المبتدئ ساهم بمبلغ 3.3 مليون دولار. ارتفع عدد العملاء إلى 14200 من 11800 في Q3 2024. بلغ صافي معدل الاحتفاظ 118%. وكانت مصاريف التشغيل 38.4 مليون دولار. EBITDA كان 8.8 مليون دولار بهامش 18.6%. وبلغ التدفق النقدي الحر 6.2 مليون دولار. التوجيهات الخاصة بـ Q4 2025 هي 51-53 مليون دولار في الإيرادات مع استمرار توسع الهامش.""",

    """مدة تشغيل شركة Acme Corp وموثوقيتها. تضمن شركة Acme Corp وقت تشغيل بنسبة 99.9% للخطط الاحترافية ووقت تشغيل بنسبة 99.99% لخطط المؤسسة. يتم حساب وقت التشغيل شهريًا باستثناء المقرر نوافذ الصيانة التي يتم الإعلان عنها قبل 72 ساعة. إذا الجهوزية إذا انخفض إلى ما دون المستوى المضمون، يحصل العملاء على أرصدة الخدمة: رصيد 10% لكل 0.1% أقل من الحد SLA، بحد أقصى 30% من الرسوم الشهرية. يجب طلب أرصدة الخدمة خلال 30 يومًا من الحادث. يتم نشر تحديثات صفحة الحالة على Status.acme.com في غضون 5 دقائق من أي حادث تم اكتشافه. تقارير ما بعد الحادث هي يتم النشر خلال 48 ساعة عن أي انقطاع يتجاوز 15 دقيقة."""
]


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 1: BM25 Keyword Search")
    print("=" * 65)

    all_chunks = []
    chunk_sources = []
    source_names = ["refund", "product", "security", "api", "earnings", "uptime"]
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        doc_chunks = chunk_text(doc, chunk_size=50, overlap=10)
        for c in doc_chunks:
            all_chunks.append(c)
            chunk_sources.append(source_names[i])

    bm25 = BM25()
    bm25.index(all_chunks)

    test_query = "What was revenue last quarter?"
    bm25_results = bm25.search(test_query, top_k=5)
    print(f"  Query: {test_query}")
    print(f"  BM25 top-5:")
    for rank, (idx, score) in enumerate(bm25_results):
        preview = all_chunks[idx][:70].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 2: Vector Search vs BM25")
    print("=" * 65)

    vocab = build_vocabulary(all_chunks)
    idf = compute_idf(all_chunks, vocab)
    embeddings = [tfidf_embed(c, vocab, idf) for c in all_chunks]

    queries = [
        "What is the refund policy for enterprise customers?",
        "What was revenue last quarter?",
        "How is data encrypted?",
        "What are the API rate limits for enterprise?",
        "What happens if uptime falls below SLA?"
    ]

    for query in queries:
        query_emb = tfidf_embed(query, vocab, idf)
        vec_top1 = vector_search(query_emb, embeddings, top_k=1)[0]
        bm25_top1 = bm25.search(query, top_k=1)[0]

        print(f"\n  Query: {query}")
        print(f"    Vector #1: [{chunk_sources[vec_top1[0]]}] score={vec_top1[1]:.4f}")
        print(f"    BM25   #1: [{chunk_sources[bm25_top1[0]]}] score={bm25_top1[1]:.4f}")
        agree = "AGREE" if chunk_sources[vec_top1[0]] == chunk_sources[bm25_top1[0]] else "DISAGREE"
        print(f"    {agree}")

    print("\n" + "=" * 65)
    print("STEP 3: Reciprocal Rank Fusion (Hybrid Search)")
    print("=" * 65)

    query = "What was revenue last quarter?"
    print(f"  Query: {query}")

    query_emb = tfidf_embed(query, vocab, idf)
    vec_results = vector_search(query_emb, embeddings, top_k=10)
    bm25_results = bm25.search(query, top_k=10)

    print(f"\n  Vector top-3:")
    for rank, (idx, score) in enumerate(vec_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  BM25 top-3:")
    for rank, (idx, score) in enumerate(bm25_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    fused = reciprocal_rank_fusion([vec_results, bm25_results])
    print(f"\n  RRF fused top-5:")
    for rank, (idx, score) in enumerate(fused[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] rrf={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 4: Reranking")
    print("=" * 65)

    query = "enterprise refund policy"
    print(f"  Query: {query}")

    hybrid_results = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
    reranked = rerank(query, hybrid_results, all_chunks)

    print(f"\n  Before reranking (top-5):")
    for rank, (idx, score) in enumerate(hybrid_results[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print(f"\n  After reranking (top-5):")
    for rank, (idx, score) in enumerate(reranked[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 5: HyDE (Hypothetical Document Embeddings)")
    print("=" * 65)

    query = "How much money did the company make?"
    print(f"  Query: {query}")
    print(f"  (Note: query uses 'money', docs use 'revenue' and 'earnings')")

    query_emb = tfidf_embed(query, vocab, idf)
    direct_results = vector_search(query_emb, embeddings, top_k=3)
    hyde_results, hypothesis = hyde_search(query, embeddings, vocab, idf, top_k=3)

    print(f"\n  Hypothesis: {hypothesis[:100]}...")

    print(f"\n  Direct search top-3:")
    for rank, (idx, score) in enumerate(direct_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  HyDE search top-3:")
    for rank, (idx, score) in enumerate(hyde_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print("\n" + "=" * 65)
    print("STEP 6: Parent-Child Chunking")
    print("=" * 65)

    full_text = " ".join(SAMPLE_DOCUMENTS)
    parents, children, child_to_parent = create_parent_child_chunks(
        full_text, parent_size=100, child_size=25
    )

    print(f"  Total words: {len(full_text.split())}")
    print(f"  Parent chunks: {len(parents)} (100 words each)")
    print(f"  Child chunks: {len(children)} (25 words each)")
    print(f"  Ratio: {len(children)/len(parents):.1f} children per parent")

    child_vocab = build_vocabulary(children)
    child_idf = compute_idf(children, child_vocab)
    child_embeddings = [tfidf_embed(c, child_vocab, child_idf) for c in children]

    query = "enterprise refund 60 days"
    query_emb = tfidf_embed(query, child_vocab, child_idf)
    child_results = vector_search(query_emb, child_embeddings, top_k=3)

    print(f"\n  Query: {query}")
    print(f"\n  Matched children:")
    for rank, (idx, score) in enumerate(child_results):
        parent_idx = child_to_parent[idx]
        print(f"    Child #{idx} (score={score:.4f}):")
        print(f"      Child text: {children[idx][:80]}...")
        print(f"      Parent #{parent_idx}: {parents[parent_idx][:80]}...")

    print("\n" + "=" * 65)
    print("STEP 7: Faithfulness Evaluation")
    print("=" * 65)

    good_answer = (
        "Enterprise customers receive a 60-day refund window. "
        "Refunds are pro-rated from the date of cancellation. "
        "Processing takes 5-7 business days."
    )
    bad_answer = (
        "Enterprise customers receive a 90-day refund window. "
        "Refunds are processed instantly. "
        "There is a $50 processing fee."
    )
    context_chunks = [all_chunks[i] for i, _ in hybrid_search(
        "enterprise refund", all_chunks, embeddings, vocab, idf, bm25, top_k=3
    )]

    good_score, good_ungrounded = evaluate_faithfulness(good_answer, context_chunks)
    bad_score, bad_ungrounded = evaluate_faithfulness(bad_answer, context_chunks)

    print(f"  Context: {len(context_chunks)} chunks about refund policy")
    print(f"\n  Good answer: \"{good_answer[:80]}...\"")
    print(f"  Faithfulness: {good_score:.2f}")
    if good_ungrounded:
        print(f"  Ungrounded claims: {good_ungrounded}")
    else:
        print(f"  All claims grounded in context.")

    print(f"\n  Bad answer: \"{bad_answer[:80]}...\"")
    print(f"  Faithfulness: {bad_score:.2f}")
    if bad_ungrounded:
        print(f"  Ungrounded claims:")
        for claim in bad_ungrounded:
            print(f"    - \"{claim}\"")

    print("\n" + "=" * 65)
    print("STEP 8: Full Advanced RAG Pipeline Comparison")
    print("=" * 65)

    comparison_queries = [
        ("What is the refund policy for enterprise?", "refund"),
        ("What was Q3 revenue?", "earnings"),
        ("How is customer data encrypted?", "security"),
        ("What are the API rate limits?", "api"),
        ("What is the uptime guarantee?", "uptime"),
    ]

    print(f"  {'Query':<45s} {'Vector':>8s} {'BM25':>8s} {'Hybrid':>8s} {'Rerank':>8s}")
    print("  " + "-" * 77)

    for query, expected_source in comparison_queries:
        query_emb = tfidf_embed(query, vocab, idf)

        vec_top = vector_search(query_emb, embeddings, top_k=1)[0]
        vec_hit = "HIT" if chunk_sources[vec_top[0]] == expected_source else "miss"

        bm25_top = bm25.search(query, top_k=1)[0]
        bm25_hit = "HIT" if chunk_sources[bm25_top[0]] == expected_source else "miss"

        hybrid_top = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=1)[0]
        hybrid_hit = "HIT" if chunk_sources[hybrid_top[0]] == expected_source else "miss"

        hybrid_pool = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
        reranked_top = rerank(query, hybrid_pool, all_chunks)[0]
        rerank_hit = "HIT" if chunk_sources[reranked_top[0]] == expected_source else "miss"

        print(f"  {query:<45s} {vec_hit:>8s} {bm25_hit:>8s} {hybrid_hit:>8s} {rerank_hit:>8s}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("  Advanced RAG techniques:")
    print("    1. BM25 keyword search catches exact term matches")
    print("    2. Hybrid search (vector + BM25 + RRF) combines both signals")
    print("    3. Reranking scores candidates more carefully with cross-attention")
    print("    4. HyDE bridges the query-document vocabulary gap")
    print("    5. Parent-child chunking: precise search, rich context")
    print("    6. Faithfulness evaluation catches hallucinated claims")
    print("\n  In production:")
    print("    - Replace TF-IDF with neural embeddings")
    print("    - Replace the simple reranker with a cross-encoder model")
    print("    - Replace HyDE templates with actual LLM hypothesis generation")
    print("    - Add metadata filtering before search")
    print("    - Evaluate with Recall@k and faithfulness on a test set")
