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


def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


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


def simple_generate(prompt, retrieved_chunks):
    query_section = prompt.lower().split("question:")[-1]
    query_words = set(query_section.split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that"}
    query_words = query_words - stop_words

    best_sentence = ""
    best_score = 0

    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

    return best_sentence if best_sentence else "I don't have enough information."


class RAGPipeline:
    def __init__(self, chunk_size=200, overlap=50, top_k=5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []
        self.sources = []

    def index(self, documents, source_names=None):
        all_chunks = []
        all_sources = []
        for i, doc in enumerate(documents):
            doc_chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(doc_chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(doc_chunks))

        self.chunks = all_chunks
        self.sources = all_sources
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]
        return len(all_chunks)

    def query(self, question, top_k=None):
        k = top_k or self.top_k
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, k)

        retrieved = []
        for idx, score in results:
            retrieved.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "score": score,
                "index": idx
            })

        chunk_texts = [r["chunk"] for r in retrieved]
        prompt = build_rag_prompt(question, chunk_texts)
        answer = simple_generate(prompt, chunk_texts)

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "retrieved": retrieved
        }


SAMPLE_DOCUMENTS = [
    """سياسة استرداد الأموال لشركة Acme Corp. جميع عملاء الخطة القياسية مؤهلون لاسترداد أموالهم بالكامل خلال 30 يومًا من تاريخ الشراء. يحصل عملاء خطة Enterprise على فترة استرداد ممتدة مدتها 60 يومًا مع عمليات استرداد مقسمة بشكل تناسبي تحسب من تاريخ الإلغاء. تتم معالجة المبالغ المستردة في غضون 5-7 أيام عمل والعودة إلى طريقة الدفع الأصلية. لا توجد مبالغ مستردة متاحة بعد استرداد الأموال تغلق النافذة. يجب على العملاء تقديم طلبات استرداد الأموال من خلال بوابة الدعم أو عن طريق الاتصال بمدير حساباتهم مباشرة. الاشتراكات السنوية التي يتم إلغاؤها في منتصف المدة سيحصل على رصيد متناسب للأشهر المتبقية.""",

    """نظرة عامة على منتج شركة Acme. تقدم شركة Acme Corp ثلاثة مستويات من المنتجات: Starter وProfessional وEnterprise. تتضمن خطة Starter ميزات أساسية للمستخدمين الفرديين بسعر 29 دولارًا شهريًا. تضيف الخطة الاحترافية تعاون الفريق والتحليلات المتقدمة والأولوية دعم بمبلغ 99 دولارًا شهريًا لكل مستخدم. تتضمن خطة Enterprise كل شيء في عمليات تكامل احترافية ومخصصة، وإدارة حساب مخصصة، SSO، سجلات التدقيق، ووقت تشغيل بنسبة 99.99% SLA. تسعير المؤسسة مخصص ويبدأ بسعر 500 دولار شهريًا لما يصل إلى 50 مستخدمًا. تتضمن جميع الخطط نسخة تجريبية مجانية مدتها 14 يومًا مع عدم وجود بطاقة الائتمان المطلوبة.""",

    """الممارسات الأمنية لشركة Acme Corp. تحافظ شركة Acme Corp على امتثال SOC 2 من النوع II وتخضع سنويًا لطرف ثالث عمليات التدقيق الأمني. يتم تشفير جميع البيانات أثناء عدم النشاط باستخدام AES-256 وأثناء النقل باستخدام TLS 1.3. يتم تخزين بيانات العميل في مستأجرين معزولين خلال AWS مناطق شرق الولايات المتحدة 1 وغرب الاتحاد الأوروبي 1. يمكن تكوين إقامة البيانات لكل تنظيم لعملاء المؤسسات. يتم إجراء النسخ الاحتياطية كل 6 ساعات مع احتفاظ لمدة 30 يومًا. لا تقوم شركة Acme Corp ببيع بيانات العملاء أو مشاركتها معها أطراف ثالثة. يمكن لعملاء المؤسسات طلب حذف البيانات خلال 24 ساعة. برنامج مكافأة الأخطاء متاح من خلال HackerOne.""",

    """Acme Corp API الوثائق. يستخدم Acme API REST مع JSON نصوص الطلب والاستجابة. المصادقة يتم عبر الرموز المميزة لحاملها الصادرة من خلال OAuth 2.0. حدود المعدل هي 100 طلب للدقيقة للمبتدئين، و1000 للمحترفين، و10000 للمؤسسات. يتم تضمين رؤوس حدود المعدل في كل استجابة: X-RateLimit-Limit، X-RateLimit المتبقية، وX-RateLimit-Reset. تجاوز الحد الأقصى للسعر تُرجع HTTP 429 برأس "إعادة المحاولة بعد". يدعم API ترقيم الصفحات عبر ترقيم الصفحات المستند إلى المؤشر باستخدام حقل next_cursor. خطافات الويب هي متاح لإشعارات الأحداث في الوقت الفعلي على Professional وEnterprise خطط. يستخدم إصدار API الإصدارات المستندة إلى التاريخ في المسار URL.""",

    """مدة تشغيل شركة Acme Corp وموثوقيتها. تضمن شركة Acme Corp وقت تشغيل بنسبة 99.9% للخطط الاحترافية ووقت تشغيل بنسبة 99.99% لخطط المؤسسة. يتم حساب وقت التشغيل شهريًا باستثناء المقرر نوافذ الصيانة التي يتم الإعلان عنها قبل 72 ساعة. إذا الجهوزية إذا انخفض إلى ما دون المستوى المضمون، يحصل العملاء على أرصدة الخدمة: رصيد 10% لكل 0.1% أقل من الحد SLA، بحد أقصى 30% من الرسوم الشهرية. يجب طلب أرصدة الخدمة خلال 30 يومًا من الحادث. يتم نشر تحديثات صفحة الحالة على Status.acme.com في غضون 5 دقائق من أي حادث تم اكتشافه. تقارير ما بعد الحادث هي يتم النشر خلال 48 ساعة عن أي انقطاع يتجاوز 15 دقيقة."""
]


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Document Chunking")
    print("=" * 60)

    sample = SAMPLE_DOCUMENTS[0]
    chunks = chunk_text(sample, chunk_size=30, overlap=10)
    print(f"  Document length: {len(sample.split())} words")
    print(f"  Chunk size: 30 words, overlap: 10 words")
    print(f"  Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}: ({len(chunk.split())} words)")
        print(f"    {chunk[:100]}...")

    print("\n" + "=" * 60)
    print("STEP 2: TF-IDF Embedding")
    print("=" * 60)

    mini_docs = [
        "The cat sat on the mat",
        "The dog sat on the rug",
        "Machine learning is a branch of artificial intelligence"
    ]
    vocab = build_vocabulary(mini_docs)
    idf = compute_idf(mini_docs, vocab)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Sample words and IDF scores:")
    for word, score in sorted(zip(vocab, idf), key=lambda x: x[1], reverse=True)[:8]:
        print(f"    {word:20s} IDF={score:.3f}")

    emb1 = tfidf_embed(mini_docs[0], vocab, idf)
    emb2 = tfidf_embed(mini_docs[1], vocab, idf)
    emb3 = tfidf_embed(mini_docs[2], vocab, idf)

    print(f"\n  Embedding dimensions: {len(emb1)}")
    print(f"  Non-zero entries in 'cat sat on mat': {sum(1 for v in emb1 if v > 0)}")
    print(f"  Non-zero entries in 'dog sat on rug': {sum(1 for v in emb2 if v > 0)}")
    print(f"  Non-zero entries in 'machine learning': {sum(1 for v in emb3 if v > 0)}")

    print("\n" + "=" * 60)
    print("STEP 3: Cosine Similarity")
    print("=" * 60)

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)
    sim_23 = cosine_similarity(emb2, emb3)

    print(f"  'cat on mat' vs 'dog on rug':     {sim_12:.4f}  (similar structure)")
    print(f"  'cat on mat' vs 'machine learning': {sim_13:.4f}  (unrelated)")
    print(f"  'dog on rug' vs 'machine learning': {sim_23:.4f}  (unrelated)")
    print(f"\n  As expected: similar sentences score higher.")

    print("\n" + "=" * 60)
    print("STEP 4: Full RAG Pipeline")
    print("=" * 60)

    rag = RAGPipeline(chunk_size=50, overlap=10, top_k=3)
    source_names = [
        "refund-policy.md",
        "product-overview.md",
        "security.md",
        "api-docs.md",
        "uptime-sla.md"
    ]
    num_chunks = rag.index(SAMPLE_DOCUMENTS, source_names)
    print(f"  Indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} chunks")
    print(f"  Vocabulary size: {len(rag.vocab)} terms")

    queries = [
        "What is the refund policy for enterprise customers?",
        "What are the API rate limits?",
        "How is customer data encrypted?",
        "What happens if uptime falls below the SLA?",
        "How much does the Professional plan cost?"
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        result = rag.query(query, top_k=3)
        print(f"  Answer: {result['answer']}")
        print(f"  Retrieved {len(result['retrieved'])} chunks:")
        for r in result["retrieved"]:
            preview = r["chunk"][:80].replace("\n", " ")
            print(f"    [{r['source']}] score={r['score']:.4f} | {preview}...")

    print("\n" + "=" * 60)
    print("STEP 5: Chunk Size Comparison")
    print("=" * 60)

    test_query = "What is the refund policy for enterprise customers?"
    for chunk_size in [20, 50, 100, 200]:
        rag_test = RAGPipeline(chunk_size=chunk_size, overlap=max(5, chunk_size // 5))
        n = rag_test.index(SAMPLE_DOCUMENTS)
        result = rag_test.query(test_query, top_k=3)
        top_score = result["retrieved"][0]["score"] if result["retrieved"] else 0
        print(f"  chunk_size={chunk_size:>3d}: {n:>3d} chunks, "
              f"top_score={top_score:.4f}, "
              f"answer_len={len(result['answer'])}")

    print("\n" + "=" * 60)
    print("STEP 6: Prompt Inspection")
    print("=" * 60)

    result = rag.query("What encryption does Acme use?", top_k=2)
    prompt_lines = result["prompt"].split("\n")
    print(f"  Prompt length: {len(result['prompt'])} chars")
    print(f"  Prompt lines: {len(prompt_lines)}")
    print(f"\n  First 5 lines of generated prompt:")
    for line in prompt_lines[:5]:
        print(f"    {line}")
    print(f"  ...")
    print(f"  Last 3 lines of generated prompt:")
    for line in prompt_lines[-3:]:
        print(f"    {line}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  RAG pipeline: Query -> Embed -> Search -> Augment -> Generate")
    print(f"  Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"  Total chunks: {num_chunks}")
    print(f"  Vocabulary size: {len(rag.vocab)}")
    print(f"  Embedding dimensions: {len(rag.vocab)}")
    print("  Similarity metric: cosine similarity")
    print("  Embedding method: TF-IDF")
    print("\n  In production, replace TF-IDF with neural embeddings")
    print("  (text-embedding-3-small) and the simple generator with")
    print("  an actual LLM API call. The pipeline stays the same.")
