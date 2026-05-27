---
name: skill-image-text-retriever
description: Build an image embedding index with any CLIP checkpoint; support query-by-text and query-by-image
version: 1.0.0
phase: 4
lesson: 18
tags: [clip, retrieval, faiss, zero-shot]
---

#مسترجع الصورة والنص
قم بتحويل مجلد الصور إلى فهرس قابل للبحث باستخدام عمليات التضمين CLIP.
##متى يستخدم
- بناء بحث بالصور بدون لقطة على كتالوج داخلي.
- إلغاء تكرار الصور شبه المتطابقة عن طريق تضمين المسافة.
- إنشاء مكون سريع "للبحث عن مماثل" بدون مجموعة بيانات مصنفة.
## المدخلات
- `image_folder`: دليل ملفات الصور.
- `clip_model`: معرف HuggingFace مثل `openai/clip-vit-base-patch32` أو `google/siglip-base-patch16-224`.
- `index_type`: مسطح | IVF | __المصطلح_1__.
- `embedding_dim`: يتم استنتاجه من النموذج.
## الخطوات
1. قم بتحميل النموذج CLIP والمعالج المسبق.
2. قم بتشفير كل صورة في المجلد دفعة واحدة. حفظ التضمينات كقائمة (N,D) float32 + أسماء الملفات.
3. أنشئ فهرس FAISS فوق التضمينات. استخدم المنتج الداخلي على المتجهات المقيسة L2 لتشابه جيب التمام.
4. كشف واجهتين للاستعلام:
   - `search_by_text(text, k)` — قم بتضمين النص والبحث.
   - `search_by_image(image_path, k)` — تضمين الصورة والبحث.
## قالب الإخراج
```python
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import faiss


class ImageTextRetriever:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        self.index = None
        self.filenames = []

    @torch.no_grad()
    def _encode_images(self, paths, batch=16):
        embs = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            embs.append(out.cpu().numpy())
        return np.concatenate(embs).astype(np.float32)

    @torch.no_grad()
    def _encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        out = self.model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu().numpy().astype(np.float32)

    def build_index(self, folder, index_type="flat"):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        self.filenames = sorted(files)
        embs = self._encode_images(self.filenames)
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(256, max(4, len(embs) // 32))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embs)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search_by_text(self, text, k=5):
        q = self._encode_text([text])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def search_by_image(self, image_path, k=5):
        q = self._encode_images([image_path])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]
```

## تقرير
```
[retriever]
  model:          <name>
  num_images:     <int>
  dim:            <int>
  index_type:     flat | IVF | HNSW
  index_size_mb:  <float>
```

## قواعد
- دائمًا L2-تسوية التضمينات قبل الفهرسة؛ المنتج الداخلي لـ FAISS على المتجهات المقيسة يساوي تشابه جيب التمام.
- بالنسبة إلى أقل من 100 ألف صورة، فإن `IndexFlatIP` (بالضبط) هو الأبسط والأسرع.
- بالنسبة إلى 100 ألف - 10 مليون، `IndexIVFFlat` هو المقايضة القياسية.
- بالنسبة إلى أكثر من 10 ملايين، استخدم HNSW أو متغير كمي للمنتج.
- لا تقم أبدًا بإعادة بناء الفهرس عند كل استعلام؛ تضمين مرة واحدة، والبحث عدة مرات.