"""محاكاة جدول خطوط الأنابيب – 1F1B مقابل Zero Bubble مقابل DualPipe مقابل DualPipeV. أداة تعليمية. حساب pipeline فقاعات لكل جدول زمني معين (P، micro_batches).
النواتج: - جزء الفقاعة لكل جدول زمني ثابت (P، micro_batches) - تحجيم الفقاعات مع نمو الدفعات الصغيرة ليس محاكاة الإنتاج. يتم تطبيع تكاليف القطع الأمامية/الخلفية للوحدة.
يتم تصميم تكاليف الاتصالات على أنها نوافذ متداخلة، وليس نماذج نواة كاملة.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ScheduleStats:
    name: str
    stable_bubble_frac: float
    scales_with_micro_batches: bool
    param_copies: int
    comm_overlap: str


def bubble_1f1b(P: int, M: int) -> float:
    """1F1B: تحتوي مرحلة الإحماء على فتحات أمامية (P-1) دون تداخل خلفي. مرايا التبريد. المرحلة المستقرة لديها صفر فقاعة لكل رتبة لكل دفعة صغيرة، لكن فقاعة الإحماء/التهدئة هي (P-1) قطع أمامية + (P-1) قطع خلفية لكل الرتبة، من إجمالي 2 * M + 2 * (P - 1).
    """
    total = 2 * M + 2 * (P - 1)
    bubble = 2 * (P - 1)
    return bubble / total


def bubble_zero_bubble(P: int, M: int) -> float:
    """تنقسم الفقاعة الصفرية (Qi 2023) إلى الخلف إلى B + W. ويمكن ملء الجزء W فقاعة 1F1B. الفقاعة المتبقية التقريبية هي (P - 1) / قطعتين من الإحماء بالإضافة إلى نفس فترة التهدئة، من 3 * M + 2 * (P - 1) قطع فرعية.
    """
    total = 3 * M + 2 * (P - 1)
    bubble = (P - 1)
    return bubble / total


def bubble_dualpipe(P: int, M: int) -> float:
    """يقوم DualPipe بحقن دفعات صغيرة من طرفي الخط pipe. مستقر فقاعة المرحلة صفر. يحتوي الإحماء/التهدئة على فقاعة ثابتة مستقلة عن M.
    """
    total = 3 * M + (P - 1)
    bubble = (P - 1) // 2
    return bubble / total


def bubble_dualpipev(P: int, M: int) -> float:
    """يستخدم DualPipeV جدولًا على شكل حرف V لنسخة معلمة واحدة. لها الفقاعة أكبر قليلاً من فقاعة DualPipe مع الاستفادة من النصف الذاكرة. تقريبي مثل فقاعة DualPipe بمقدار 1.2x."""
    return bubble_dualpipe(P, M) * 1.2


def summarize(P: int, M: int) -> List[tuple[str, float, int, str]]:
    return [
        ("1F1B",       bubble_1f1b(P, M),        1, "minimal"),
        ("Zero Bubble", bubble_zero_bubble(P, M), 1, "partial"),
        ("DualPipe",   bubble_dualpipe(P, M),    2, "full"),
        ("DualPipeV",  bubble_dualpipev(P, M),   1, "partial"),
    ]


def gpu_hours_recovered(P: int, M: int, total_gpu_hours: float) -> dict:
    b1 = bubble_1f1b(P, M)
    bd = bubble_dualpipe(P, M)
    recovered = (b1 - bd) * total_gpu_hours
    return {
        "1F1B_bubble_frac": b1,
        "DualPipe_bubble_frac": bd,
        "recovered_gpu_hours": recovered,
    }


def main() -> None:
    print("=" * 70)
    print("DUALPIPE PARALLELISM SIMULATOR (Phase 10, Lesson 19)")
    print("=" * 70)
    print()

    print("-" * 70)
    print("Step 1: bubble fraction at P=8, micro_batches=16")
    print("-" * 70)
    print(f"  {'schedule':<14} {'bubble':>10} {'param copies':>14} {'comm overlap':>14}")
    for name, b, pc, co in summarize(P=8, M=16):
        print(f"  {name:<14} {b:>9.1%}  {pc:>14}  {co:>14}")
    print()

    print("-" * 70)
    print("Step 2: bubble fraction scaling vs micro_batches (P=8)")
    print("-" * 70)
    header = "  " + "M".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for M in (4, 8, 16, 32, 64, 128):
        row = f"  {M:>6}"
        for _, b, _, _ in summarize(P=8, M=M):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("Step 3: bubble fraction scaling vs pipeline depth (M=64 fixed)")
    print("-" * 70)
    header = "  " + "P".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for P in (4, 8, 16, 32, 64):
        row = f"  {P:>6}"
        for _, b, _, _ in summarize(P=P, M=64):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("Step 4: recovered GPU-hours (DeepSeek-V3-shape run)")
    print("-" * 70)
    print("  DeepSeek-V3: 2048 H800 GPUs, ~2.8M GPU-hours total.")
    print("  Assume P=16 pipeline depth, M=128 micro-batches per step.")
    r = gpu_hours_recovered(P=16, M=128, total_gpu_hours=2_800_000)
    print(f"  1F1B bubble     : {r['1F1B_bubble_frac']:.1%}")
    print(f"  DualPipe bubble : {r['DualPipe_bubble_frac']:.1%}")
    print(f"  recovered       : {r['recovered_gpu_hours']:,.0f} GPU-hours")
    print(f"  (that is roughly the cost of a full 70B dense pre-training run)")
    print()

    print("takeaway: bubbles do not grow with M for DualPipe. the 2x parameter")
    print("          replication pays for itself at MoE scale because Expert")
    print("          Parallelism already spreads the dominant weights thin.")
    print("          DualPipeV drops the 2x at a small bubble cost.")


if __name__ == "__main__":
    main()
