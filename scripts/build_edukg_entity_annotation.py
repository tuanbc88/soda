"""Build the human-annotation package for the Edu-KG entity-type inversion (review L2, Q2).

Why this exists
---------------
On the three English benchmarks the entity signal ordering by discovered inventory size is
phi_e=1 > phi_e=2 > phi_e=3. On Edu-KG it INVERTS: phi_e=3 (definition+parent) yields the LARGEST
inventory. The paper reports that and deliberately does not interpret it, because Edu-KG has no gold
graph -- and Sec. 5.3.3 already proves an inventory count cannot name a merge direction. The reviewer
asks whether the inversion in SIZE is also an inversion in ACCURACY, and whether a small annotated
subset can settle it inside the revision window.

This script produces that subset. It samples entity mentions, attaches their chunk context, and
writes a BLIND annotation sheet: the annotators never see what any signal predicted, so their labels
cannot anchor on the model. The predictions are written to a separate key file used only for scoring.

Sampling is stratified into two strata, and both are reported separately at scoring time:
  * disagree -- the three signals assigned different canonical types to the same mention. These are
    where the inversion lives, so they carry the information, but they are not representative.
  * random   -- a uniform sample over all mentions, so the subset also supports an unbiased read.

Scoring (later, once labels come back): entity B-cubed of each signal's grouping against the
adjudicated human labels, per stratum, plus Cohen's kappa between the two annotators.

Usage
-----
    python scripts/build_edukg_entity_annotation.py --n_disagree 120 --n_random 80
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

RUN_DIR = ("output/edu_kg_core_selfcanon2_mode1_item_qwen2.5-7b_bgem3vni_A100_edukg_release/iter0")
SIGNALS = {
    "ec1": "entity_canon_ec1_name_only.json",
    "ec2": "entity_canon_ec2_name_definition.json",
    "ec3": "entity_canon_ec3_definition_parent.json",
}
OUT_DIR = "assets/review/peer_review_by_models_l2/edukg_entity_annotation"
CONTEXT_CHARS = 400


def load_signal(run_dir, fname):
    """(record_idx, mention) -> canonical type assigned by this signal, plus the chunk texts."""
    with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
        records = json.load(f)
    assign = {}
    texts = []
    for idx, rec in enumerate(records):
        texts.append(rec.get("text", ""))
        for group in rec.get("entities", []):
            ptype = group.get("pred_entity_type")
            for m in group.get("members", []):
                assign[(idx, m)] = ptype
    return assign, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--n_disagree", type=int, default=120)
    ap.add_argument("--n_random", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_suggested_types", type=int, default=30)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    assigns, texts = {}, None
    for sig, fname in SIGNALS.items():
        assigns[sig], t = load_signal(args.run_dir, fname)
        texts = texts or t
        print(f"{sig}: {len(assigns[sig])} mention assignments")

    # A mention is comparable only where all three signals placed it.
    common = set(assigns["ec1"]) & set(assigns["ec2"]) & set(assigns["ec3"])
    print(f"mentions assigned by all three signals: {len(common)}")

    disagree, agree = [], []
    for key in common:
        types = tuple(assigns[s][key] for s in ("ec1", "ec2", "ec3"))
        (disagree if len(set(types)) > 1 else agree).append(key)
    print(f"  disagree: {len(disagree)}   agree: {len(agree)}")

    sample = [(k, "disagree") for k in rng.sample(disagree, min(args.n_disagree, len(disagree)))]
    sample += [(k, "random") for k in rng.sample(sorted(common), min(args.n_random, len(common)))]
    rng.shuffle(sample)   # so the annotator cannot infer the stratum from the row order

    # Suggested vocabulary: the most frequent canonical types across all three signals. Offered as
    # a starting point only -- annotators may write their own label, and are told so.
    freq = Counter()
    for s in SIGNALS:
        freq.update(v for v in assigns[s].values() if v)
    suggested = [t for t, _ in freq.most_common(args.n_suggested_types)]

    sheet = os.path.join(args.out_dir, "annotation_sheet.csv")
    key_path = os.path.join(args.out_dir, "key_hidden.csv")
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fs, \
         open(key_path, "w", newline="", encoding="utf-8-sig") as fk:
        ws = csv.writer(fs)
        wk = csv.writer(fk)
        ws.writerow(["id", "mention", "context", "gold_type", "confidence_1_3", "note"])
        wk.writerow(["id", "stratum", "record_idx", "mention", "ec1_type", "ec2_type", "ec3_type"])
        for i, ((idx, mention), stratum) in enumerate(sample, 1):
            ctx = (texts[idx] or "").strip().replace("\n", " ")[:CONTEXT_CHARS]
            ws.writerow([i, mention.replace("_", " "), ctx, "", "", ""])
            wk.writerow([i, stratum, idx, mention,
                         assigns["ec1"][(idx, mention)],
                         assigns["ec2"][(idx, mention)],
                         assigns["ec3"][(idx, mention)]])

    guide = os.path.join(args.out_dir, "HUONG_DAN_GAN_NHAN.md")
    with open(guide, "w", encoding="utf-8") as f:
        f.write(GUIDE_TMPL.format(
            n=len(sample),
            n_disagree=sum(1 for _, s in sample if s == "disagree"),
            n_random=sum(1 for _, s in sample if s == "random"),
            suggested="\n".join(f"- `{t}`" for t in suggested),
        ))

    print(f"\nannotation sheet -> {sheet}  ({len(sample)} rows)")
    print(f"hidden key       -> {key_path}   (DO NOT give this to the annotators)")
    print(f"guide            -> {guide}")


GUIDE_TMPL = """# Hướng dẫn gán nhãn kiểu thực thể — Edu-KG ({n} mention)

**Mục đích.** Trên ba benchmark tiếng Anh, tín hiệu entity càng nhiều thông tin thì inventory càng
NHỎ. Trên Edu-KG thì ngược lại. Bài báo báo cáo hiện tượng này nhưng **không diễn giải**, vì Edu-KG
không có gold graph. Bộ nhãn này để trả lời: đảo về **kích thước** có kèm đảo về **độ chính xác** không.

**Anh/chị gán nhãn "mù"**: bảng KHÔNG chứa dự đoán của mô hình. Đó là chủ ý — thấy dự đoán trước sẽ
kéo nhãn theo mô hình và số đo mất giá trị.

## Việc cần làm

Với mỗi dòng trong `annotation_sheet.csv`:

1. Đọc `mention` (chuỗi thực thể) và `context` (câu/đoạn chứa nó).
2. Điền cột **`gold_type`**: *trong ngữ cảnh này, thực thể đó thuộc KIỂU gì?*
   - Ưu tiên chọn từ danh sách gợi ý bên dưới nếu có kiểu phù hợp.
   - **Không có kiểu phù hợp thì cứ tự viết** một nhãn ngắn (tiếng Việt hoặc tiếng Anh, nhất quán
     trong toàn bộ file). Danh sách gợi ý chỉ là gợi ý, không phải danh sách đóng.
   - Quy tắc: gán kiểu **cụ thể nhất mà anh/chị chắc chắn**. Không chắc thì lùi về kiểu tổng quát hơn.
3. Điền **`confidence_1_3`**: 3 = chắc chắn · 2 = khá chắc · 1 = đoán.
4. `note`: tuỳ ý (ví dụ: mention bị cắt sai, ngữ cảnh không đủ để quyết).

## Quan trọng

- **Hai người gán nhãn ĐỘC LẬP trên cùng file**, không trao đổi trong lúc làm. Mức đồng thuận giữa
  hai người (Cohen's κ) sẽ được báo cáo trong bài, nên trao đổi sẽ làm hỏng con số đó. Bất đồng sẽ
  được hoà giải ở vòng sau.
- Nhãn cần **nhất quán**: cùng một loại thực thể thì dùng cùng một chuỗi nhãn (đừng lúc `Person`
  lúc `Nguoi`). Nếu đổi ý giữa chừng, ghi vào `note` rồi sửa lại các dòng trước.
- Mention hiển thị đã thay `_` bằng khoảng trắng cho dễ đọc.
- Mẫu gồm {n_disagree} dòng thuộc nhóm "ba tín hiệu bất đồng" và {n_random} dòng lấy ngẫu nhiên;
  bảng đã trộn nên **không đoán được dòng nào thuộc nhóm nào** — cứ gán nhãn như nhau.

## Danh sách kiểu gợi ý (không đóng)

{suggested}

## Nộp lại

Lưu thành `annotation_sheet__<tên>.csv` (giữ nguyên cột `id`) và gửi lại. Không cần điền cột nào khác.
"""


if __name__ == "__main__":
    main()
