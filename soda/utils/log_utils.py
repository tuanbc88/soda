import os
from datetime import datetime

COL_WIDTH=32

def write_log(log_dir, log_filename, context):
    """
    Ghi log ra file riêng.

    Args:
        log_dir (str): thư mục chứa log (vd: 'logs_sd2a')
        log_filename (str): tên file log (vd: 'idx_0.sd2a_prompt.log')
        context (str | dict | list): nội dung cần ghi
    """

    # tạo thư mục nếu chưa có
    os.makedirs(log_dir, exist_ok=True)

    # full path
    log_path = os.path.join(log_dir, log_filename)

    # nếu context không phải string thì convert cho đẹp
    if not isinstance(context, str):
        import json
        context = json.dumps(context, indent=2, ensure_ascii=False)

    # ghi file
    #'a' → append (giữ log cũ, thêm vào cuối)
    #'w' → overwrite (xóa file cũ, ghi mới hoàn toàn)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'[{datetime.now()}]\n')
        f.write(context + '\n')
        f.write('-' * 50 + '\n')


import os
from datetime import datetime

def format_candidate(rel, score):
    full_rel = rel
    short_rel = shorten_relation(rel)

    if score < -1e6:
        score_str = "-inf"
    else:
        score_str = f"{score:.2f}"

    return f"{short_rel} ({score_str})"




def pad(s, width=COL_WIDTH):
    s = str(s) if s is not None else ""

    if len(s) > width:
        return s[:width - 3] + "..."

    return s.ljust(width)


def shorten_relation(rel, max_len=24):
    if rel is None:
        return ""

    if len(rel) <= max_len:
        return rel

    # 🔥 giữ prefix + suffix (rất quan trọng)
    return rel[:12] + "..." + rel[-8:]    

def build_triplet_table(triplet, case_data_map, cases, top_k=5):
    """
    triplet: (h, r, t)
    case_data_map:
        {
            case_name: {
                "pred": str,
                "top_k": [rel1, rel2, ...],
                "scores": [s1, s2, ...]
            }
        }
    """

    h, r, t = triplet

    lines = []
    lines.append(f"[Triplet] ({h}, {r}, {t})\n")

    # header
    header = "| FIELD".ljust(22)
    for c in cases:
        header += f"| {pad(c)}"
    header += "|"

    sep = "+" + "-"*22
    for _ in cases:
        sep += "+" + "-"*24
    sep += "+"

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # ======================
    # Pred row
    # ======================
    row = "| Pred".ljust(22)

    for c in cases:
        pred = case_data_map[c]["pred"]
        row += f"| {pad(pred)}"

    row += "|"
    lines.append(row)

    # ======================
    # Top-k rows
    # ======================
    for k in range(top_k):
        row = f"| Top-{k+1}".ljust(22)

        for c in cases:
            topk = case_data_map[c]["top_k"]
            scores = case_data_map[c]["scores"]
            pred = case_data_map[c]["pred"]

            rel = topk[k] if k < len(topk) else None
            score = scores[k] if k < len(scores) else None

            text = format_candidate(rel, score)

            # ✅ mark selected
            if rel == pred:
                text += " ✓"

            row += f"| {pad(text)}"

        row += "|"
        lines.append(row)

    lines.append(sep)
    lines.append("")  # blank line

    return "\n".join(lines)




def write_sc_table_log(log_dir, idx, triplet_blocks, text=None, vertical_flag=False):
    """
    triplet_blocks: list[str] (each is a table block)
    """

    os.makedirs(log_dir, exist_ok=True)
    if vertical_flag:
        path = os.path.join(log_dir, f"idx_{idx}.sc.output_vertical.log")
    else:
        path = os.path.join(log_dir, f"idx_{idx}.sc.output.log")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"====================== IDX {idx} ======================\n")
        f.write(f"[TIME] {datetime.now()}\n")

        
        if text:
            f.write(f"[TEXT] {text}\n")

        f.write("\n")

        for block in triplet_blocks:
            if isinstance(block, list):
                block = "\n".join(block)
            f.write(block)
            f.write("\n")

        f.write("======================================================\n")

# TuanBC 2026.04.26 18h00
def format_candidate_vertical(rel, score):
    if score < -1e6:
        score_str = "-inf"
    else:
        score_str = f"{score:.2f}"
    return f"({score_str}) {rel} "

def build_triplet_table_vertical(triplet, case_data_map, cases, top_k=5):
    h, r, t = triplet
    lines = []

    lines.append(f"[Triplet] ({h}, {r}, {t})\n")

    for c in cases:
        data = case_data_map[c]

        pred = data["pred"]
        topk = data["top_k"]
        scores = data["scores"]
        is_new = data["is_new_relation"]
        decision = data.get("decision", "")
        llm_choice = data.get("llm_choice", None)
        prompt_key = data.get("prompt_key", "")
        prompt_path = data.get("prompt_path", "")

        lines.append(f"--- {c} ---")

        # ✅ PROMPT INFO
        lines.append(f"Prompt: {prompt_key} | {prompt_path}")

        # ✅ DECISION
        status = "NEW" if is_new else "RETRIEVED"
        lines.append(f"Predict ({status}): {pred}")
        lines.append(f"Decision: {decision} | LLM choice: {llm_choice}")
        

        lines.append("Retrieval Candidates:")

        for k in range(top_k):
            rel = topk[k] if k < len(topk) else None
            score = scores[k] if k < len(scores) else None

            if score is None:
                score_str = "-"
            else:
                score_str = f"{score:.2f}"

            text = f"({score_str}) {rel}"

            marks = []
            if rel == pred:
                marks.append("✓")

            if marks:
                text += " [" + ",".join(marks) + "]"

            lines.append(f"  Top-{k+1}: {text}")

        lines.append(f"LLM MCP (raw output): {data.get('raw_output')}")
        
        lines.append("###")

    return "\n".join(lines)