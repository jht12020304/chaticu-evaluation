# D4 blinded LLM coding pipeline (spec §3.1–3.6). Expects D3 (DataFrame) and blinded (DataFrame) in namespace.
import json, time, re
import pandas as pd

TEMPERATURE = None  # `temperature` is deprecated/rejected (400) for this model; server default used
MAX_TOKENS = 7000
MODEL = host.reasoning_model()
CONC = 6
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "你是一位重症醫學臨床藥師，擔任「盲化文本判讀員」。你將收到：一題ICU藥物治療問題、藥師撰寫的參考標準（核心答案＋說明），"
    "以及一份「來源未知」的回答（僅有匿名編號）。你的任務是依固定編碼規則，嚴格以參考標準為主要依據（必要時輔以可查證之臨床知識），"
    "對該回答進行結構化編碼。不要猜測回答的來源系統，不要因文風、長度或排版而加分或扣分；只評內容。"
    "所有欄位都必須填寫；必須使用提供的工具（record_coding）輸出。"
)

PROMPT_TEMPLATE = """【題目（含病例背景）】
{case_context}
{question_text}

【參考標準：核心答案（A）】
{gold_answer}

【參考標準：說明/補充（臨床理由）】
{rationale}

【參考標準中的關鍵藥物與劑量（輔助清單，由說明文自動摘錄）】
{key_drugs_doses}

【待編碼回答（匿名編號 {blind_id}）】
{answer}

════════ 編碼規則 ════════
(1) core_correctness（核心答案正確性，比對「核心答案 A」）三分類：
  - 完整正確：回答的核心結論與 A 一致，且未附帶與 A 相矛盾的主張。
  - 部分正確：核心結論方向正確但不完整（漏掉 A 的關鍵要素）、或含有與 A 部分不一致的內容、或以含糊/多選方式涵蓋正確答案。
  - 不正確：核心結論與 A 相反、答錯選項、或未回答 A 所問的核心問題。
  附一句話理由（core_justification）。

(2) hallucination（內容幻覺）0/1：回答中是否含至少一則「無法由參考標準或可查證的臨床知識支持」的事實性陳述（例如錯誤的藥理機轉、虛構的數據/指引建議、與已知證據相反的具體宣稱）。
  與 A 不同但臨床上可查證為真的陳述，不算幻覺。列出被標記的陳述（hallucinated_statements，逐句原文摘錄；若為 0 則為空清單）。

(3) semantic_fidelity（語意忠實度）0–100，固定錨點：
  0 = 與核心答案矛盾；25 = 與參考標準大致不同；50 = 部分一致，缺少關鍵要素；75 = 一致，僅有輕微遺漏；100 = 完全一致，含關鍵劑量與監測要點。
  可使用錨點之間的整數值。附一句話理由（fidelity_justification）。

(4) dose_mentions（逐次劑量提及）：逐一列舉回答中每一則「劑量／輸注速率／濃度／給藥頻次」的具體數值陳述（每一則數值陳述為一筆；同一藥物的 bolus 與維持劑量分開列）。
  每筆給出 drug、stated_value（原文數值與單位）、context（用途/情境，簡述），並分類：
  - correct：與參考標準或可查證的標準劑量一致。
  - imprecise：方向正確但範圍過寬/過窄、單位或條件不完整、或與參考標準有輕微出入且無安全疑慮。
  - incorrect：數值明顯錯誤（與參考標準或標準劑量不符），但在該情境下不太可能直接造成傷害。
  - unsafe：數值錯誤且在該病人情境下可能造成傷害（過量、禁忌、忽略器官功能調整等）。
  對每筆 incorrect / unsafe 者，另給：S（嚴重度 0–4：0=無臨床影響, 1=輕微, 2=中度, 3=重大, 4=可能致命/永久傷害）、L（發生可能性 1–3：1=不太可能被照做, 2=可能, 3=很可能）、severity_label（Mild/Moderate/Severe）。correct / imprecise 者 S=0、L=1、severity_label="None"。
  回答完全沒有劑量陳述時，dose_mentions 為空清單。

(5) omitted_but_needed：依參考標準（A 與說明）判斷「此題應該給出、但回答未提及」的關鍵劑量／速率／目標值，每項給 drug、expected_value（參考標準的數值）、reason。若參考標準本身沒有要求任何劑量，或回答已全部涵蓋，則為空清單。

(6) 七項臨床構面（per-answer 格位），每格四分類：
  - not_applicable：此題／此回答不涉及該構面，或參考標準未要求。
  - adequate：回答在該構面的內容正確且足夠。
  - incorrect：回答在該構面有錯誤但不太可能直接造成傷害。
  - unsafe：回答在該構面有錯誤且可能造成病人傷害。
  構面：drug_choice（藥物選擇）、deescalation_stop_rule（降階／停藥規則）、interaction_contraindication（交互作用／禁忌）、route_formulation（途徑／劑型）、organ_function_adjustment（器官功能調整）、titration_target（滴定目標）、monitoring_parameter（監測參數）。
  對 incorrect / unsafe 者給 severity（Mild/Moderate/Severe）與一句話 reason；其他給 severity="None"，reason 可簡短。

請務必用 record_coding 工具輸出。"""

DIMS = ["drug_choice", "deescalation_stop_rule", "interaction_contraindication",
        "route_formulation", "organ_function_adjustment", "titration_target", "monitoring_parameter"]

SEV = {"type": "string", "enum": ["None", "Mild", "Moderate", "Severe"]}
dim_schema = {
    "type": "object",
    "properties": {
        "rating": {"type": "string", "enum": ["not_applicable", "adequate", "incorrect", "unsafe"]},
        "severity": SEV,
        "reason": {"type": "string"},
    },
    "required": ["rating", "severity", "reason"],
}
TOOL = {
    "name": "record_coding",
    "description": "記錄一份匿名回答的結構化編碼結果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "core_correctness": {"type": "string", "enum": ["完整正確", "部分正確", "不正確"]},
            "core_justification": {"type": "string"},
            "hallucination": {"type": "integer", "enum": [0, 1]},
            "hallucinated_statements": {"type": "array", "items": {"type": "string"}},
            "semantic_fidelity": {"type": "integer", "minimum": 0, "maximum": 100},
            "fidelity_justification": {"type": "string"},
            "dose_mentions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drug": {"type": "string"},
                        "stated_value": {"type": "string"},
                        "context": {"type": "string"},
                        "class": {"type": "string", "enum": ["correct", "imprecise", "incorrect", "unsafe"]},
                        "reason": {"type": "string"},
                        "S": {"type": "integer", "minimum": 0, "maximum": 4},
                        "L": {"type": "integer", "minimum": 1, "maximum": 3},
                        "severity_label": SEV,
                    },
                    "required": ["drug", "stated_value", "context", "class", "reason", "S", "L", "severity_label"],
                },
            },
            "omitted_but_needed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"drug": {"type": "string"}, "expected_value": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["drug", "expected_value", "reason"],
                },
            },
            "dimensions": {
                "type": "object",
                "properties": {d: dim_schema for d in DIMS},
                "required": DIMS,
            },
        },
        "required": ["core_correctness", "core_justification", "hallucination", "hallucinated_statements",
                     "semantic_fidelity", "fidelity_justification", "dose_mentions", "omitted_but_needed", "dimensions"],
    },
}


def build_prompt(row):
    g = D3.loc[row.question_idx - 1]
    return PROMPT_TEMPLATE.format(
        case_context=g.case_context or "（無額外病例背景）",
        question_text=g.question_text,
        gold_answer=g.gold_answer_expanded,
        rationale=g.rationale,
        key_drugs_doses=g.key_drugs_doses or "無",
        blind_id=row.blind_id,
        answer=row.blind_text,
    )


def validate(inp):
    """Return list of problems (empty = valid)."""
    p = []
    if not isinstance(inp, dict):
        return ["not a dict"]
    for k in TOOL["input_schema"]["required"]:
        if k not in inp:
            p.append(f"missing {k}")
    if p:
        return p
    if inp["core_correctness"] not in ["完整正確", "部分正確", "不正確"]:
        p.append("bad core_correctness")
    if inp["hallucination"] not in (0, 1):
        p.append("bad hallucination")
    sf = inp["semantic_fidelity"]
    if not (isinstance(sf, (int, float)) and 0 <= sf <= 100):
        p.append("bad semantic_fidelity")
    if not isinstance(inp["dose_mentions"], list):
        p.append("dose_mentions not list")
    else:
        for i, m in enumerate(inp["dose_mentions"]):
            if not isinstance(m, dict) or m.get("class") not in ["correct", "imprecise", "incorrect", "unsafe"]:
                p.append(f"mention {i} bad class"); continue
            if not (isinstance(m.get("S"), int) and 0 <= m["S"] <= 4): p.append(f"mention {i} bad S")
            if not (isinstance(m.get("L"), int) and 1 <= m["L"] <= 3): p.append(f"mention {i} bad L")
            if m.get("severity_label") not in ["None", "Mild", "Moderate", "Severe"]: p.append(f"mention {i} bad label")
            if m["class"] in ("incorrect", "unsafe") and (m.get("S", 0) == 0 or m.get("severity_label") == "None"):
                p.append(f"mention {i} incorrect/unsafe without S/label")
    if not isinstance(inp["omitted_but_needed"], list):
        p.append("omitted not list")
    dims = inp["dimensions"]
    if not isinstance(dims, dict):
        p.append("dimensions not dict")
    else:
        for d in DIMS:
            c = dims.get(d)
            if not isinstance(c, dict) or c.get("rating") not in ["not_applicable", "adequate", "incorrect", "unsafe"]:
                p.append(f"dim {d} bad rating"); continue
            if c.get("severity") not in ["None", "Mild", "Moderate", "Severe"]:
                p.append(f"dim {d} bad severity")
            if c["rating"] in ("incorrect", "unsafe") and c.get("severity") == "None":
                p.append(f"dim {d} incorrect/unsafe without severity")
    return p


def make_request(row, feedback=None):
    prompt = build_prompt(row)
    if feedback:
        prompt += "\n\n【上一輪輸出未通過格式檢核，問題如下，請修正後重新完整輸出】\n- " + "\n- ".join(feedback)
    return {"prompt": prompt, "system": SYSTEM_PROMPT, "model": MODEL, "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},
            "tools": [TOOL], "tool_choice": {"type": "tool", "name": "record_coding"}}


def run_coding(blinded_df, raw_path="D4_raw_responses.jsonl"):
    pending = {r.blind_id: (r, None) for r in blinded_df.itertuples()}
    results, attempts_log, model_ids = {}, {}, set()
    with open(raw_path, "w", encoding="utf-8") as raw:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if not pending:
                break
            ids = list(pending)
            reqs = [make_request(pending[b][0], pending[b][1]) for b in ids]
            t0 = time.time()
            res = host.llm(reqs, max_concurrency=CONC)
            print(f"attempt {attempt}: {len(ids)} requests, {time.time()-t0:.0f}s")
            nxt = {}
            for b, req, r in zip(ids, reqs, res):
                attempts_log[b] = attempt
                rec = {"blind_id": b, "attempt": attempt, "model_requested": MODEL,
                       "temperature": TEMPERATURE, "response": r}
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if "error" in r:
                    nxt[b] = (pending[b][0], [f"API error: {str(r['error'])[:200]}"]); continue
                model_ids.add(r.get("model"))
                if r.get("stop_reason") == "max_tokens":
                    nxt[b] = (pending[b][0], ["output truncated (max_tokens); please be more concise in reasons"]); continue
                tu = r.get("tool_use")
                inp = tu.get("input") if tu else None
                probs = validate(inp)
                if probs:
                    nxt[b] = (pending[b][0], probs)
                else:
                    results[b] = {"coding": inp, "model": r.get("model"), "usage": r.get("usage"), "attempt": attempt}
            pending = nxt
    return results, pending, attempts_log, model_ids
