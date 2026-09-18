# TF-IDF cosine similarity of each blinded answer to its question's gold text (spec §3.7).
# Expects D3 and blinded DataFrames in the namespace.
import re, jieba, sklearn, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jieba.setLogLevel(60)
JIEBA_VERSION = jieba.__version__
SKLEARN_VERSION = sklearn.__version__
CUT_MODE = "jieba.lcut(text, cut_all=False, HMM=True)  # precise mode, default dictionary, no user dictionary"

# Stopword list: Traditional-Chinese function words/particles + generic clinical-template tokens + English stopwords.
ZH_STOP = """的 了 在 是 我 有 和 就 不 人 都 一 一個 上 也 很 到 說 要 去 你 會 著 沒有 看 好 自己 這 那 與 及 或 並 且 而 但 如 若 因 為 因為 所以 因此 此 其 之 以 於 對 從 由 被 把 將 使 讓 可 可以 能 應 應該 需 需要 須 必須 可能 較 更 最 非常 相當 亦 也 又 再 仍 仍然 已 已經 曾 尚 未 才 只 僅 均 皆 各 每 該 這些 那些 此外 另外 另 同時 以及 或者 還是 還 甚至 例如 如下 如 下列 所 所有 任何 某 某些 者 時 中 內 外 前 後 間 等 等等 然而 但是 不過 雖然 儘管 即使 如果 假如 一旦 除了 除 至 至於 對於 關於 根據 依 依據 按 按照 透過 藉由 經由 通過 由於 基於 作為 成為 屬於 具有 含有 包括 包含 進行 給予 使用 採用 考慮 建議 應注意 注意 說明 補充 回答 問題 題目 病人 患者 病患 目前 此時 情況 情形 狀況 一般 通常 主要 常見 重要 相關 明確 具體 明顯 顯著 適當 合適 合理 必要 需求 方面 部分 整體 整個 全部 之間 之後 之前 以上 以下 以內 以外 左右 大約 約 大概 mg mcg μg µg ml mL kg hr min h g L""".split()
EN_STOP = "the a an of and or to in on for with is are be as by at from that this it its into than then which who whom whose when where while may might can could should would will shall not no yes vs versus per".split()
STOPWORDS = sorted(set(ZH_STOP + EN_STOP))
PUNCT_RE = re.compile(r"^[\W_]+$")


def tokenize(text):
    toks = jieba.lcut(text, cut_all=False, HMM=True)
    out = []
    for t in toks:
        t = t.strip().lower()
        if not t or PUNCT_RE.match(t) or t in STOPWORDS:
            continue
        if re.fullmatch(r"[\d\.\-–~/%:]+", t):  # bare numbers/ranges — kept out; dose numbers are judged by the LLM coder, not TF-IDF
            continue
        out.append(t)
    return out


def run_tfidf(blinded_df, D3_df):
    gold_texts = (D3_df.gold_answer_expanded + "\n" + D3_df.rationale).tolist()
    answer_texts = blinded_df.blind_text.tolist()
    corpus = answer_texts + gold_texts
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True, norm="l2", min_df=1)
    X = vec.fit_transform(corpus)
    A = X[: len(answer_texts)]
    G = X[len(answer_texts):]
    q = blinded_df.question_idx.values - 1
    sims = np.array([cosine_similarity(A[i], G[q[i]])[0, 0] for i in range(len(answer_texts))])
    return sims, vec, X
