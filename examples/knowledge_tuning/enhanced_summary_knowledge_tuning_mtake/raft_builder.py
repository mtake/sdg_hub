from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import re
from collections import defaultdict
import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# @@@ahoaho XXX tokenize for Japanese
# See https://challenge-pg.com/2024/11/09/python-janome/
# See https://qiita.com/kiyuka/items/3de09e313a75248ca029#appendix2analyzer%E3%81%A7%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90%E3%82%82%E3%81%99%E3%82%8B
# from janome.tokenizer import Tokenizer
# tk = Tokenizer()  # for Japanese
# def analyzer(x):  # for Japanese
#     return list(tk.tokenize(x, wakati=True))
#     # return [token.surface for token in tk.tokenize(x)]
analyzer = "word"  # for English

# splitter = analyzer  # for Japanese
splitter = lambda x : x.split()  # for English

# ========================
# Chunking utilities
# ========================
def chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    """
    Splits long text into chunks with overlap.
    Token proxy = words (fast approximation).
    """
    if not text:
        return []
    # @@@ahoaho XXX tokenize for Japanese
    # words = text.split()
    words = splitter(text)
    chunks, step = [], max(1, max_tokens - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_tokens >= len(words):
            break
    return chunks


# @@@ahoaho XXX sentence split for Japanese
# _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SENT_SPLIT = re.compile(r"(?<=[。．！？.!?])\s+")


def take_best_sentence(context: str, query: str) -> str:
    """Extract a plausible supportive sentence from context for quoting."""
    if not context:
        return ""
    sents = _SENT_SPLIT.split(context.strip())
    if not sents:
        return context[:400]
    # @@@ahoaho XXX tokenize for Japanese
    # vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(sents + [query])
    sims = cosine_similarity(X[:-1], X[-1])
    idx = int(np.argmax(sims))
    return sents[idx][:400]


def default_answer_builder(
    example: Dict[str, Any], oracle_chunk: str
) -> Tuple[str, str]:
    """
    Build (support_quote, final_answer).
    - Use `messages` if available.
    - Else, use an extractive fallback from the oracle chunk.
    """
    q = example.get("question", "").strip()
    final_answer = ""

    msgs = example.get("messages")
    if isinstance(msgs, list):
        for m in msgs[::-1]:
            if isinstance(m, dict) and m.get("role") in {"assistant", "system"}:
                content = m.get("content") or ""
                if content.strip():
                    final_answer = content.strip()
                    break

    support_quote = take_best_sentence(oracle_chunk, q)

    if not final_answer:
        print("No final answer found")
        final_answer = support_quote if support_quote else "Answer not available."

    return support_quote, final_answer


# ========================
# Model utilities
# ========================
from transformers import AutoConfig, PretrainedConfig, AutoTokenizer

def is_known_model(
    model_path_or_config: str | PretrainedConfig, known_model_type: str | list[str]
) -> bool:
    """
    Determine if the model is a known model.
    """
    if not isinstance(model_path_or_config, (PretrainedConfig, str)):
        raise ValueError(
            f"cannot detect model: received invalid argument of type {type(model_path_or_config)}"
        )

    # convert to config
    model_config = model_path_or_config
    if isinstance(model_path_or_config, str):
        model_config = AutoConfig.from_pretrained(model_path_or_config)

    known_model_types = (
        [known_model_type] if isinstance(known_model_type, str) else known_model_type
    )
    return getattr(model_config, "model_type", None) in known_model_types


def load_tokenizer(student_model):
    """Initialize and return tokenizer."""
    print(f"Loading tokenizer: {student_model}")
    return AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)


# ========================
# Template utilities
# ========================
from datetime import datetime

def strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


import json
from markupsafe import Markup

def tojson_allow_non_ascii(obj, **kwargs):
    return Markup(json.dumps(obj, ensure_ascii=False, **kwargs))


import html
import jinja2

def render_system_message_granite(documents: list[dict], chat_template: jinja2.environment.Template):

    render_dict: dict[str, any] = {}

    # XXX TODO convert format
    # @@@ JFEの実例
    # documents = [
    #     {
    #         'doc_id': '# 鋼管事業の新エネルギーへの取り組み\n\nInitiatives for Energy Transition in the Pipe and Tubular Sector\n石川\u3000信行\u3000ISHIKAWA Nobuyuki 平田\u3000知正\u3000HIRATA Norimasa 要旨JFE スチール\u3000厚板セクター部\u3000主任部員（部長）・博士（工学）JFE スチール\u3000鋼管センター\u3000鋼管企画部\u3000主任部員（課長）\n\n## 要旨\n\nカーボンニュートラルの実現のため，水素へのエネルギー転換とCO 2分離回収貯留（CCS）の技術開発が世界各地で進められている。本稿では，CCSを含む水素サプライチェーンで必要とされる鋼管とJFEスチールでの開発状況を概説する。\n\n## Abstract\n\nTechnologies for energy transition to hydrogen and CO 2 capture and storage (CCS) are being developed around the world for the realization of carbon neutrality. This paper briefly introduces pipe and tubular products used in the hydrogen supply chain including CCS and development status of these products in JFE Steel.\n\n## 1. はじめに\n\nカーボンニュートラル社会の実現には，従来の化石燃料から水素，アンモニアなどへのエネルギー転換が不可欠であり，これら新エネルギーの大規模サプライチェーン構築に向けた技術開発や実証事業が世界各地で進められている。日本では2040年で1 200万トン，2050年で2 000万トンの水素導入（アンモニア含む） 1）を目指して，海外で製造した水素の海上輸送や水素混焼発電など，多くの実証事業が進行中である。水素には，再生可能エネルギーによる電力などで製造したグリーン水素のほかに，石油天然ガスの改質で製造し，製造過程で発生するCO 2を分離回収・貯留（CCS：Carbon dioxide Capture and Storage）するブルー水素があり，CCSを活用したブルー水素も水素導入量拡大のために極めて重要と考えられている。さらに，CCSまたはCCUS（Utilization）は CO 2 排出削減に直接寄与する重要な技術であり，国内数か所でCCS事業化の取り組みも進められている 2）。JFE スチールは長年，石油天然ガス分野で必要とされる多くの鋼管商品を開発・製造し，世界のエネルギー安定供給に貢献してきた。エネルギー転換による脱炭素化を進めるためには，水素やCO 2の輸送や貯蔵または貯留（圧入）のための新たなインフラが必要となるが，使用される材料には耐水素脆性や耐炭酸ガス腐食性など，従来にない性能が必要となっている。図1にCCUSを含む水素サプライチェーンと必要とされる鋼管材料の模式図を示す。本稿では，水素に代表される新エネルギー分野におけるJFEスチール鋼管事業の取り組み，特に鋼管商品の開発状況に関して概説する。\n\n## 2． 水素輸送・貯蔵に必要な鋼管商品\n\n### 2.1 水素輸送用ラインパイプ\n\n海外からの水素の大量輸送には液体水素での船舶輸送が検討されているが，陸上での輸送には高圧水素ガスでのパイプライン輸送が適している。国内でも小規模な水素のパイプライン輸送は行われているが，水素圧力が1 MPa未満であり，水素による材質劣化（水素脆化）が顕著に表れないため，従来の材料が使用できている。しかし，水素を大量に輸送するためには1 MPa以上の高圧で輸送する必要があり，圧力に応じて鋼中への水素侵入量が増えるため，特に材料の破壊特性が顕著に低下することが懸念されている 3）。\u3000水素パイプラインの国際規格としてはASME B31.12 “Hydrogen Pipeline and Piping”があり，API（American Petroleum Institute，アメリカ石油協会）規格のラインパイプが使用可能であるが，高圧輸送の場合は水素中での材料の破壊靭性試験と疲労き裂伝播試験による破壊安全性の検証が求められている。図2にASME B31.12に規定されている破壊安全性評価の概念図を示す。パイプ内面に微小な欠陥が存在する場合，内圧変動によって疲労き裂進展を生じ，あるところで破壊駆動力が材料の破壊靭性値を超え不安定破壊に至る。JFEスチールは水素輸送用ラインパイプの材料検討にいち早く取り組み，最適な材料設計指針を見出している。図3にAPI規格X65のUOE鋼管（外径914.4 mm，鋼管厚さ（以下，管厚）28.6 mm）母材部の高圧水素中疲労き裂進展試験結果を示す 3）。大気中に比べ水素中では疲労き裂進展速度が大幅に増加するが，従来のラインパイプ材の報告値やASME B31.12に規定されている評価曲線に比べ進展速度が遅い。これは，適切な成分設計と鋼板製造時の圧延冷却制御によって微細なベイナイト均一組織となっているためである 4）。本UOE鋼管は，水素中破壊靭性試験においても従来材よりも高い破壊靭性を有する。図4に同じX65鋼管の21 MPa水素中での破壊靭性試験結果を示す。母材（BM）に比べ，鋼管シーム溶接部の溶接熱影響部（HAZ：Heat Affected Zone）や溶接金属（WM：Weld Metal）は低い値を示しているが，ASME B31.12に規定される最小破壊靭性値（55 MPa \u2002\u2005m ）に比べ十分に高い値である。このHAZ部の水素中破壊靭性値を用いて破壊安全性の解析を行った結果，長さ25 mm，深さ3 mmの大きさの欠陥があっても十分な安全余裕度を有することが示された 5）。以上のように，適切な材質設計で製造された鋼管を使用することで水素パイプラインの破壊安全性を確保できる。パイプライン用の鋼管として，電縫鋼管も広く使用されている。JFEスチールは電縫溶接部の信頼性を高めた「マイティーシーム ®」を開発し，極低温や海底など厳しい環境のパイプラインに適用されている 6）。マイティーシームは素材の化学成分や圧延条件の最適化，電縫溶接条件やシーム熱処理条件の最適化に加え，フェーズドアレイ超音波探傷技術により電縫溶接部に形成する微小酸化物を全長にわたって監視することで，電縫溶接部の安定した性能を保証するものである。図5に電縫溶接部のシャルピー衝撃特性を示す。マイティ─シームは電縫溶接部の酸化物が低減されたことにより母材部と同等以上の高い吸収エネルギーが得られている。マイティ─シームは高圧水素輸送用ラインパイプとしてもその性能が期待されており，「海洋石油・天然ガスに係る日本財団とDeepStarの連携技術開発助成プログラム」において，石油メジャーのExxonMobil社 （ 米 ）， TOTAL Energies 社（仏）と共同で高圧水素中での材料適合性評価を進めている 7）。\n\n### 2.2 水素ステーション用蓄圧器\n\n国内での燃料電池車の普及と水素ステーションの整備は着実に進んでおり，東京オリンピックを契機に燃料電池バスの普及が大幅に拡大し，さらに燃料電池トラックなどの開発も進められている。燃料電池車の普及拡大のためには水素ステーションの建設コスト低減が重要課題であり，低コストの水素ステーション用蓄圧器の開発が進められている 8）。JFE スチールは耐水素脆化特性に優れたシームレス鋼管を素材に炭素繊維強化樹脂（CFRP：Carbon Fiber Reinforced Plastics）を胴部に巻付けたType2蓄圧器（写真1）をJFEコンテイナーと共同で開発した。Type2蓄圧器は，高圧力範囲での長寿命化，ストレート構造とすることによる製造コスト低減，さらにはメンテナンスの簡略化が可能という特長がある 9, 10）。JFE スチールとJFEコンテイナーは水素ステーションのさらなる低コスト化に対応できる，大容量のType1蓄圧器（CFRPを使用しない鋼製容器）も商品化しており，Type2 蓄圧器と併せて今後の水素ステーション普及への寄与が期待される。\n\n## 3． CO 2 輸送・貯留に必要な鋼管商品\n\n### 3.1 CO 2 輸送用ラインパイプ\n\nCO 2（二酸化炭素）は常温常圧では気体で，－79℃で固体（ドライアイス）となるが，図6の状態図のように高圧では液化する特徴を持っている。CO 2のパイプライン輸送は常温以上で圧力10 MPa弱の液相または超臨界相（Supercritical fluid）で運ばれることが多い。CO 2 パイプラインの規格としては，ISO 27913“Carbon dioxide capture, transportation and geological storage - Pipeline transportation systems”やDNV-RP-F104“Design and operation of carbon dioxide pipelines”があり CO 2 流の不純物成分や使用材料が規定されている。いずれの規格でもCO 2流に液体の水が存在するといわゆる炭酸ガス腐食 11）を起こすため，不純物中の水分は厳しく制限されている。その上でラインパイプ材料には通常のガスパイプラインと同様の炭素鋼が使用される。高圧ガスパイプラインの安全性の課題として高速延性破壊があげられる。これは何らかの事故でラインパイプにき裂が発生した場合，高いガス圧力を駆動力として長距離にわたりき裂が高速で伝播する現象のことである。通常のガスパイプラインでは適切な材料が使用されていればガス圧の低下とともにき裂は停止するが，高圧のCO 2（液相又は超臨界相）の場合は圧力が低下すると，図6の状態図で分かるように，気相が発生し圧力が低下しにくくなりき裂が停止しない現象が起きる。CO 2パイプラインでの高速延性破壊を防止するために必要な材料性能を明らかにするため，これまでに多くの実管バースト実験が行われている 12）。図7はそれらの実験結果をDNV-RP-F104に規定されている設計式に従ってプロットしたものである。ここで，Pはき裂先端の圧力（MPa）， Dはパイプ外径（mm）， tは管厚（mm）， σ fは流動応力（降伏応力と引張強さの平均値MPa）， R CVNは断面積当たりのシャルピーエネルギー（J/mm 2）， Eはヤング率（MPa）， Rはパイプ半径（mm）である。グラフの右下の領域ではき裂停止（Arrest）しており，この領域に入るならば，シャルピー吸収エネルギーに従った材料設計が可能である。例えば外径610 mm，管厚19.1 mmのAPI X65でCO 2 圧力10 MPa の場合，必要吸収エネルギーが210 J以上となり，CO 2パイプラインでは通常のガスパイプラインに比べ高い吸収エネルギーが要求される。なお，強度グレードがX70以上の場合は実管での実証試験が必要とされている。\u3000日本および世界でCCSまたはCCUSと合わせたCO 2のパイプライン輸送の検討が進められている。大量輸送のためにはより高強度で大径のラインパイプが必要となり，図7のDNV設計式からも読み取れるように，より高い吸収エネルギーが求められる。JFEスチールは鋼板製造時の緻密な組織制御によって，高吸収エネルギー型の高強度ラインパイプを開発している 13）。表1，表2に開発したAPI X80 UOE鋼管の化学成分と機械的特性を示す。API X80 UOE鋼管はきわめて高い吸収エネルギーを有している。このような材質制御はほかのグレードのラインパイプにも適用できることから，CO 2 パイプライン用としてさまざまなラインパイプを供給可能である。',
    #         'passage_id': 0,
    #         'text': '1. **The article focuses on initiatives for energy transition in the pipe and tubular sector.** → It highlights JFE Steel\'s efforts in the transition to new energy forms, specifically within the pipe and tubular industry. 2. **Carbon neutrality requires energy transition to hydrogen and CO2 capture and storage (CCS).** → Achieving carbon neutrality involves shifting from fossil fuels to hydrogen and implementing CCS technologies. 3. **Technologies for energy transition to hydrogen and CCS are being developed globally.** → The article notes that these technologies are being pursued worldwide to achieve carbon neutrality. 4. **The article introduces pipe and tubular products needed in the hydrogen supply chain, including CCS.** → JFE Steel’s role in developing necessary products for hydrogen supply chains, including CCS, is emphasized. 5. **JFE Steel has been involved in developing many products for the oil and natural gas sector, contributing to global energy stability.** → JFE Steel\'s historical contributions to energy stability through product development in the oil and gas sector are acknowledged. 6. **New infrastructure is needed for the transport and storage of hydrogen and CO2, requiring materials with new performance characteristics like resistance to hydrogen embrittlement and carbonic acid corrosion.** → The transition to new energy sources necessitates new infrastructure with materials that can withstand specific challenges like hydrogen embrittlement and corrosion. 7. **JFE Steel has been working on material development for hydrogen transport pipelines, focusing on high-pressure conditions.** → The article describes JFE Steel\'s efforts in developing materials suited for hydrogen transport under high-pressure conditions. 8. **ASME B31.12 is a standard for hydrogen pipelines, requiring specific tests for rupture safety.** → The article mentions the importance of adhering to ASME B31.12 standards for ensuring the safety of hydrogen pipelines. 9. **JFE Steel\'s UOE steel pipes show higher rupture toughness in hydrogen than traditional materials.** → JFE Steel\'s UOE steel pipes are highlighted for their superior performance in terms of rupture toughness in hydrogen environments. 10. **"Mighty Seam®" is a technology developed by JFE Steel to enhance the reliability of electric fusion welds in pipelines.** → JFE Steel\'s innovation, Mighty Seam®, improves the reliability of electric fusion welds, crucial for pipeline integrity. 11. **Mighty Seam® has comparable or higher impact energy to the base material in electric fusion welds.** → The Mighty Seam® technology ensures that electric fusion welds have impact energy levels comparable to or exceeding those of the base material. 12. **JFE Steel, in'
    #     },
    # ]

    # documents = [
    #     {"doc_id": 1, "title": "Bridget Jones: The Edge of Reason (2004)", "text": "Bridget Jones: The Edge of Reason (2004) - Bridget is currently living a happy life with her lawyer boyfriend Mark Darcy, however not only does she start to become threatened and jealous of Mark's new young intern, she is angered by the fact Mark is a Conservative voter. With so many issues already at hand, things get worse for Bridget as her ex-lover, Daniel Cleaver, re-enters her life; the only help she has are her friends and her reliable diary.", "source": ""},
    #     {"doc_id": 2, "title": "Bridget Jones's Baby (2016)", "text": "Bridget Jones's Baby (2016) - Bridget Jones is struggling with her current state of life, including her break up with her love Mark Darcy. As she pushes forward and works hard to find fulfilment in her life seems to do wonders until she meets a dashing and handsome American named Jack Quant. Things from then on go great, until she discovers that she is pregnant but the biggest twist of all, she does not know if Mark or Jack is the father of her child.", "source": ""},
    #     {"doc_id": 3, "title": "Bridget Jones's Diary (2001)", "text": "Bridget Jones's Diary (2001) - Bridget Jones is a binge drinking and chain smoking thirty-something British woman trying to keep her love life in order while also dealing with her job as a publisher. When she attends a Christmas party with her parents, they try to set her up with their neighbours' son, Mark. After being snubbed by Mark, she starts to fall for her boss Daniel, a handsome man who begins to send her suggestive e-mails that leads to a dinner date. Daniel reveals that he and Mark attended college together, in that time Mark had an affair with his fiancée. Bridget decides to get a new job as a TV presenter after finding Daniel being frisky with a colleague. At a dinner party, she runs into Mark who expresses his affection for her, Daniel claims he wants Bridget back, the two fight over her and Bridget must make a decision who she wants to be with.", "source": ""},
    # ]

    # print(f"XXX documents = XXX{documents}XXX")

    render_dict['documents'] = documents

    messages = [
        {"role": "", "content": ""},  # a dummy message with an empty role. required for granite 4
    ]
    render_dict['messages'] = messages

    render_dict['strftime_now'] = strftime_now  # required for granite 3

    system_msg = chat_template.render(**render_dict)

    # print(f"XXX PRE system_msg = XXX{system_msg}XXX")

    # Remove prefix from system_msg
    # granite 3 uses normal but granite 4 uses escape.
    prefix_normal = "<|start_of_role|>system<|end_of_role|>"
    prefix_escape = html.escape(prefix_normal)
    prefix_start = -1
    prefix_normal_start = system_msg.find(prefix_normal)
    prefix_escape_start = system_msg.find(prefix_escape)
    if prefix_normal_start >= 0:
        prefix = prefix_normal
        prefix_start = prefix_normal_start
    elif prefix_escape_start >= 0:
        prefix = prefix_escape
        prefix_start = prefix_escape_start
    if prefix_start >= 0:
        system_msg = system_msg[prefix_start + len(prefix):]

    # Remove a trailing dummy message from granite 3 result. Granite 4 doesn't render unknown message.
    # granite 3 uses normal but granite 4 uses escape.
    dummy_normal = "<|start_of_role|><|end_of_role|><|end_of_text|>\n"
    dummy_escape = html.escape(dummy_normal)
    dummy_start = -1
    dummy_normal_start = system_msg.rfind(dummy_normal)
    dummy_escape_start = system_msg.rfind(dummy_escape)
    if dummy_normal_start >= 0:
        # dummy = dummy_normal
        dummy_start = dummy_normal_start
    elif dummy_escape_start >= 0:
        # dummy = dummy_escape
        dummy_start = dummy_escape_start
    if dummy_start >= 0:
        system_msg = system_msg[:dummy_start]

    # Remove suffix from system_msg
    # granite 3 uses normal but granite 4 uses escape.
    suffix_normal = "<|end_of_text|>\n"
    suffix_escape = html.escape(suffix_normal)
    suffix_start = -1
    suffix_normal_start = system_msg.rfind(suffix_normal)
    suffix_escape_start = system_msg.rfind(suffix_escape)
    if suffix_normal_start >= 0:
        # suffix = suffix_normal
        suffix_start = suffix_normal_start
    elif suffix_escape_start >= 0:
        # suffix = suffix_escape
        suffix_start = suffix_escape_start
    if suffix_start >= 0:
        system_msg = system_msg[:suffix_start]

    # print(f"XXX POST system_msg = XXX{system_msg}XXX")

    return system_msg


# ========================
# RAFT Config
# ========================
@dataclass
class RAFTConfig:
    k_passages: int = 5  # total retrieved passages
    max_tokens_per_chunk: int = 400
    chunk_overlap: int = 60
    p_include_oracle: float = 0.9  # probability to include oracle
    quote_begin: str = "##begin_quote##"
    quote_end: str = "##end_quote##"
    instruction_template: str = (
        "You are given a question and several passages (some are distractors). "
        "Quote exactly one span from a relevant passage, then explain reasoning, "
        "then provide the final answer. Ignore unrelated passages."
    )
    add_doc_ids: bool = True
    shuffle_passages: bool = True
    seed: int = 42
    # @@@ahoaho XXX
    student_model: Optional[str] = None


# ========================
# RAFT builder
# ========================
def build_raft_samples(
    hf_dataset,
    cfg: RAFTConfig = RAFTConfig(),
    answer_builder: Callable[
        [Dict[str, Any], str], Tuple[str, str]
    ] = default_answer_builder,
    text_field: str = "document",
    question_field: str = "question",
    group_by_doc: Optional[str] = "raw_document",
) -> List[Dict[str, Any]]:
    """
    Builds RAFT-style training samples from your dataset.
    """
    rng = np.random.default_rng(cfg.seed)

    # ---- Step 1: Chunk all docs ----
    all_chunks = []
    per_doc_chunks = defaultdict(list)

    tmp = []
    for i, ex in enumerate(hf_dataset):
        ex = dict(ex)
        ex["__row_id__"] = i
        tmp.append(ex)
    data = tmp

    def doc_id_for(ex):
        return (
            ex.get(group_by_doc)
            if group_by_doc and ex.get(group_by_doc)
            else f"doc_{ex['__row_id__']}"
        )

    for ex in data:
        doc_text = (ex.get(text_field) or "").strip()
        doc_id = doc_id_for(ex)
        chunks = chunk_text(doc_text, cfg.max_tokens_per_chunk, cfg.chunk_overlap)
        for j, ch in enumerate(chunks):
            gid = len(all_chunks)
            all_chunks.append({"doc_id": doc_id, "passage_id": j, "text": ch})
            per_doc_chunks[doc_id].append((ch, gid))

    if not all_chunks:
        return []

    # ---- Step 2: Fit TF-IDF retriever ----
    # @@@ahoaho XXX tokenize for Japanese
    # vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98)
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2), min_df=1, max_df=0.98)
    X = vect.fit_transform([c["text"] for c in all_chunks])

    def retrieve_k(query: str, k: int) -> List[int]:
        if not query.strip():
            return list(rng.choice(len(all_chunks), size=k, replace=False))
        qv = vect.transform([query])
        sims = cosine_similarity(X, qv).ravel()
        order = np.argsort(-sims)
        return list(map(int, order[:k]))

    # ---- Step 3: Build RAFT examples ----
    raft_records = []

    # @@@ahoaho XXX
    student_model = cfg.student_model
    # is_granite = False
    is_granite = student_model is not None and is_known_model(student_model, "granite")
    # is_granitemoehybrid = False
    is_granitemoehybrid = student_model is not None and is_known_model(student_model, "granitemoehybrid")
    if is_granite or is_granitemoehybrid:
        chat_template_str = load_tokenizer(student_model).chat_template
        jinja_env = jinja2.Environment()
        jinja_env.filters['tojson'] = tojson_allow_non_ascii  # mainly for granite 4
        chat_template = jinja_env.from_string(chat_template_str)

    for ex in data:
        q = (ex.get(question_field) or "").strip()
        if not q:
            continue

        cand_ids = retrieve_k(q, cfg.k_passages * 5)
        this_doc = doc_id_for(ex)
        oracle_gids = [gid for gid in cand_ids if all_chunks[gid]["doc_id"] == this_doc]

        include_oracle = (rng.random() < cfg.p_include_oracle) and len(oracle_gids) > 0
        oracle_gid = oracle_gids[0] if include_oracle else None

        chosen = []
        if oracle_gid is not None:
            chosen.append(oracle_gid)

        for gid in cand_ids:
            if len(chosen) >= cfg.k_passages:
                break
            if gid == oracle_gid:
                continue
            if all_chunks[gid]["doc_id"] != this_doc:
                chosen.append(gid)

        while len(chosen) < cfg.k_passages:
            gid = int(rng.integers(0, len(all_chunks)))
            if oracle_gid is None or gid != oracle_gid:
                chosen.append(gid)

        documents = []
        for gid in chosen:
            c = all_chunks[gid]
            doc_entry = {
                **(
                    {"doc_id": c["doc_id"], "passage_id": c["passage_id"]}
                    if cfg.add_doc_ids
                    else {}
                ),
                "text": c["text"],
            }
            documents.append(doc_entry)

        oracle_chunk = (
            all_chunks[oracle_gid]["text"]
            if oracle_gid is not None
            else documents[0]["text"]
        )
        support_quote, final_answer = answer_builder(ex, oracle_chunk)

        quote_wrapped = f"{cfg.quote_begin} {support_quote} {cfg.quote_end}"
        cot = "Reasoning: The quote supports the answer because ..."
        output = "\n".join([quote_wrapped, cot, f"Final Answer: {final_answer}"])

        if cfg.shuffle_passages:
            order = np.arange(len(documents))
            rng.shuffle(order)
            documents = [documents[i] for i in order]
            oracle_index = order.tolist().index(0) if oracle_gid is not None else None
        else:
            oracle_index = 0

        raft_record = {
            "question": q,
            "oracle_context": oracle_chunk if oracle_gid is not None else "",
            "cot_answer": output,
            "answer": final_answer,
            "instruction": cfg.instruction_template,
            "type": "with_oracle" if oracle_gid is not None else "no_oracle",
            "meta": {"source_row": ex["__row_id__"], "oracle_index": oracle_index},
        }

        # @@@ahoaho XXX
        if is_granite or is_granitemoehybrid:
            system_msg = render_system_message_granite(documents=documents, chat_template=chat_template)
            raft_record["system"] = system_msg
            raft_record["context"] = []
        else:
            raft_record["context"] = [d["text"] for d in documents]

        raft_records.append(raft_record)

    return Dataset.from_list(raft_records)


def build_messages(raft_record: Dict[str, Any]):
    """
    Construct RAFT-style chat messages for supervised fine-tuning.

    Input:
      raft_record: dict with keys:
        - "system"
        - "question"
        - "context" (list of passages)
        - "cot_answer" (the full target output)
        - "instruction" (optional high-level system instruction)

    Output:
      messages: list of {"role": "system"|"user"|"assistant", "content": str}
    """
    messages = []

    # 1. System message
    system_msg = raft_record.pop("system", None)
    if system_msg is not None:
        messages.append({"role": "system", "content": system_msg})

    # 2. User message: serialize passages + question
    context = raft_record["context"]
    question = raft_record["question"]
    if context:
        passages = "\n\n".join(
            [f"[Passage {i + 1}] {p}" for i, p in enumerate(context)]
        )
        user_msg = f"Passages:\n{passages}\n\nQuestion: {question}"
    else:
        user_msg = question

    # 3. Assistant message: the gold output
    assistant_msg = raft_record["answer"]

    return {
        "messages": messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }
