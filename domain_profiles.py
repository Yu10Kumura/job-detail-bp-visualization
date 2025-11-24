"""Domain profile definitions for manufacturing technical roles.
Provides get_domain_profile(industry, job_title) returning a profile dict.
Profile keys:
  name, industry, role
  core_terms, secondary_terms, optional_terms
  query_blocks: {category: [query strings with placeholders]}
  phase_overrides: {phase_x: {activities, inputs, outputs, tools, stakeholders, kpi, risks, countermeasures}}
  scale_stages, key_tests, stakeholder_roles, kpi_templates
Placeholders available in phase_overrides:
  {materials_core} {tools_core} {processes_core} {key_tests} {scale_stage} {reg_terms} {fail_terms} {stakeholder_matrix}
"""
from typing import Dict, Any
import re

# ─────────────────────────────────────────────
# 共通フェーズ適合性マップ / 技術職メタ構造 / 抽出フィルタ定義
# ─────────────────────────────────────────────

PHASE_AFFINITY_MAP = {
    'materials_or_products': {
        'phase_1': 0.8,
        'phase_2': 0.9,
        'phase_3': 1.0,
        'phase_4': 0.7,
        'phase_5': 0.3,
        'phase_6': 0.2,
        'phase_7': 0.4
    },
    'tools_and_equipment': {
        'phase_1': 0.7,
        'phase_2': 0.2,
        'phase_3': 0.8,
        'phase_4': 1.0,
        'phase_5': 0.9,
        'phase_6': 0.1,
        'phase_7': 0.5
    },
    'processes': {
        'phase_1': 0.3,
        'phase_2': 0.4,
        'phase_3': 1.0,
        'phase_4': 0.9,
        'phase_5': 0.6,
        'phase_6': 0.2,
        'phase_7': 0.7
    },
    'industry_specific_kpi': {
        'phase_1': 0.5,
        'phase_2': 0.9,
        'phase_3': 0.7,
        'phase_4': 0.6,
        'phase_5': 1.0,
        'phase_6': 0.4,
        'phase_7': 0.8
    },
    'constraints_or_regulations': {
        'phase_1': 1.0,
        'phase_2': 0.9,
        'phase_3': 0.3,
        'phase_4': 0.2,
        'phase_5': 0.8,
        'phase_6': 0.7,
        'phase_7': 0.4
    },
    'common_failures': {
        'phase_1': 0.4,
        'phase_2': 0.5,
        'phase_3': 0.8,
        'phase_4': 0.7,
        'phase_5': 0.9,
        'phase_6': 0.3,
        'phase_7': 1.0
    },
    'stakeholders': {
        'phase_1': 0.5, 'phase_2': 0.5, 'phase_3': 0.5,
        'phase_4': 0.5, 'phase_5': 0.5, 'phase_6': 0.5, 'phase_7': 0.5
    },
    'deliverables': {
        'phase_1': 0.6,
        'phase_2': 0.7,
        'phase_3': 0.9,
        'phase_4': 0.5,
        'phase_5': 1.0,
        'phase_6': 0.8,
        'phase_7': 0.6
    }
}

TECHNICAL_ROLE_META_STRUCTURE = {
    'phase_1': {
        'meta_purpose': '情報収集',
        'core_activities': ['技術トレンド調査', '規格・法規制調査', '既存技術分析'],
        'expected_deliverables': ['調査レポート', '技術要件候補', '規格適合性サマリ']
    },
    'phase_2': {
        'meta_purpose': '要件定義',
        'core_activities': ['性能要件定義', '法規制整理', '制約条件明確化'],
        'expected_deliverables': ['要件仕様書', '法規制適合一覧', 'KPI設定表']
    },
    'phase_3': {
        'meta_purpose': '設計・計画',
        'core_activities': ['技術仕様設計', '実験計画(DOE)', 'スケールアップ計画'],
        'expected_deliverables': ['設計書', 'DOE計画書', 'スケールアップ計画']
    },
    'phase_4': {
        'meta_purpose': '実行',
        'core_activities': ['試作実行', 'プロセス条件最適化', 'データ収集'],
        'expected_deliverables': ['試作品', '条件ログ', '実行記録']
    },
    'phase_5': {
        'meta_purpose': '検証・評価',
        'core_activities': ['性能試験', '規格適合評価', 'データ分析'],
        'expected_deliverables': ['試験レポート', '規格適合評価書', '性能マップ']
    },
    'phase_6': {
        'meta_purpose': '承認・リリース',
        'core_activities': ['設計承認', '量産移行判定', 'OEM提出'],
        'expected_deliverables': ['承認設計書', '量産条件書', 'OEM提出文書']
    },
    'phase_7': {
        'meta_purpose': '改善',
        'core_activities': ['フィールドデータ解析', '不具合対策', '次世代要件フィードバック'],
        'expected_deliverables': ['改善報告', '改良設計提案', '次期要件候補']
    }
}

EXCLUSION_PATTERNS = [
    r'株式会社', r'有限会社', r'合同会社', r'一般社団法人', r'一般財団法人', r'国立研究開発法人',
    r'独立行政法人', r'大学', r'研究所', r'センター', r'協会', r'学会', r'連盟', r'組合', r'省', r'庁',
    r'委員会', r'Inc\.?', r'Corp\.?', r'Ltd\.?', r'LLC', r'Co\.', r'University', r'Institute',
    r'Association', r'Society'
]

ALLOWED_ROLE_PATTERNS = [r'OEM', r'品質保証', r'製造技術', r'開発部門', r'エンジニア', r'担当', r'部門', r'チーム']

ALLOWED_CATEGORY_PATTERNS = {
    'materials_or_products': [r'NCM[0-9]+', r'LFP', r'NCA', r'LiPF6', r'Li[A-Za-z0-9]+', r'セパレータ', r'バインダー', r'スラリー', r'焼結', r'混練'],
    'tools_and_equipment': [r'XRD', r'SEM', r'EDS', r'AFM', r'VSM', r'JMP', r'Minitab', r'CAD', r'CAE', r'FEA', r'LCR'],
    'processes': [r'混練', r'スラリー', r'塗工', r'乾燥', r'焼結', r'DOE', r'フォーメーション', r'化成'],
    'industry_specific_kpi': [r'エネルギー密度', r'サイクル寿命', r'内部抵抗', r'Wh/kg', r'Ah', r'歩留まり'],
    'constraints_or_regulations': [r'UN38\.3', r'IEC[0-9]+', r'AEC-Q[0-9]+', r'RoHS', r'REACH', r'JIS', r'ISO', r'規格'],
    'common_failures': [r'劣化', r'膨張', r'短絡', r'熱暴走', r'SEI', r'デンドライト', r'ガス'],
    'stakeholders': [r'OEM', r'品質保証', r'製造技術', r'開発部門', r'プロセスエンジニア', r'法務', r'環境安全'],
    'deliverables': [r'仕様書', r'計画書', r'レポート', r'要件仕様', r'評価レポート', r'試験レポート']
}

def filter_category_items(category: str, items):
    """カテゴリ別許可/除外ルールに基づきアイテムフィルタ"""
    allowed_patterns = ALLOWED_CATEGORY_PATTERNS.get(category, [])
    filtered = []
    for it in items:
        if any(re.search(pat, it) for pat in EXCLUSION_PATTERNS):
            # stakeholder の場合は役割抽出できる可能性
            if category == 'stakeholders' and any(re.search(p, it) for p in ALLOWED_ROLE_PATTERNS):
                # 役割語のみ抽出
                role = next((p for p in ALLOWED_ROLE_PATTERNS if re.search(p, it)), None)
                if role:
                    filtered.append(role)
            continue
        if allowed_patterns:
            if not any(re.search(p, it) for p in allowed_patterns):
                continue
        # 抽象語のみの除外（短すぎる・一般語）
        if len(it) < 2:
            continue
        filtered.append(it)
    return filtered

# ─────────────────────────────────────────────
# Phase A 追加: 検索範囲 / 技術ヒント / 専門KPI 定義
# ─────────────────────────────────────────────

SEARCH_SCOPES = {
    "EV材料開発": """
必ず以下の技術領域を中心に検索すること：
• 電池材料：正極(NCM/NCA/LFP) 負極(黒鉛/Si) 電解液(LiPF6/溶媒) セパレータ(PP/PE) バインダー(PVDF/CMC)
• プロセス：混練 分散 スラリー調整 塗工 乾燥 焼結 ロールプレス 粒径分布測定 粘度測定
• 評価技術：XRD SEM TEM FIB-SEM 粒径分布 粘度 電気化学評価 インピーダンス LCR 充放電 サイクル
• 信頼性：高温高湿試験 熱サイクル DCIR サイクル寿命 容量維持率 レート特性
• 規格：AEC-Q200 UN38.3 IEC62133 RoHS REACH JIS C 8711
• KPI：エネルギー密度 容量維持率 初期DCIR DCIR低下率 接着強度 シート膨れ率
""",
    "電池セル開発": """
必ず以下の技術領域を中心に検索すること：
• セル形状：コインセル パウチセル 円筒セル 角形セル 18650 21700 4680
• 製造工程：巻回 積層 溶接 封止 化成 フォーメーション 初期充電
• 評価：充放電試験 インピーダンス SOC SOH レート試験 サイクル試験 温度分布
• 信頼性：熱暴走 釘刺し 過充電 過放電 膨張 内部短絡 安全性試験
• KPI：内部抵抗増加率 容量劣化率 サイクル寿命 温度均一性 セル間電圧ばらつき
""",
    "モーター設計": """
必ず以下の技術領域を中心に検索すること：
• 解析：電磁界解析 熱解析 構造解析 NVH解析 モーダル解析 FEA JMAG ANSYS Maxwell MotorCAD
• 設計要素：巻線 スロット 冷却 磁石材 磁束 密度 トルク コギングトルク 効率マップ 鉄損 銅損
• 試験：性能試験 効率測定 NVH試験 熱上昇試験 振動試験 トルク脈動測定
• KPI：トルク密度 効率 トルクリップル NVH 鉄損 総損失 力率 出力密度
"""
}

TECHNICAL_HINT_SETS = {
    "EV材料開発": {
        "materials_or_products": [
            # 正極/負極/電解液/バインダー/固体電解質/補助材 深堀り
            "NCM811","NCM622","NCM523","LFP","LTO","NCA","LiPF6","LiBF4","EC","DMC","EMC","DEC",
            "PVDF","PVdF","CMC","SBR","NMP","PP/PEセパレータ","セラミックコートセパレータ","アルミ箔","銅箔","カーボンブラック","CNT",
            "SiOx","Si-C複合負極","ハードカーボン","グラファイト","活物質","導電助剤","Li6PS5Cl","LLZO","コインセル","パウチセル","円筒セル","角形セル"
        ],
        "tools_and_equipment": [
            # 分析/評価/工程装置 + 電気化学評価
            "XRD","SEM","FE-SEM","TEM","FIB-SEM","AFM","ICP-MS","ICP-AES","TGA","DSC","DTA","FT-IR","ラマン分光",
            "レーザー回折粒度分布計","混練機","プラネタリーミキサー","ボールミル","スロットダイコーター","ロールプレス","焼結炉",
            "LCRメータ","インピーダンスアナライザ","サイクラー","環境試験装置","熱衝撃試験機","恒温恒湿槽","EIS（電気化学インピーダンス）","CV（サイクリックボルタンメトリ）","粘度計","レオメータ"
        ],
        "processes": [
            # スラリー詳細工程/電極成形/化成/物性測定
            "混練","分散","脱泡","スラリー調整","スラリー粘度管理","Binder濃度最適化","分散剤添加","塗工","塗布","乾燥",
            "カレンダー加工","プレス","焼結","裁断","積層","巻回","封止","初期充電","エージング","化成",
            "粒径分布測定","粘度測定","密度測定","電極密度制御"
        ],
        "industry_specific_kpi": [
            # 材料/工程/電気化学特化KPI
            "エネルギー密度","出力密度","Wh/kg","W/kg","初期DCIR","DCIR低下率","容量維持率","サイクル寿命","カレンダー寿命",
            "充放電サイクル特性","レート特性","高温保持後容量維持率","低温放電特性","接着強度","剥離強度","シート膨れ率",
            "電極密度","活物質充填率","導電ネットワーク","粒径D50","Cpk","バッチ内変動"
        ],
        "constraints_or_regulations": [
            "AEC-Q200","UN38.3","IEC62133","UL1642","RoHS","REACH","JIS C 8711","JIS C 8714","IATF16949","ISO9001","UL2580","ISO26262","VDA規格","JIS C 8715-1"
        ],
        "deliverables": [
            "配合仕様書","工程条件書","材料規格書","信頼性試験報告書","PPAP文書","FMEA","工程能力評価書","承認図面","スラリー粘度レポート","DCIR評価報告"
        ],
        "common_failures": [
            "SEI形成","デンドライト","熱暴走","膨張","ガス発生","スラリー凝集","粒径ムラ","電極膨れ","バインダー不足","導電パス不足","集電体腐食"
        ]
    },
    "電池セル開発": {
        "materials_or_products": [
            "コインセル","パウチセル","円筒セル","角形セル","18650","21700","26650","4680","プリズマティック","正極タブ","負極タブ","アルミラミネート",
            "セパレータ","電解液","ガスベント","タブ溶接部"
        ],
        "tools_and_equipment": [
            "充放電試験装置","サイクラー","インピーダンスアナライザ","SOCテスター","SOHアナライザ","カロリーメータ","巻回機","積層機","レーザー溶接機","乾燥炉","化成装置","セル外観検査装置"
        ],
        "industry_specific_kpi": [
            "SOC","SOH","DOD","C-rate","内部抵抗","内部抵抗増加率","容量劣化率","セル間電圧ばらつき","温度分布均一性","ガス発生頻度","セル膨張率"
        ],
        "processes": [
            "巻回","積層","溶接","封止","化成","フォーメーション","初期充電","エージング","乾燥","タブ溶接","ガスベント成形"
        ]
    },
    "モーター設計": {
        "tools_and_equipment": [
            "ANSYS Maxwell","JMAG","MotorCAD","トルクリップル測定器","鉄損測定装置","振動計","騒音計","熱電対","高速度カメラ","温度監視システム"
        ],
        "industry_specific_kpi": [
            "トルク密度","出力密度","効率マップ","トルクリップル","コギングトルク","鉄損","銅損","総損失","力率","NVH","温度上昇率"
        ],
        "processes": [
            "電磁界解析","熱解析","構造解析","モーダル解析","NVH解析","巻線設計","冷却設計","トルクリップル最適化","効率マップ測定"
        ]
    }
}

DOMAIN_SPECIFIC_KPIS = {
    "EV材料開発": [
        "初期DCIR","DCIR低下率","容量維持率","サイクル寿命","高温保持後容量維持率","レート特性","接着強度","シート膨れ率","粒径D50","Cpk"
    ],
    "電池セル開発": [
        "内部抵抗","内部抵抗増加率","容量劣化率","サイクル寿命","セル間電圧ばらつき","温度分布均一性","SOC精度","SOH推定精度","ガス発生頻度"
    ],
    "モーター設計": [
        "トルク密度","効率","トルクリップル","NVH","鉄損","総損失","力率","出力密度","温度上昇率"
    ]
}

def _attach_common(profile: Dict[str, Any], profile_key: str = None) -> Dict[str, Any]:
    profile['phase_affinity_map'] = PHASE_AFFINITY_MAP
    profile['meta_structure'] = TECHNICAL_ROLE_META_STRUCTURE
    profile['exclusion_patterns'] = EXCLUSION_PATTERNS
    profile['allowed_category_patterns'] = ALLOWED_CATEGORY_PATTERNS
    if profile_key:
        profile['search_scope'] = SEARCH_SCOPES.get(profile_key, "")
        profile['technical_hints'] = TECHNICAL_HINT_SETS.get(profile_key, {})
        profile['domain_kpi'] = DOMAIN_SPECIFIC_KPIS.get(profile_key, [])
    else:
        profile['search_scope'] = ""
        profile['technical_hints'] = {}
        profile['domain_kpi'] = []
    return profile

BASE_PHASE_KEYS = [
    "phase_1","phase_2","phase_3","phase_4","phase_5","phase_6","phase_7"
]

def get_domain_profile(industry: str, job_title: str) -> Dict[str, Any]:
    text = f"{industry} {job_title}".lower()

    # EV materials development profile
    if any(k in text for k in ["ev","battery","電池"]) and any(k in text for k in ["材料","material"]):
        return _attach_common({
            "name": "EV材料開発",
            "industry": industry,
            "role": job_title,
            "core_terms": ["NCM811","LFP","NCA","SEI","LiPF6","セパレータ","スラリー","焼結","混練"],
            "secondary_terms": ["正極","負極","バインダー","粒径分布","乾燥","溶媒","ドープ","セル評価","歩留まり"],
            "optional_terms": ["コインセル","インピーダンス","ESR","熱伝導","OEMSOP","BMS","安全性試験"],
            "scale_stages": ["ラボ","パイロット","量産"],
            "key_tests": ["容量","ESR","インピーダンス","熱サイクル","高温高湿"],
            "stakeholder_roles": {
                "材料開発エンジニア": "R",
                "プロセスエンジニア": "C",
                "品質保証": "C",
                "開発部門長": "A",
                "OEM窓口": "I",
                "製造技術": "C",
                "法務・環境安全": "C"
            },
            "query_blocks": {
                "materials_or_products": ["EV battery cathode 正極 負極 材料 NCM811 LFP NCA 粒径 分散"],
                "tools_and_equipment": ["EV battery 材料評価 XRD SEM EDS 混練 コーター 焼結炉"],
                "processes": ["電池 材料 スラリー 混練 分散 乾燥 焼結 ラミネーション 塗工"],
                "industry_specific_kpi": ["EV battery サイクル寿命 エネルギー密度 レート性能 歩留まり ばらつき"],
                "constraints_or_regulations": ["EV battery 規格 UN/ECE REACH AEC-Q200 JIS IEC RoHS"],
                "common_failures": ["EV battery 材料 不具合 リチウムメッキ ガス発生 劣化 インピーダンス上昇"],
                "stakeholders": ["EV battery OEM 品質保証 製造技術 プロセスエンジニア"],
                "deliverables": ["EV battery 材料 評価レポート DOE計画書 スケールアップ計画 仕様書"],
            },
            "kpi_templates": {
                "phase_1": "市場ニーズ整合度 規格カバー率",
                "phase_2": "要件明確度 規格適合率",
                "phase_3": "設計手戻り件数 スケジュール遵守率",
                "phase_4": "試作成功率 歩留まり プロセス安定性",
                "phase_5": "試験合格率 性能ばらつきσ",
                "phase_6": "承認リードタイム 初回量産不良率",
                "phase_7": "不具合削減率 改善実施率"
            },
            "phase_overrides": {
                "phase_1": {
                    "activities": "EV用材料技術トレンド調査・規格(JIS/IEC/AEC-Q200)確認・競合分析",
                    "inputs": "特許/文献/規格文書",
                    "outputs": "材料特性要件リスト 規格適合性サマリ",
                    "tools": "特許DB 文献DB 規格DB",
                    "stakeholders": "材料開発エンジニア(R) 規格担当(C) 品質保証(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "市場ニーズ整合度 規格カバー率",
                    "risks": "最新規格取りこぼし OEM要求未反映",
                    "countermeasures": "定期規格レビュー OEM技術MTG"
                },
                "phase_2": {
                    "activities": "電気/熱/信頼性要件定義 法規(RoHS/REACH)整理",
                    "inputs": "市場ニーズ 既存評価データ 規格サマリ",
                    "outputs": "材料要件仕様書 法規制適合一覧",
                    "tools": "要件管理ツール ドキュメント管理",
                    "stakeholders": "材料開発エンジニア(R) 製品マネージャー(C) 法務・環境安全(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "要件明確度 規格適合率",
                    "risks": "仕様曖昧 OEM要求未反映",
                    "countermeasures": "ステークホルダー要件レビュー"
                },
                "phase_3": {
                    "activities": "配合設計 粒径分布/焼結条件設計 DOE試験計画 ラボ→パイロット計画",
                    "inputs": "材料要件仕様書",
                    "outputs": "配合レシピ DOE計画書 スケールアップ計画",
                    "tools": "統計解析(DOE) プロセスシミュレーション",
                    "stakeholders": "材料開発エンジニア(R) プロセスエンジニア(C) 品質保証(C) 開発部門長(A) 製造技術(I)",
                    "kpi": "設計手戻り件数 スケジュール遵守率",
                    "risks": "ラボと量産条件乖離 シミュレーション不一致",
                    "countermeasures": "両条件検証 レビューゲート設定"
                },
                "phase_4": {
                    "activities": "ラボ試作 スラリー調整 塗工 乾燥 焼結 パイロット試作",
                    "inputs": "配合レシピ DOE計画 スケールアップ計画",
                    "outputs": "ラボ試作品 パイロット試作品 条件ログ",
                    "tools": "混練機 コーター 焼結炉 モニタリングシステム",
                    "stakeholders": "材料開発エンジニア(R) 製造技術(R) 品質管理(C) 開発部門長(A)",
                    "kpi": "試作成功率 歩留まり プロセス安定性",
                    "risks": "スケールアップ特性劣化 設備制約",
                    "countermeasures": "条件マトリクス検証 設備能力明文化"
                },
                "phase_5": {
                    "activities": "電気特性/信頼性試験(容量 ESR インピーダンス 熱サイクル 高温高湿)",
                    "inputs": "試作品 条件ログ",
                    "outputs": "試験レポート 規格適合評価 性能マップ",
                    "tools": "LCRメータ 環境試験装置 品質管理システム",
                    "stakeholders": "品質保証(R) 材料開発エンジニア(C) 製造技術(C) 品質保証部門長(A) OEM窓口(I)",
                    "kpi": "試験合格率 性能ばらつきσ",
                    "risks": "評価基準不一致 試験条件不足",
                    "countermeasures": "評価基準事前合意 評価項目テンプレ化"
                },
                "phase_6": {
                    "activities": "設計承認 量産移行判定 OEM提出(PPAP等) 量産条件凍結",
                    "inputs": "評価レポート 規格試験結果",
                    "outputs": "承認設計書 量産条件シート OEM提出ドキュメント",
                    "tools": "PLM ワークフロー ドキュメント管理",
                    "stakeholders": "開発部門長(R/A) 製造部門長(C) 品質保証部門長(C) OEM窓口(I)",
                    "kpi": "承認リードタイム 初回不良率",
                    "risks": "承認遅延 ドキュメント不備",
                    "countermeasures": "承認プロセス標準化 チェックリスト運用"
                },
                "phase_7": {
                    "activities": "量産後不具合分析 フィールドデータ解析 改善提案 次世代材料フィードバック",
                    "inputs": "量産品質データ フィールド不具合 OEMクレーム",
                    "outputs": "改善報告 改良設計提案 次期要件インプット",
                    "tools": "不具合DB BIツール 統計解析ツール",
                    "stakeholders": "製造部門(R) 品質保証(R) 材料開発エンジニア(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "不具合削減率 改善実施率 クレーム件数",
                    "risks": "改善継続性途切れ データ品質不足",
                    "countermeasures": "改善会議定例化 データ品質モニタ"
                }
            }
        }, "EV材料開発")

    # Battery cell development profile
    if any(k in text for k in ["cell","セル"]) and any(k in text for k in ["開発","設計","engineer"]):
        return _attach_common({
            "name": "電池セル開発",
            "industry": industry,
            "role": job_title,
            "core_terms": ["正極活物質","負極活物質","セパレータ","電解液","セルスタック","エネルギー密度","サイクル寿命"],
            "secondary_terms": ["レート性能","安全性試験","セル膨張","内部抵抗","熱暴走","バランス充電"],
            "optional_terms": ["プリマテリアル","コインセル","パウチセル","18650","4680"],
            "scale_stages": ["ラボ","パイロット","量産"],
            "key_tests": ["充放電試験","レート試験","寿命試験","安全性試験","内部抵抗測定"],
            "stakeholder_roles": {
                "セル開発エンジニア": "R",
                "材料開発エンジニア": "C",
                "品質保証": "C",
                "製造技術": "C",
                "開発部門長": "A",
                "OEM窓口": "I",
                "安全規格担当": "C"
            },
            "query_blocks": {
                "materials_or_products": ["battery cell 正極 負極 セパレータ 電解液 サイクル寿命 エネルギー密度"],
                "tools_and_equipment": ["battery cell 評価 LCR インピーダンス 熱試験 XRD SEM"],
                "processes": ["battery cell スタック 組立 フォーメーション 化成 レート検証"],
                "industry_specific_kpi": ["battery cell エネルギー密度 サイクル寿命 内部抵抗 レート性能"],
                "constraints_or_regulations": ["battery cell UN/ECE REACH AEC-Q200 安全規格"],
                "common_failures": ["battery cell 膨張 劣化 内部短絡 熱暴走 SEI不安定"],
                "stakeholders": ["battery cell OEM 品質保証 製造技術 安全規格"] ,
                "deliverables": ["battery cell 設計仕様書 評価レポート 化成条件 マスバランス"]
            },
            "kpi_templates": {},
            "phase_overrides": {
                "phase_1": {
                    "activities": "セル技術トレンド調査 既存セル分解分析 安全規格(UN/ECE/AEC-Q200)確認",
                    "inputs": "特許/文献/規格文書 既存セル分析レポート",
                    "outputs": "セル構造要件リスト 規格適合性サマリ",
                    "tools": "特許DB 文献DB 分析ツール(XRD/SEM/LCR)",
                    "stakeholders": "セル開発エンジニア(R) 安全規格担当(C) 品質保証(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "規格カバー率 技術トレンド反映度",
                    "risks": "最新規格未反映 分解分析不十分",
                    "countermeasures": "定期規格レビュー 追加分析計画"
                },
                "phase_2": {
                    "activities": "性能/寿命/安全要件定義 化成条件初期要件設定",
                    "inputs": "セル構造要件リスト 規格適合性サマリ 既存試験データ",
                    "outputs": "セル要件仕様書 化成前処理条件案",
                    "tools": "要件管理ツール ドキュメント管理",
                    "stakeholders": "セル開発エンジニア(R) 材料開発エンジニア(C) 安全規格担当(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "要件明確度 規格適合率",
                    "risks": "要求曖昧化 安全要件抜け",
                    "countermeasures": "OEMレビュー 要件レビューミーティング"
                },
                "phase_3": {
                    "activities": "セルスタック設計 マスバランス計算 化成プロトコル設計 DOE試験計画",
                    "inputs": "セル要件仕様書 化成前処理条件案",
                    "outputs": "スタック設計図 化成プロトコル DOE計画書",
                    "tools": "CAD LCR インピーダンス解析 統計解析(DOE)",
                    "stakeholders": "セル開発エンジニア(R) 材料開発エンジニア(C) 品質保証(C) 開発部門長(A) 製造技術(I)",
                    "kpi": "設計手戻り件数 スケジュール遵守率",
                    "risks": "内部抵抗予測不一致 マスバランス誤差",
                    "countermeasures": "計算モデル検証 中間レビュー"
                },
                "phase_4": {
                    "activities": "試作セル組立 フォーメーション 化成 条件ログ収集",
                    "inputs": "スタック設計図 化成プロトコル DOE計画書",
                    "outputs": "試作セル群 条件ログ 化成結果",
                    "tools": "組立治具 化成装置 LCRメータ 監視システム",
                    "stakeholders": "セル開発エンジニア(R) 製造技術(R) 品質保証(C) 開発部門長(A)",
                    "kpi": "試作成功率 内部抵抗目標達成率",
                    "risks": "セル膨張 化成異常",
                    "countermeasures": "化成条件微調整 監視閾値最適化"
                },
                "phase_5": {
                    "activities": "充放電試験 レート試験 寿命試験 安全性試験",
                    "inputs": "試作セル群 条件ログ",
                    "outputs": "試験レポート 性能マップ 規格適合評価",
                    "tools": "充放電試験装置 環境試験機 インピーダンス測定",
                    "stakeholders": "品質保証(R) セル開発エンジニア(C) 安全規格担当(C) 品質保証部門長(A) OEM窓口(I)",
                    "kpi": "サイクル寿命達成率 レート性能達成率",
                    "risks": "試験条件不整合 データばらつき過大",
                    "countermeasures": "試験条件標準化 再測定プロトコル"
                },
                "phase_6": {
                    "activities": "設計承認 パイロット→量産移行判定 OEM提出",
                    "inputs": "試験レポート 規格適合評価",
                    "outputs": "承認セル設計書 量産化成条件 OEM提出文書",
                    "tools": "PLM ワークフロー ドキュメント管理",
                    "stakeholders": "開発部門長(R/A) 品質保証部門長(C) 製造部門長(C) OEM窓口(I)",
                    "kpi": "承認リードタイム 初回不良率",
                    "risks": "承認遅延 化成条件不明確",
                    "countermeasures": "承認チェックリスト 化成条件凍結手続き"
                },
                "phase_7": {
                    "activities": "量産後性能監視 フィールド不具合分析 改善提案 次期セル要件フィードバック",
                    "inputs": "量産品質データ フィールド不具合 OEMクレーム",
                    "outputs": "改善報告 改良設計提案 次期要件候補",
                    "tools": "不具合DB BIツール 統計解析ツール",
                    "stakeholders": "製造部門(R) 品質保証(R) セル開発エンジニア(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "性能維持率 不具合削減率",
                    "risks": "改善停滞 データ品質不足",
                    "countermeasures": "定例改善会議 データ品質モニタ"
                }
            }
        }, "電池セル開発")

    # Motor design profile
    if any(k in text for k in ["motor","モーター","eモーター"]) and any(k in text for k in ["設計","開発","engineer"]):
        return _attach_common({
            "name": "モーター設計",
            "industry": industry,
            "role": job_title,
            "core_terms": ["ステータ","ロータ","磁石材","トルク定数","効率マップ","NVH","コギングトルク"],
            "secondary_terms": ["損失解析","有限要素解析","巻線抵抗","逆起電力","熱解析"],
            "optional_terms": ["インバータ連携","冷却チャネル","スロット形状","磁束密度"],
            "scale_stages": ["設計","試作","耐久試験","量産"],
            "key_tests": ["性能試験","効率測定","NVH試験","熱上昇試験"],
            "stakeholder_roles": {
                "モーター設計エンジニア": "R",
                "CAE解析エンジニア": "C",
                "製造技術": "C",
                "品質保証": "C",
                "開発部門長": "A",
                "OEM窓口": "I"
            },
            "query_blocks": {
                "materials_or_products": ["EV motor 磁石材 ステータ ロータ 銅巻線"],
                "tools_and_equipment": ["EV motor 解析 CAE FEA NVH 測定装置"],
                "processes": ["EV motor 設計 試作 巻線 組立 バランス調整"],
                "industry_specific_kpi": ["EV motor 効率 トルク密度 NVH コギングトルク"],
                "constraints_or_regulations": ["EV motor 規格 ISO IEC 安全規格"],
                "common_failures": ["EV motor 振動 過熱 絶縁劣化 トルク脈動"],
                "stakeholders": ["EV motor 製造技術 品質保証 OEM"],
                "deliverables": ["EV motor 設計図 面 寸法検証レポート 試験結果"]
            },
            "kpi_templates": {},
            "phase_overrides": {
                "phase_1": {
                    "activities": "モーター技術トレンド調査 磁石材特性比較 競合効率マップ分析 規格(ISO/IEC)確認",
                    "inputs": "特許/文献/性能カタログ 規格文書",
                    "outputs": "性能要件候補リスト 規格適合性サマリ",
                    "tools": "特許DB 文献DB 性能データベース",
                    "stakeholders": "モーター設計エンジニア(R) CAE解析エンジニア(C) 品質保証(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "規格カバー率 技術トレンド反映度",
                    "risks": "最新効率技術取りこぼし 規格未反映",
                    "countermeasures": "定期技術レビュー 規格更新監視"
                },
                "phase_2": {
                    "activities": "トルク/効率/NVH要件定義 冷却方式選定 初期磁石材選定",
                    "inputs": "性能要件候補リスト 規格適合性サマリ 過去試験データ",
                    "outputs": "モーター要件仕様書 冷却/磁石材選定根拠",
                    "tools": "要件管理ツール ドキュメント管理",
                    "stakeholders": "モーター設計エンジニア(R) CAE解析エンジニア(C) 品質保証(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "要件明確度 規格適合率",
                    "risks": "NVH要件曖昧 冷却余裕不足",
                    "countermeasures": "要件レビュー NVHベンチマーク"
                },
                "phase_3": {
                    "activities": "ステータ/ロータ形状設計 巻線/スロット設計 FEA解析 NVH予測",
                    "inputs": "モーター要件仕様書",
                    "outputs": "設計図 FEA解析レポート NVH予測レポート",
                    "tools": "FEAソフト NVH解析ツール CAD",
                    "stakeholders": "モーター設計エンジニア(R) CAE解析エンジニア(C) 品質保証(C) 開発部門長(A) 製造技術(I)",
                    "kpi": "設計手戻り件数 解析一致率",
                    "risks": "解析と実測乖離 過熱リスク未検出",
                    "countermeasures": "解析モデル検証 熱シミュレーション追加"
                },
                "phase_4": {
                    "activities": "試作部品加工 巻線組立 バランス調整 初期性能測定",
                    "inputs": "設計図 FEA解析レポート",
                    "outputs": "試作モーター 初期性能ログ",
                    "tools": "巻線装置 バランス測定装置 トルク計",
                    "stakeholders": "モーター設計エンジニア(R) 製造技術(R) 品質保証(C) 開発部門長(A)",
                    "kpi": "試作成功率 初期効率達成率",
                    "risks": "トルク脈動 過熱",
                    "countermeasures": "組立条件最適化 バランス再調整"
                },
                "phase_5": {
                    "activities": "性能試験 効率マップ測定 NVH試験 熱上昇試験",
                    "inputs": "試作モーター 初期性能ログ",
                    "outputs": "試験レポート 効率/NVH/熱評価",
                    "tools": "性能試験装置 NVH測定システム 熱試験装置",
                    "stakeholders": "品質保証(R) モーター設計エンジニア(C) CAE解析エンジニア(C) 品質保証部門長(A) OEM窓口(I)",
                    "kpi": "効率目標達成率 NVH基準達成率",
                    "risks": "NVH過大 効率不足",
                    "countermeasures": "設計要素微調整 NVH原因分析"
                },
                "phase_6": {
                    "activities": "設計承認 量産移行判定 OEM提出",
                    "inputs": "試験レポート 効率/NVH/熱評価",
                    "outputs": "承認設計書 量産組立条件 OEM提出文書",
                    "tools": "PLM ワークフロー ドキュメント管理",
                    "stakeholders": "開発部門長(R/A) 製造部門長(C) 品質保証部門長(C) OEM窓口(I)",
                    "kpi": "承認リードタイム 初回不良率",
                    "risks": "承認遅延 ドキュメント不足",
                    "countermeasures": "承認チェックリスト 組立条件凍結"
                },
                "phase_7": {
                    "activities": "量産後性能監視 フィールド不具合解析 改善提案 次期設計フィードバック",
                    "inputs": "量産品質データ フィールド不具合 OEMクレーム",
                    "outputs": "改善報告 改良設計案 次期要件候補",
                    "tools": "不具合DB BIツール 解析ツール",
                    "stakeholders": "製造部門(R) 品質保証(R) モーター設計エンジニア(C) 開発部門長(A) OEM窓口(I)",
                    "kpi": "不具合削減率 改善実施率",
                    "risks": "改善停滞 データ不足",
                    "countermeasures": "定例改善会議 データ品質監視"
                }
            }
        }, "モーター設計")

    # Production engineering profile
    if any(k in text for k in ["生産技術","production engineering","manufacturing engineering"]) :
        return _attach_common({
            "name": "生産技術",
            "industry": industry,
            "role": job_title,
            "core_terms": ["ラインバランシング","タクトタイム","OEE","歩留まり","設備保全","SMED"],
            "secondary_terms": ["自働化","予知保全","治具設計","段取り改善","レイアウト最適化"],
            "optional_terms": ["IoTセンサ","デジタルツイン","AGV","MES","SCADA"],
            "scale_stages": ["ラボ","試作ライン","量産ライン"],
            "key_tests": ["能力検証","品質安定性試験"],
            "stakeholder_roles": {
                "生産技術": "R",
                "品質保証": "C",
                "設備保全": "C",
                "工場長": "A",
                "安全衛生": "C"
            },
            "query_blocks": {
                "materials_or_products": ["production engineering 治具 材料 部品 歩留まり"],
                "tools_and_equipment": ["production engineering 設備 モニタリング IoT MES"],
                "processes": ["production engineering 工程最適化 レイアウト 改善 SMED TPM"],
                "industry_specific_kpi": ["production engineering OEE 歩留まり タクトタイム"],
                "constraints_or_regulations": ["production engineering 安全規格 ISO JIS 労働安全"],
                "common_failures": ["production engineering ダウンタイム 不良率 ボトルネック"],
                "stakeholders": ["production engineering 品質保証 設備保全 工場長"],
                "deliverables": ["production engineering 改善報告書 工程設計書 設備仕様書"]
            },
            "kpi_templates": {},
            "phase_overrides": {
                "phase_1": {
                    "activities": "現行ライン性能/OEE分析 ボトルネック特定 安全規格遵守状況確認",
                    "inputs": "ライン稼働データ 不良率レポート 安全監査結果",
                    "outputs": "改善課題リスト 現状性能ベースライン",
                    "tools": "MES 生産監視BI レポートシステム",
                    "stakeholders": "生産技術(R) 品質保証(C) 設備保全(C) 工場長(A) 安全衛生(C)",
                    "kpi": "データ取得率 問題抽出網羅度",
                    "risks": "データ欠損 ボトルネック過小評価",
                    "countermeasures": "データ品質チェック 追加計測設定"
                },
                "phase_2": {
                    "activities": "改善要件定義 タクト/歩留まり目標設定 SMED/TPM適用範囲決定",
                    "inputs": "改善課題リスト 現状性能ベースライン",
                    "outputs": "改善要件仕様書 KPI設定表",
                    "tools": "要件管理ツール BI分析",
                    "stakeholders": "生産技術(R) 品質保証(C) 設備保全(C) 工場長(A)",
                    "kpi": "要件明確度 KPI妥当性",
                    "risks": "目標非現実的 要件抜け",
                    "countermeasures": "レビュー会議 ベンチマーク比較"
                },
                "phase_3": {
                    "activities": "ラインバランシング設計 レイアウト最適化 治具/設備改善案設計",
                    "inputs": "改善要件仕様書 現状レイアウト",
                    "outputs": "改善レイアウト案 治具設計図 設備改善仕様書",
                    "tools": "レイアウト設計ツール シミュレーションソフト CAD",
                    "stakeholders": "生産技術(R) 設備保全(C) 品質保証(C) 工場長(A)",
                    "kpi": "設計手戻り件数 レイアウト効率改善率",
                    "risks": "レイアウト制約未考慮 治具干渉",
                    "countermeasures": "現場レビュー 3D干渉確認"
                },
                "phase_4": {
                    "activities": "設備改造 治具製作 SMED導入 パイロット改善試行",
                    "inputs": "改善レイアウト案 治具設計図 設備改善仕様書",
                    "outputs": "改造設備 新治具 パイロット稼働ログ",
                    "tools": "工作機械 計測機器 MES",
                    "stakeholders": "生産技術(R) 設備保全(R) 品質保証(C) 工場長(A)",
                    "kpi": "稼働率改善率 SMED時間短縮率",
                    "risks": "改造遅延 稼働安定化遅延",
                    "countermeasures": "進捗モニタリング 予備計画"
                },
                "phase_5": {
                    "activities": "能力検証 タクト/歩留まり測定 安定性評価",
                    "inputs": "パイロット稼働ログ 改造設備",
                    "outputs": "能力検証レポート KPI達成状況",
                    "tools": "MES 品質測定システム BI分析",
                    "stakeholders": "品質保証(R) 生産技術(C) 設備保全(C) 工場長(A)",
                    "kpi": "タクト達成率 歩留まり改善率",
                    "risks": "歩留まり未達 稼働変動",
                    "countermeasures": "追加調整計画 変動要因解析"
                },
                "phase_6": {
                    "activities": "量産改善承認 標準作業書更新 条件凍結",
                    "inputs": "能力検証レポート KPI達成状況",
                    "outputs": "承認改善実施書 標準作業書 改善後条件",
                    "tools": "ドキュメント管理 ワークフロー",
                    "stakeholders": "工場長(R/A) 品質保証部門長(C) 生産技術(C) 設備保全(I)",
                    "kpi": "承認リードタイム 初期安定稼働率",
                    "risks": "承認遅延 文書不備",
                    "countermeasures": "承認チェックリスト 文書レビュー"
                },
                "phase_7": {
                    "activities": "量産後KPIモニタ 不具合分析 継続改善提案 次期自働化検討",
                    "inputs": "量産KPIログ 不具合記録",
                    "outputs": "継続改善報告 次期自働化案 KPIトレンド",
                    "tools": "MES BIツール 不具合DB",
                    "stakeholders": "生産技術(R) 品質保証(R) 設備保全(C) 工場長(A)",
                    "kpi": "不良率継続削減率 稼働安定性指数",
                    "risks": "改善停滞 データ品質低下",
                    "countermeasures": "定例改善会議 データ品質監視"
                }
            }
        })

    # Fallback generic manufacturing technical role profile
    return _attach_common({
        "name": "汎用製造技術職",
        "industry": industry,
        "role": job_title,
        "core_terms": ["工程設計","品質管理","歩留まり","設備保全"],
        "secondary_terms": ["ラインバランシング","タクトタイム","OEE","SMED"],
        "optional_terms": ["自働化","治具設計","予知保全"],
        "scale_stages": ["ラボ","パイロット","量産"],
        "key_tests": ["性能試験","信頼性試験"],
        "stakeholder_roles": {
            "製造技術": "R",
            "品質保証": "C",
            "設備保全": "C",
            "工場長": "A"
        },
        "query_blocks": {
            "materials_or_products": [f"{industry} {job_title} 材料 製品 主要 要素"],
            "tools_and_equipment": [f"{industry} {job_title} 設備 ツール 測定 装置"],
            "processes": [f"{industry} {job_title} 工程 手法 技術 改善"],
            "industry_specific_kpi": [f"{industry} {job_title} KPI 指標 歩留まり タクト OEE"],
            "constraints_or_regulations": [f"{industry} 規格 法規制 ISO IEC JIS"],
            "common_failures": [f"{industry} 不具合 失敗 課題 ボトルネック"],
            "stakeholders": [f"{industry} 部門 役職 関係者"],
            "deliverables": [f"{industry} 報告書 仕様書 計画書 改善提案"]
        },
        "kpi_templates": {},
        "phase_overrides": {}
    }, "汎用製造技術職")
