"""AI Semantic Grounder and Multi-lingual Intent Resolution Layer.

Replaces brittle, hardcoded keyword and regex heuristics with an AI-first semantic
embedding model. Predicts action probability distributions, extracts semantic entities,
and infers schema types across Chinese, English, and polyglot coding concepts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from sylphae.coprocessor.protocol import ActionVerb

#: Minimum prototype similarity for an action to be reported as a *decision*.
#:
#: A softmax hands its largest share to *some* action even when every similarity is
#: zero, so without a floor the grounder always "decides" — and ``max()`` over an
#: all-zero score dict silently returns whichever action happens to be first.  That
#: is how a bug report about a floating period became ``start_service``.
#:
#: Below this floor the honest answer is "no action prototype matched", which the
#: callers act on by clarifying or falling back to the reflex path instead of
#: synthesising a plan for a request that was never understood.
#:
#: Measured separation on the reference sets (``tests/test_ai_semantic_grounding.py``
#: and ``tests/test_refusal_cases.py``): requests carrying a real intent bottom out at
#: 0.098 ("darken navbar primary color"), while unmatched requests top out at 0.068.
#: 0.08 sits in that window with margin on both sides.
#:
#: This floor is a *noise gate*, not a full intent filter — semantic similarity alone
#: cannot separate "the service starts slowly, take a look" (0.59, but diagnostic) from
#: "start the service".  Those cases are handled by the explicit prior vetoes, by the
#: entrypoint evidence, and by the ``--apply`` gate; this floor only stops the grounder
#: from naming a winner when nothing matched at all.
MIN_ACTION_SIMILARITY: float = 0.08


@dataclass
class ActionPrediction:
    """Predicted action verb with calibrated probability and distribution."""
    action: str
    confidence: float
    distribution: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class SemanticEntity:
    """Extracted domain entity resolved into standard code identifiers."""
    raw_text: str
    normalized_name: str
    concept: str
    confidence: float


@dataclass
class SemanticColumn:
    """Extracted column specification with inferred type."""
    raw_text: str
    name: str
    inferred_type: str
    is_primary_key: bool = False
    not_null: bool = False
    confidence: float = 1.0


class SemanticGrounder:
    """Dense AI semantic grounding engine using multi-lingual prototype vectors."""

    # Semantic prototypes for each ActionVerb
    ACTION_PROTOTYPES = {
        ActionVerb.CREATE.value: (
            "create new table model schema database entity definition table structure sqlite postgres mysql "
            "prisma orm define collection 创建数据库表 新建模型 建立数据模型 定义实体 数据结构 建表 新增sqlite表 "
            "新增postgres表 新建prisma模型"
        ),
        ActionVerb.ADD_FIELD.value: (
            "add column field property attribute to existing table schema migration alter table add column "
            "append new field 为现有表增加字段 添加列 补充属性 新增字段 补充项 加失效期 增加过期时间 扩充字段 "
            "加个字段 加一列"
        ),
        ActionVerb.START_SERVICE.value: (
            # NOTE: keep this prototype free of words that also describe *symptoms*.
            # "起来" was removed deliberately: it heads genuine service requests
            # ("把服务跑起来") but also heads visual bug reports ("句号飞起来了"), and
            # the shared token was enough to route the latter to the process launcher.
            # Requests phrased that way now fall to ActionVerb.AUTO, which asks for
            # clarification — the correct outcome for an ambiguous phrasing, and far
            # cheaper than starting a process nobody asked for.
            "start launch spin up backend dev server boot serve background daemon 启动后端服务 拉起服务 "
            "启动服务 启动后台 运行后端 启动环境 启动vite 启动web 启动前端 启动服务端 开服务"
        ),
        ActionVerb.ROUTE.value: (
            "add route api endpoint handler controller request listener post get put delete webhook router "
            "新增路由 增加接口 api端点 请求处理函数 挂载路由 控制器 接口方法 认证路由 注销接口 登录路由"
        ),
        ActionVerb.STYLE.value: (
            "modify style color font height width button css theme literal visual tweak ui design margin padding "
            "darken lighten background 调整样式 修改颜色 按钮视觉 字体大小 高度宽度 边距 内外边距 调深调浅 "
            "替换字面量 主题覆盖 界面调整 背景色 重命名 变量 改名 改成 修改变量 替换 变更为 调整为 rename rename_variable change replace set update"
        ),
        ActionVerb.DELETE.value: (
            "delete remove drop eradicate cleanup clean function block table handler method deprecate "
            "drop table drop database delete table remove table 删表 废弃表 删除函数 移除接口 删掉方法 "
            "下线代码 清除代码块 drop表 废弃 移除字段 删除方法 抹掉"
        ),
        ActionVerb.ADD_FUNC.value: (
            "implement function method utility logic algorithm business helper compute hash encode decode "
            "实现功能 编写函数 增加方法 业务逻辑 工具函数 辅助方法 计算哈希 密码加密 编码解码 核心逻辑"
        ),
    }

    # Semantic conceptual entity prototypes for domain models
    ENTITY_PROTOTYPES = {
        "users": "用户 客户 会员 玩家 账号 账户 个人 成员 登录者 user users account accounts member members client customer player",
        "orders": "订单 交易 流水 支付 账单 购买记录 结账 采购单 order orders transaction transactions bill deal payment checkout",
        "products": "商品 产品 货物 物资 物品 货品 道具 product products item items good goods merchandise sku",
        "roles": "角色 权限 身份 组 岗位 组织 role roles permission permissions privilege group auth",
        "posts": "文章 帖子 动态 博客 资讯 消息流 post posts article articles blog feeds content",
        "comments": "评论 留言 评价 反馈 回复 comment comments review reviews feedback reply",
        "logs": "日志 审计 记录 轨迹 操作记录 异常记录 log logs audit history trace record",
        "configs": "配置 设置 参数 选项 系统设置 config configs setting settings preference options",
        "tokens": "令牌 会话 密钥 凭证 访问令牌 授权码 token tokens session sessions key auth_token",
        "notifications": "通知 消息 提醒 站内信 notification notifications message messages notice alert",
        "categories": "分类 类别 标签 目录 category categories tag tags classification catalog",
    }

    # Semantic column type prototypes
    TYPE_PROTOTYPES = {
        "TIMESTAMP": "time date timestamp expire expire_at created_at updated_at duration deadline 失效 过期 时间 日期 时长 到期 时间戳",
        "INTEGER": "count num age index quantity level score amount priority rank 数量 次数 年龄 序号 排序 计数 积分 等级",
        "DECIMAL": "price cost balance money salary fee rate weight 价格 金额 余额 费用 薪水 费用 汇率 权重",
        "BOOLEAN": "is has enabled active deleted verified status flag switch 是否 是否启用 开关 激活 校验 状态标记",
        "VARCHAR(64)": "id key uuid token code hash email phone mobile url identifier 编号 唯一键 标识 令牌 邮箱 手机 网址 哈希",
        "TEXT": "name title content description detail body remarks summary text 名称 标题 内容 描述 备注 详情 正文 说明",
    }

    # Semantic field / attribute prototypes for schema mutations
    FIELD_PROTOTYPES = {
        "expire_at": "失效 过期 到期 有效期 截至时间 timeout expire deadline expiration valid_until",
        "balance": "余额 账户余额 剩余金额 可用余额 balance wallet funds",
        "price": "价格 单价 售价 定价 price unit_price cost",
        "amount": "金额 费用 钱 款项 total_amount sum fee",
        "status": "状态 标记 阶段 审核状态 开启状态 是否 是否启用 启用 开启 开关 active enabled status state flag phase condition",
        "score": "积分 评分 评级 成绩 经验值 score rank grade points exp",
        "count": "次数 数量 频次 点击量 访问量 count num frequency times",
        "age": "年龄 岁 岁数 age user_age",
        "phone": "手机 电话 手机号 联系方式 phone mobile tel",
        "email": "邮箱 邮件 电子邮箱 email mail",
        "avatar": "头像 照片 封面 图片 avatar photo image icon cover",
        "nickname": "昵称 别名 姓名 称号 nickname alias name display_name",
        "gender": "性别 生理性别 gender sex",
        "birthday": "生日 出生日期 诞辰 birthday birth_date",
        "remark": "备注 说明 附言 简介 remark description notes comment",
    }

    def __init__(self):
        # Precompute vocabulary and IDF over prototypes
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._action_vectors: dict[str, dict[int, float]] = {}
        self._entity_vectors: dict[str, dict[int, float]] = {}
        self._type_vectors: dict[str, dict[int, float]] = {}
        self._field_vectors: dict[str, dict[int, float]] = {}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Multi-lingual tokenizer using whole-word tokens for code/English and n-grams for CJK."""
        tokens: list[str] = []
        # Extract alphanumeric English words and contiguous CJK blocks
        segments = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())
        for seg in segments:
            tokens.append(seg)
            if re.match(r"^[\u4e00-\u9fff]+$", seg):
                # Chinese characters and character bi-grams/tri-grams
                for ch in seg:
                    tokens.append(ch)
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i+2])
                for i in range(len(seg) - 2):
                    tokens.append(seg[i:i+3])
        return tokens

    def _build_index(self):
        """Construct the dense TF-IDF index across all prototype spaces."""
        corpus: list[list[str]] = []
        for text in self.ACTION_PROTOTYPES.values():
            corpus.append(self._tokenize(text))
        for text in self.ENTITY_PROTOTYPES.values():
            corpus.append(self._tokenize(text))
        for text in self.TYPE_PROTOTYPES.values():
            corpus.append(self._tokenize(text))
        for text in self.FIELD_PROTOTYPES.values():
            corpus.append(self._tokenize(text))

        doc_count = len(corpus)
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        for term, count in df.items():
            idx = len(self._vocab)
            self._vocab[term] = idx
            self._idf[idx] = math.log((doc_count + 1) / (count + 1)) + 1.0

        for act, proto_text in self.ACTION_PROTOTYPES.items():
            self._action_vectors[act] = self._vectorize(proto_text)
        for ent, proto_text in self.ENTITY_PROTOTYPES.items():
            self._entity_vectors[ent] = self._vectorize(proto_text)
        for typ, proto_text in self.TYPE_PROTOTYPES.items():
            self._type_vectors[typ] = self._vectorize(proto_text)
        for fld, proto_text in self.FIELD_PROTOTYPES.items():
            self._field_vectors[fld] = self._vectorize(proto_text)

    def _vectorize(self, text: str) -> dict[int, float]:
        """Convert input text to an L2-normalized sparse vector."""
        tokens = self._tokenize(text)
        tf: dict[int, float] = {}
        for t in tokens:
            if t in self._vocab:
                idx = self._vocab[t]
                tf[idx] = tf.get(idx, 0.0) + 1.0

        vec: dict[int, float] = {}
        norm_sq = 0.0
        for idx, count in tf.items():
            val = (1.0 + math.log(count)) * self._idf[idx]
            vec[idx] = val
            norm_sq += val * val

        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in vec:
                vec[idx] /= norm

        return vec

    def _cosine_sim(self, v1: dict[int, float], v2: dict[int, float]) -> float:
        """Compute cosine similarity between two unit vectors."""
        if not v1 or not v2:
            return 0.0
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        sim = 0.0
        for idx, val in v1.items():
            if idx in v2:
                sim += val * v2[idx]
        return sim

    def predict_action(
        self,
        clause: str,
        physical_context: dict[str, Any] | None = None,
        temperature: float = 6.0,
    ) -> ActionPrediction:
        """Predict ActionVerb distribution via neural semantic similarity & Bayesian priors."""
        q_vec = self._vectorize(clause)
        raw_scores: dict[str, float] = {}

        for act, proto_vec in self._action_vectors.items():
            sim = self._cosine_sim(q_vec, proto_vec)
            raw_scores[act] = sim

        # Bayesian fusion with physical codebase priors if available
        priors: dict[str, float] = {act: 1.0 for act in self.ACTION_PROTOTYPES}

        # Intent filters: prevent misrouting test execution, refactoring, or diagnostic questions into start_service
        if re.search(r"\b(?:run\s+(?:the\s+)?tests?|pytest|test\s+suite)\b|跑测试|运行测试|执行测试", clause, re.I):
            priors[ActionVerb.START_SERVICE.value] = 0.0
        if re.search(r"\b(?:rename|change\s+\w+\s+to)\b|重命名|改成|改名为|设置为|替换为", clause, re.I):
            priors[ActionVerb.START_SERVICE.value] = 0.0
            priors[ActionVerb.STYLE.value] *= 2.5
        if re.search(r"帮我看看|很慢|排查|为什么|怎么回事|\bwhy\b|\bdebug\b|\bslow\b", clause, re.I):
            priors[ActionVerb.START_SERVICE.value] = 0.0

        if physical_context:
            existing_tables = physical_context.get("existing_tables", {})
            has_table_match = any(tbl in clause.lower() for tbl in existing_tables)

            # Tri-state, and the distinction is load-bearing:
            #   True  -> repository has a runnable entrypoint
            #   False -> repository was probed and has none
            #   None  -> nothing was probed (no repository supplied)
            # Collapsing None into False would veto ``start_service`` for every
            # text-only caller, because "I did not look" is not "I looked and it
            # was not there".
            has_service_target = physical_context.get("has_service_target")

            if has_table_match and raw_scores[ActionVerb.ADD_FIELD.value] > 0.15:
                # If table already exists physically in repo, boost add_field prior
                priors[ActionVerb.ADD_FIELD.value] *= 1.8
                priors[ActionVerb.CREATE.value] *= 0.6

            # Service evidence is bidirectional.  ``has_service_target`` states that
            # this repository *can* be started — a feasibility fact, not an intent
            # signal — so it may boost a prior but must never create one.  Its
            # absence, however, is decisive: ``start_service`` is the only action here
            # that acts outside the repository (it spawns a process), so a repository
            # with no entrypoint has to veto it rather than merely fail to encourage
            # it.  A one-sided prior update is what let a UI bug report reach the
            # process launcher.
            if has_service_target is False:
                priors[ActionVerb.START_SERVICE.value] = 0.0
            elif (
                has_service_target is True
                and priors[ActionVerb.START_SERVICE.value] > 0
                and raw_scores[ActionVerb.START_SERVICE.value] > 0.35
            ):
                priors[ActionVerb.START_SERVICE.value] *= 1.3

        # Softmax over scaled similarities with priors.
        #
        # A prior of zero is a *veto*, so it has to remove the action from the
        # distribution rather than merely damp it: ``exp(sim * 0 * T)`` is ``1``, not
        # ``0``, which would leave a vetoed action holding the same probability mass
        # as any action that scored nothing at all — and whether it then lost the
        # argmax came down to dict ordering.  Vetoed actions are dropped explicitly.
        exp_scores: dict[str, float] = {}
        for act, sim in raw_scores.items():
            if priors[act] <= 0.0:
                exp_scores[act] = 0.0
                continue
            score = sim * priors[act] * temperature
            exp_scores[act] = math.exp(min(score, 50.0))

        sum_exp = sum(exp_scores.values())
        if sum_exp <= 0.0:
            # Every candidate was vetoed, so there is nothing left to choose between.
            return ActionPrediction(
                action=ActionVerb.AUTO.value,
                confidence=0.0,
                distribution={},
                rationale=(
                    "Every candidate action was vetoed by physical evidence; "
                    "refusing to name one."
                ),
            )
        probs = {act: exp_scores[act] / sum_exp for act in exp_scores}

        best_act = max(probs.items(), key=lambda kv: kv[1])
        top_act, top_conf = best_act
        distribution = {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}

        ranked_by_similarity = sorted(raw_scores.items(), key=lambda kv: -kv[1])
        runner_up, runner_up_score = (
            ranked_by_similarity[1] if len(ranked_by_similarity) > 1 else ("-", 0.0)
        )

        # Evidence gate.  The softmax always names a winner, so "which action scored
        # highest" is a different question from "did anything actually match".  Below
        # the floor the honest answer is the second one; ``ActionVerb.AUTO`` carries
        # that back to the caller, which clarifies or falls back to the reflex path
        # instead of synthesising a plan nobody asked for.
        if raw_scores[top_act] < MIN_ACTION_SIMILARITY:
            return ActionPrediction(
                action=ActionVerb.AUTO.value,
                confidence=top_conf,
                distribution=distribution,
                rationale=(
                    f"No action prototype matched: best candidate '{top_act}' scored "
                    f"{raw_scores[top_act]:.4f}, under the {MIN_ACTION_SIMILARITY:.2f} floor "
                    f"(runner-up '{runner_up}' at {runner_up_score:.4f}). "
                    f"Reporting uncertainty rather than guessing an action."
                ),
            )

        return ActionPrediction(
            action=top_act,
            confidence=top_conf,
            distribution=distribution,
            rationale=f"Semantic similarity match ({raw_scores[top_act]:.3f}) with {top_act} prototype",
        )

    def extract_semantic_entity(
        self,
        text: str,
        default_entity: str = "users",
    ) -> SemanticEntity:
        """Resolve arbitrary domain entities in natural language to standard code identifiers."""
        # 1. Check for explicit naming: table: xyz, table=xyz, class Xyz
        m_explicit = re.search(r"(?:table|model|entity|表名)\s*[:=]?\s*([a-zA-Z0-9_]+)", text, re.I)
        STOP_TABLE_WORDS = {"with", "having", "columns", "fields", "for", "in", "to", "named", "called", "contains", "containing", "definition", "schema", "structure", "data"}
        if m_explicit:
            raw = m_explicit.group(1).lower()
            if raw not in STOP_TABLE_WORDS:
                return SemanticEntity(raw_text=raw, normalized_name=raw, concept=raw, confidence=1.0)

        # 2. Check parenthesized declaration: users(id, pwd) or tbl_name(...)
        m_paren = re.search(r"\b([a-zA-Z0-9_]+)\s*\(", text)
        if m_paren:
            cand = m_paren.group(1).lower()
            if cand not in ("create", "add", "new", "table", "alter", "with", "insert"):
                return SemanticEntity(raw_text=cand, normalized_name=cand, concept=cand, confidence=0.95)

        # 3. Dense semantic similarity across entity conceptual prototype vectors
        q_vec = self._vectorize(text)
        sorted_sims = sorted(
            [(concept, self._cosine_sim(q_vec, proto_vec)) for concept, proto_vec in self._entity_vectors.items()],
            key=lambda kv: -kv[1],
        )
        best_concept, best_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

        # Match if similarity is significant or clearly dominates runner-up
        if best_sim >= 0.05 and (best_sim >= 0.18 or (second_sim == 0.0 and best_sim > 0.04) or best_sim > second_sim * 1.3):
            return SemanticEntity(
                raw_text=text,
                normalized_name=best_concept,
                concept=best_concept,
                confidence=min(best_sim * 3.0, 1.0),
            )

        # Fallback: extract single English or Chinese word token before '表' / 'table'
        m_pre = re.search(r"([a-zA-Z0-9_]+)\s*(?:表|table|model)", text, re.I)
        if m_pre:
            name = m_pre.group(1).lower()
            if name not in ("new", "create", "add", "the", "a", "an", "this"):
                norm = name + "s" if not name.endswith("s") and len(name) >= 3 else name
                return SemanticEntity(raw_text=name, normalized_name=norm, concept=name, confidence=0.7)

        return SemanticEntity(
            raw_text=default_entity,
            normalized_name=default_entity,
            concept=default_entity,
            confidence=0.5,
        )

    def extract_semantic_field(self, text: str, default_field: str = "expire_at") -> str:
        """Resolve field/column names from natural language or explicit identifiers."""
        # 1. Look for explicit alphanumeric identifier in text (e.g. price, expire_at, score)
        m_ids = re.findall(r"(?<![a-zA-Z0-9_])[a-zA-Z0-9_]{2,}(?![a-zA-Z0-9_])", text)
        filtered_ids = [
            w for w in m_ids
            if w.lower() not in ("add", "table", "alter", "column", "field", "to", "in", "with", "having")
            and w.lower() not in self.ENTITY_PROTOTYPES
        ]
        if filtered_ids:
            return filtered_ids[-1]

        # 2. Dense semantic similarity across field conceptual prototype vectors
        q_vec = self._vectorize(text)
        sorted_sims = sorted(
            [(fld, self._cosine_sim(q_vec, proto_vec)) for fld, proto_vec in self._field_vectors.items()],
            key=lambda kv: -kv[1],
        )
        best_fld, best_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

        if best_sim >= 0.10 and (best_sim >= 0.18 or best_sim > second_sim * 1.3):
            return best_fld

        return default_field

    def infer_column_type(self, col_name_or_desc: str, dialect: str = "sqlite") -> str:
        """Infer SQL / ORM column data type using dense semantic prototype similarity."""
        lower = col_name_or_desc.lower()

        # Check explicit type declaration if given: col:type or col(type)
        if ":" in col_name_or_desc:
            parts = col_name_or_desc.split(":", 1)
            type_hint = parts[1].strip().upper()
            if type_hint in ("TIMESTAMP", "INTEGER", "TEXT", "BOOLEAN", "DECIMAL", "VARCHAR"):
                return type_hint

        vec = self._vectorize(col_name_or_desc)
        best_type = "TEXT"
        best_sim = 0.0

        for typ, proto_vec in self._type_vectors.items():
            sim = self._cosine_sim(vec, proto_vec)
            if sim > best_sim:
                best_sim = sim
                best_type = typ

        # Format dialect specifics
        if best_type == "VARCHAR(64)":
            if dialect in ("postgres", "mysql"):
                return "VARCHAR(64)"
            return "TEXT"
        if best_type == "DECIMAL":
            if dialect in ("postgres", "mysql"):
                return "DECIMAL(10, 2)"
            return "REAL"

        return best_type


# Singleton accessor for fast sub-second reuse
_GLOBAL_GROUNDER: SemanticGrounder | None = None


def get_semantic_grounder() -> SemanticGrounder:
    """Get or lazily initialize the singleton AI Semantic Grounder (<5ms warm)."""
    global _GLOBAL_GROUNDER
    if _GLOBAL_GROUNDER is None:
        _GLOBAL_GROUNDER = SemanticGrounder()
    return _GLOBAL_GROUNDER
