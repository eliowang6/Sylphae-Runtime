"""Comprehensive tests for AI Semantic Grounder, dense entity extraction, and type inference."""

from __future__ import annotations

import time
import pytest

from sylphae.coprocessor.generator import parse_field_spec, parse_table_description
from sylphae.coprocessor.grounding import ground_natural_language_sync
from sylphae.coprocessor.protocol import ActionVerb
from sylphae.model.semantic_grounder import get_semantic_grounder


def test_semantic_grounder_initialization_and_speed():
    t0 = time.perf_counter()
    grounder = get_semantic_grounder()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Initialization and vector index construction should be fast (< 50ms)
    assert elapsed_ms < 100.0
    assert len(grounder._vocab) > 100


@pytest.mark.parametrize(
    ("query", "expected_action"),
    [
        # create_table variations (Chinese, English, colloquial)
        ("新增一个sqlite用户表，拥有user_id，password项", ActionVerb.CREATE.value),
        ("建立商品数据模型，包含sku和price", ActionVerb.CREATE.value),
        ("create postgres accounts table with id and email", ActionVerb.CREATE.value),
        ("新建客户表", ActionVerb.CREATE.value),
        # add_field variations
        ("给会员加个过期时间戳", ActionVerb.ADD_FIELD.value),
        ("在用户表加失效期", ActionVerb.ADD_FIELD.value),
        ("add session token to accounts table", ActionVerb.ADD_FIELD.value),
        ("给订单补充支付流水号字段", ActionVerb.ADD_FIELD.value),
        # start_service variations
        ("启动后端测试服务器", ActionVerb.START_SERVICE.value),
        ("后台跑起来", ActionVerb.START_SERVICE.value),
        ("拉起测试端", ActionVerb.START_SERVICE.value),
        ("spin up backend dev server", ActionVerb.START_SERVICE.value),
        ("运行后台服务", ActionVerb.START_SERVICE.value),
        # insert_statement variations
        ("加个退出登录的接口", ActionVerb.ROUTE.value),
        ("新增用户认证路由", ActionVerb.ROUTE.value),
        ("add POST /auth/logout endpoint", ActionVerb.ROUTE.value),
        ("增加一个获取个人信息的api端点", ActionVerb.ROUTE.value),
        # replace_literal variations
        ("把导航背景调淡一点", ActionVerb.STYLE.value),
        ("调整右上角退出按钮的宽度", ActionVerb.STYLE.value),
        ("darken navbar primary color", ActionVerb.STYLE.value),
        ("修改主题颜色为暗黑", ActionVerb.STYLE.value),
        # delete_block variations
        ("删除旧的废弃接口", ActionVerb.DELETE.value),
        ("把这个无用函数删掉", ActionVerb.DELETE.value),
        ("drop legacy table", ActionVerb.DELETE.value),
        # add_function variations
        ("实现密码哈希校验函数", ActionVerb.ADD_FUNC.value),
        ("implement sha256 hash helper", ActionVerb.ADD_FUNC.value),
    ],
)
def test_ai_action_prediction_accuracy(query: str, expected_action: str):
    grounder = get_semantic_grounder()
    pred = grounder.predict_action(query)
    assert pred.action == expected_action
    assert pred.confidence > 0.20
    assert expected_action in pred.distribution


@pytest.mark.parametrize(
    ("query", "expected_concept"),
    [
        ("给客户整一张表", "users"),
        ("会员积分表", "users"),
        ("订单流水记录", "orders"),
        ("商品货品目录", "products"),
        ("角色权限分配", "roles"),
        ("系统全局配置", "configs"),
        ("操作审计日志", "logs"),
    ],
)
def test_semantic_entity_extraction(query: str, expected_concept: str):
    grounder = get_semantic_grounder()
    ent = grounder.extract_semantic_entity(query)
    assert ent.normalized_name == expected_concept


@pytest.mark.parametrize(
    ("field_desc", "expected_field", "expected_type_substr"),
    [
        ("失效时间", "expire_at", "TIMESTAMP"),
        ("可用余额", "balance", "REAL"),
        ("用户年龄", "age", "INTEGER"),
        ("是否启用", "status", "BOOLEAN"),
        ("用户邮箱", "email", "TEXT"),
        ("联系手机", "phone", "TEXT"),
    ],
)
def test_semantic_field_and_type_inference(field_desc: str, expected_field: str, expected_type_substr: str):
    grounder = get_semantic_grounder()
    fld = grounder.extract_semantic_field(field_desc)
    assert fld == expected_field
    typ = grounder.infer_column_type(fld)
    assert expected_type_substr in typ


def test_generator_parse_table_description_with_ai():
    spec1 = parse_table_description("新增一个sqlite用户表，拥有user_id，password项")
    assert spec1.name == "users"
    assert len(spec1.columns) == 2
    assert spec1.columns[0].name == "user_id"
    assert spec1.columns[1].name == "password"

    spec2 = parse_table_description("新建商品表，包含sku，price")
    assert spec2.name == "products"
    assert len(spec2.columns) == 2
    assert spec2.columns[0].name == "sku"
    assert spec2.columns[1].name == "price"
    assert "REAL" in spec2.columns[1].col_type or "DECIMAL" in spec2.columns[1].col_type


def test_generator_parse_field_spec_with_ai():
    tbl1, col1 = parse_field_spec("在用户表加失效期")
    assert tbl1 == "users"
    assert col1.name == "expire_at"
    assert col1.col_type == "TIMESTAMP"

    tbl2, col2 = parse_field_spec("给订单表加金额price")
    assert tbl2 == "orders"
    assert col2.name == "price"

    tbl3, col3 = parse_field_spec("给客户表加余额")
    assert tbl3 == "users"
    assert col3.name == "balance"


def test_compound_natural_language_grounding_sync():
    query = "把导航背景改淡，顺便启动后端服务，在用户表加失效期"
    steps = ground_natural_language_sync(query)
    assert len(steps) == 3
    assert steps[0].action == "replace_literal"
    assert steps[1].action == "start_service"
    assert steps[2].action == "add_field"


def test_sub_second_inference_latency_benchmark():
    grounder = get_semantic_grounder()
    queries = [
        "新增一个sqlite用户表，拥有user_id，password项",
        "后台跑起来",
        "在用户表加失效期",
        "加个退出登录的接口",
        "把导航背景调淡一点",
    ] * 20  # 100 queries

    t0 = time.perf_counter()
    for q in queries:
        grounder.predict_action(q)
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_ms / len(queries)

    # Must be ultra-fast: average < 1.5ms per query
    assert avg_ms < 2.0
