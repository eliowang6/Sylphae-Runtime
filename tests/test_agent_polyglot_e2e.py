import asyncio
import tempfile
from pathlib import Path

from sylphae.agent.effort import LowEffort
from sylphae.agent.reflex import ReflexAgent


def test_e2e_top_right_button_tsx():
    """Test '把右上角的按钮颜色加深，字体调大一些' on a React TSX component."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src" / "components").mkdir(parents=True)
        tsx_file = root / "src" / "components" / "Header.tsx"
        tsx_file.write_text("""
import React from 'react';

export const Header: React.FC = () => {
    return (
        <header className="flex justify-between items-center h-16 px-4">
            <div className="brand font-bold">App</div>
            <div className="flex items-center gap-4">
                <button className="btn bg-blue-400 text-sm px-4 py-2">
                    Action
                </button>
            </div>
        </header>
    );
};
""")

        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把右上角的按钮颜色加深，字体调大一些",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "Header.tsx" in res.target_path
        assert res.new_source is not None
        # Color darkened from bg-blue-400 to bg-blue-600
        assert "bg-blue-600" in res.new_source
        # Font enlarged from text-sm to text-base
        assert "text-base" in res.new_source


def test_e2e_top_right_button_css():
    """Test '把右上角的按钮颜色加深，字体调大一些' on a CSS file."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "styles").mkdir(parents=True)
        css_file = root / "styles" / "header.css"
        css_file.write_text("""
.header {
    display: flex;
    justify-content: space-between;
}

.btn-top-right {
    background-color: #3b82f6;
    font-size: 14px;
    padding: 8px 16px;
}
""")

        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把右上角的按钮颜色加深，字体调大一些",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "header.css" in res.target_path
        assert res.new_source is not None
        # Background color darkened from #3b82f6
        assert "#3b82f6" not in res.new_source
        # Font enlarged from 14px
        assert "14px" not in res.new_source
        assert "font-size:" in res.new_source


def test_e2e_go_timeout_polyglot():
    """Test timeout replacement on a Go source file."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pkg" / "client").mkdir(parents=True)
        go_file = root / "pkg" / "client" / "http.go"
        go_file.write_text("""package client

import (
\t"net/http"
\t"time"
)

var DefaultClient = &http.Client{
\tTimeout: 30 * time.Second,
}
""")

        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把超时时间修改为60",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "http.go" in res.target_path
        assert res.new_source is not None
        assert "60" in res.new_source


def test_e2e_nextjs_cn_tailwind():
    """Test multi-line JSX with cn(...) and Tailwind in a Next.js component."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "components").mkdir(parents=True)
        tsx_file = root / "components" / "Navbar.tsx"
        tsx_file.write_text("""import React from 'react';
import { cn } from '../utils/cn';

export function Navbar() {
  return (
    <header className="flex h-16 w-full items-center justify-between border-b px-6 bg-slate-900 text-white">
      <div className="flex items-center gap-3">
        <button
          className={cn(
            "inline-flex items-center justify-center rounded-md font-medium transition-colors",
            "px-3 py-1.5 bg-blue-500 text-sm hover:bg-blue-600 shadow"
          )}
        >
          Create Project
        </button>
      </div>
    </header>
  );
}
""")
        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把右上角的创建项目按钮颜色加深，字体调大一些",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "Navbar.tsx" in res.target_path
        assert "bg-blue-700" in res.new_source
        assert "text-base" in res.new_source


def test_e2e_vue_sfc_scoped_style():
    """Test Vue 3 Single File Component <style scoped> modification."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src" / "components").mkdir(parents=True)
        vue_file = root / "src" / "components" / "ProductCard.vue"
        vue_file.write_text("""<script setup lang="ts">
defineProps<{ title: string }>()
</script>

<template>
  <div class="product-card">
    <button class="buy-btn">Add to Cart</button>
  </div>
</template>

<style scoped>
.product-card {
  border-radius: 8px;
}
.buy-btn {
  background-color: #10b981;
  font-size: 14px;
}
</style>
""")
        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把加入购物车按钮的颜色加深，字体调大",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "ProductCard.vue" in res.target_path
        assert "font-size: 17px" in res.new_source or "font-size: 16px" in res.new_source


def test_e2e_rust_const_workers():
    """Test snake_case const replacement in Rust crate."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir(parents=True)
        rs_file = root / "src" / "config.rs"
        rs_file.write_text("""pub const DEFAULT_MAX_WORKERS: usize = 8;
pub const DEFAULT_RETRY_COUNT: u32 = 3;
""")
        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把最大工作线程数调整为16",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "config.rs" in res.target_path
        assert "DEFAULT_MAX_WORKERS: usize = 16" in res.new_source


def test_e2e_go_server_port():
    """Test port constant replacement in Go service."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "config").mkdir(parents=True)
        go_file = root / "config" / "settings.go"
        go_file.write_text("""package config

const (
	DefaultPort        = 8080
	DefaultReadTimeout = 30
)
""")
        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="把服务端口修改为9000",
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "settings.go" in res.target_path
        assert "DefaultPort        = 9000" in res.new_source

