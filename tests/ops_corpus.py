"""Hand-written ``(old, new, expected_op_kind)`` corpus for the ops tests.

Every op kind in :data:`sylphae.ops.vocab.OP_KINDS` except ``no_op`` (which
``detect_op`` never emits) appears at least once, plus deliberately freeform cases
so the escape hatch is exercised too.

Each entry is ``(name, old_src, new_src, expected_kind)``.  ``expected_kind`` is
asserted exactly: a detector regression that silently reroutes an example to
``freeform_patch`` must turn the suite red, not quietly shrink coverage.
"""

from __future__ import annotations

CORPUS: list[tuple[str, str, str, str]] = [
    # ---------------------------------------------------------------- literal
    (
        "replace_literal_number",
        "TIMEOUT = 30\nRETRIES = 3\n",
        "TIMEOUT = 60\nRETRIES = 3\n",
        "replace_literal",
    ),
    (
        "replace_literal_in_call_multi_line_block",
        "SIZES = [\n    1,\n]\n",
        "SIZES = [\n    2,\n]\n",
        "replace_literal",
    ),
    (
        "replace_literal_none",
        "def f(x):\n    y = None\n    return (x, y)\n",
        "def f(x):\n    y = 0\n    return (x, y)\n",
        "replace_literal",
    ),
    # ------------------------------------------------------------ rename_symbol
    (
        "rename_symbol_local",
        "def render(node):\n    label = node.name\n    return label\n",
        "def render(item):\n    label = item.name\n    return label\n",
        "rename_symbol",
    ),
    (
        "rename_symbol_across_functions",
        "def a():\n    return helper()\n\n\ndef b():\n    return helper()\n",
        "def a():\n    return util()\n\n\ndef b():\n    return util()\n",
        "rename_symbol",
    ),
    # --------------------------------------------------------- change_condition
    (
        "change_condition_if",
        'if status == "ok":\n    handle()\n',
        'if status != "error":\n    handle()\n',
        "change_condition",
    ),
    (
        "change_condition_while",
        "while index < 10:\n    index += 1\n",
        "while index < 20:\n    index += 1\n",
        "change_condition",
    ),
    (
        "change_condition_assert",
        "assert len(items) == 1\n",
        "assert len(items) >= 1\n",
        "change_condition",
    ),
    # -------------------------------------------------------- modify_call_args
    (
        "modify_call_args_kwarg",
        'result = client.request("GET", url, timeout=30)\n',
        'result = client.request("GET", url, timeout=60)\n',
        "modify_call_args",
    ),
    (
        "modify_call_args_positional",
        "value = parse(raw, strict)\n",
        "value = parse(raw, strict, 2)\n",
        "modify_call_args",
    ),
    # ----------------------------------------------------------- change_default
    (
        "change_default_kwonly",
        "def connect(host, port=5432, timeout=30):\n    return (host, port, timeout)\n",
        "def connect(host, port=5432, timeout=60):\n    return (host, port, timeout)\n",
        "change_default",
    ),
    (
        "change_default_multiline_signature",
        "def connect(\n    host,\n    port=5432,\n):\n    return (host, port)\n",
        "def connect(\n    host,\n    port=5433,\n):\n    return (host, port)\n",
        "change_default",
    ),
    # --------------------------------------------------------------- add_import
    (
        "add_import_at_top",
        "import os\nimport sys\n\n\ndef main():\n    return os.getcwd()\n",
        "import json\nimport os\nimport sys\n\n\ndef main():\n    return os.getcwd()\n",
        "add_import",
    ),
    (
        "add_import_from_nested",
        "def main():\n    import os\n    return os.getcwd()\n",
        "def main():\n    import os\n    from pathlib import Path\n    return os.getcwd()\n",
        "add_import",
    ),
    # ------------------------------------------------------------ remove_import
    (
        "remove_import",
        "import os\nimport sys\n\n\nprint(os.name)\n",
        "import os\n\n\nprint(os.name)\n",
        "remove_import",
    ),
    # --------------------------------------------------------------- add_guard
    (
        "add_guard_return",
        "def first(items):\n    return items[0]\n",
        "def first(items):\n    if not items:\n        return None\n    return items[0]\n",
        "add_guard",
    ),
    (
        "add_guard_raise_after_docstring",
        'def parse(value):\n    """Parse a value."""\n    return int(value)\n',
        'def parse(value):\n    """Parse a value."""\n'
        "    if value is None:\n"
        '        raise ValueError("value is required")\n'
        "    return int(value)\n",
        "add_guard",
    ),
    (
        "add_guard_continue_in_loop",
        "def f(items):\n    for item in items:\n        print(item)\n",
        "def f(items):\n    for item in items:\n"
        "        if item is None:\n"
        "            continue\n"
        "        print(item)\n",
        "add_guard",
    ),
    # -------------------------------------------------------- insert_statement
    (
        "insert_statement_one",
        "def setup():\n    cfg = load()\n    return cfg\n",
        "def setup():\n    cfg = load()\n    cfg.validate()\n    return cfg\n",
        "insert_statement",
    ),
    (
        "insert_statement_two_line_if_counts_as_one_unit",
        "def f(x):\n    pass\n",
        "def f(x):\n    if x:\n        y = 1\n        z = 2\n    pass\n",
        "insert_statement",
    ),
    (
        "insert_statement_at_eof",
        "x = 1\n",
        "x = 1\ny = 2\n",
        "insert_statement",
    ),
    (
        # The inserted block carries a trailing blank line.  Whitespace is not
        # content, so this is still a complete-statement insertion.
        "insert_statement_with_surrounding_blank",
        "class A:\n    def a(self):\n        pass\n",
        "class A:\n    def a(self):\n        pass\n\n    def b(self):\n        pass\n",
        "insert_statement",
    ),
    # -------------------------------------------------------- delete_statement
    (
        "delete_statement_one",
        "def setup():\n    cfg = load()\n    cfg.validate()\n    return cfg\n",
        "def setup():\n    cfg = load()\n    return cfg\n",
        "delete_statement",
    ),
    (
        "delete_statement_at_eof",
        "x = 1\ny = 2\n",
        "x = 1\n",
        "delete_statement",
    ),
    # ------------------------------------------------------------ add_decorator
    (
        "add_decorator",
        "def cached(fn):\n    return fn\n",
        "@wraps_impl\ndef cached(fn):\n    return fn\n",
        "add_decorator",
    ),
    (
        "add_decorator_second",
        "@a\ndef f(x):\n    return x\n",
        "@a\n@b\ndef f(x):\n    return x\n",
        "add_decorator",
    ),
    # ------------------------------------------------------------- change_return
    (
        "change_return",
        "def size(items):\n    return len(items)\n",
        "def size(items):\n    return len(items) + 1\n",
        "change_return",
    ),
    (
        "change_return_literal",
        'def label():\n    return "old"\n',
        'def label():\n    return "new"\n',
        "change_return",
    ),
    # ------------------------------------------------------------ modify_string
    (
        "modify_string_message",
        'def fail():\n    raise RuntimeError("boom")\n',
        'def fail():\n    raise RuntimeError("kaboom")\n',
        "modify_string",
    ),
    (
        "modify_string_docstring",
        'def f():\n    """Old summary."""\n    return 1\n',
        'def f():\n    """New summary."""\n    return 1\n',
        "modify_string",
    ),
    # --------------------------------------------------------- wrap_try_except
    (
        "wrap_try_except",
        "def read(path):\n    data = open(path).read()\n    return data\n",
        "def read(path):\n    try:\n        data = open(path).read()\n"
        '    except OSError:\n        data = ""\n'
        "    return data\n",
        "wrap_try_except",
    ),
    (
        "wrap_try_except_with_finally",
        "def run():\n    job = start()\n    return job\n",
        "def run():\n    try:\n        job = start()\n"
        "    except OSError:\n        job = None\n"
        "    finally:\n        cleanup()\n"
        "    return job\n",
        "wrap_try_except",
    ),
    # --------------------------------------------------------------- wrap_block
    (
        "wrap_block_with",
        "def f():\n    for x in y:\n        print(x)\n",
        "def f():\n    with ctx():\n        for x in y:\n            print(x)\n",
        "wrap_block",
    ),
    (
        "wrap_block_if_module_level",
        "a = compute()\nb = transform(a)\nprint(b)\n",
        "if __name__ == '__main__':\n    a = compute()\n"
        "    b = transform(a)\n    print(b)\n",
        "wrap_block",
    ),
    # ------------------------------------------------------------- replace_line
    (
        "replace_line_whole_line",
        "x = foo(a)\n",
        "y = bar(b)\n",
        "replace_line",
    ),
    (
        "replace_line_mid_file",
        "def f():\n    setup()\n    result = run(a)\n    teardown()\n",
        "def f():\n    setup()\n    output = execute(b)\n    teardown()\n",
        "replace_line",
    ),
    # ---------------------------------------------------- modify_string_multi
    (
        "modify_string_multi_path_prefix",
        'NAMES = ["a.txt", "b.txt"]\n',
        'NAMES = ["out/a.txt", "out/b.txt"]\n',
        "modify_string_multi",
    ),
    (
        "modify_string_multi_two_messages",
        'def f(x):\n    raise ValueError("bad input", "retry")\n',
        'def f(x):\n    raise ValueError("invalid input", "please retry")\n',
        "modify_string_multi",
    ),
    # --------------------------------------------------------------- add_call_arg
    (
        "add_call_arg_multiline",
        "foo(\n    a=1,\n)\n",
        "foo(\n    a=1,\n    b=2,\n)\n",
        "add_call_arg",
    ),
    (
        "add_call_arg_setup",
        'setup(\n    name="pkg",\n)\n',
        'setup(\n    name="pkg",\n    version="1.0",\n)\n',
        "add_call_arg",
    ),
    # ------------------------------------------------------------ remove_call_arg
    (
        "remove_call_arg_multiline",
        "foo(\n    a=1,\n    b=2,\n)\n",
        "foo(\n    a=1,\n)\n",
        "remove_call_arg",
    ),
    # ------------------------------------------------------------ add_dict_entry
    (
        "add_dict_entry",
        'CONFIG = {\n    "a": 1,\n}\n',
        'CONFIG = {\n    "a": 1,\n    "b": 2,\n}\n',
        "add_dict_entry",
    ),
    (
        "add_dict_entry_typed",
        "MAPPING = {\n    'x': str,\n}\n",
        "MAPPING = {\n    'x': str,\n    'y': int,\n}\n",
        "add_dict_entry",
    ),
    # --------------------------------------------------------- remove_dict_entry
    (
        "remove_dict_entry",
        'CONFIG = {\n    "a": 1,\n    "b": 2,\n}\n',
        'CONFIG = {\n    "a": 1,\n}\n',
        "remove_dict_entry",
    ),
    # --------------------------------------------------------- add_list_element
    (
        "add_list_element",
        'PATHS = [\n    "/usr/lib",\n]\n',
        'PATHS = [\n    "/usr/lib",\n    "/usr/lib/debug",\n]\n',
        "add_list_element",
    ),
    (
        "add_list_element_set",
        "SEEN = {\n    'a',\n}\n",
        "SEEN = {\n    'a',\n    'b',\n}\n",
        "add_list_element",
    ),
    # ------------------------------------------------------ remove_list_element
    (
        "remove_list_element",
        'PATHS = [\n    "/usr/lib",\n    "/usr/lib/debug",\n]\n',
        'PATHS = [\n    "/usr/lib",\n]\n',
        "remove_list_element",
    ),
    # ------------------------------------------------------------------- reindent
    (
        "reindent_block",
        "def f():\n    x = 1\n    return x\n",
        "def f():\n        x = 1\n        return x\n",
        "reindent",
    ),
    (
        "reindent_single_line",
        "def f():\n    x = 1\n",
        "def f():\n        x = 1\n",
        "reindent",
    ),
    # ------------------------------------------------------------ freeform_patch
    (
        "freeform_restructure",
        "def f(x):\n    return x\n",
        "def f(x):\n    y = x * 2\n    return y\n",
        "freeform_patch",
    ),
    (
        "freeform_two_literals",
        "a = 1\nb = 2\n",
        "a = 3\nb = 4\n",
        "freeform_patch",
    ),
    (
        "freeform_comment_only_insert",
        "x = 1\n",
        "x = 1\n# a note\n",
        "freeform_patch",
    ),
    (
        # Two *different* targets on adjacent lines: not a consistent rename, and
        # not a single line replacement either, so it stays freeform.  (A single
        # inconsistent rename would now be a legitimate ``replace_line``.)
        "freeform_inconsistent_rename",
        "def f():\n    foo()\n    y = foo()\n    return y\n",
        "def f():\n    bar()\n    y = baz()\n    return y\n",
        "freeform_patch",
    ),
]

#: Kinds the corpus is required to exercise.
COVERED_KINDS: frozenset[str] = frozenset(entry[3] for entry in CORPUS)
