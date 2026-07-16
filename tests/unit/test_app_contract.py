from __future__ import annotations

import ast
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[2] / "app.py"

FORBIDDEN_CONTEXT_ATTRS = {
    "source",
    "options",
    "tracker",
    "counter",
    "metrics",
    "state",
    "error",
    "actual_backend",
    "actual_device",
    "actual_provider",
    "latest_frame",
    "latest_rendered_frame",
    "latest_rendered_version",
    "runtime_generation",
}


def _app_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))


def _functions_by_name(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _contains_call(node: ast.AST, call_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name) and child.func.id == call_name:
            return True
        if isinstance(child.func, ast.Attribute) and child.func.attr == call_name:
            return True
    return False


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    return set()


def _is_method_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
    )


def _is_engine_get_call(node: ast.AST) -> bool:
    return (
        _is_method_call(node, "get")
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "engine"
    )


class DirectContextReadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.context_vars: set[str] = set()
        self.context_collections: set[str] = set()
        self.violations: list[str] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._remember_target(target, node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._remember_target(node.target, node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        added = _target_names(node.target) if self._is_context_iter(node.iter) else set()
        old_context_vars = set(self.context_vars)
        self.context_vars.update(added)
        for child in node.body:
            self.visit(child)
        self.context_vars = old_context_vars
        for child in node.orelse:
            self.visit(child)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        old_context_vars = set(self.context_vars)
        for generator in node.generators:
            self.visit(generator.iter)
            if self._is_context_iter(generator.iter):
                self.context_vars.update(_target_names(generator.target))
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.key)
        self.visit(node.value)
        self.context_vars = old_context_vars

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_CONTEXT_ATTRS and self._is_context_expr(node.value):
            self.violations.append(f"{node.attr} at line {node.lineno}")
        self.generic_visit(node)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp) -> None:
        old_context_vars = set(self.context_vars)
        for generator in node.generators:
            self.visit(generator.iter)
            if self._is_context_iter(generator.iter):
                self.context_vars.update(_target_names(generator.target))
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.elt)
        self.context_vars = old_context_vars

    def _remember_target(self, target: ast.AST, value: ast.AST) -> None:
        names = _target_names(target)
        if _is_engine_get_call(value) or (
            isinstance(value, ast.Name) and value.id in self.context_vars
        ):
            self.context_vars.update(names)
        else:
            self.context_vars.difference_update(names)
        if _is_method_call(value, "contexts") or (
            isinstance(value, ast.Name) and value.id in self.context_collections
        ):
            self.context_collections.update(names)
        else:
            self.context_collections.difference_update(names)

    def _is_context_iter(self, node: ast.AST) -> bool:
        return _is_method_call(node, "contexts") or (
            isinstance(node, ast.Name) and node.id in self.context_collections
        )

    def _is_context_expr(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id in self.context_vars
            or _is_engine_get_call(node)
        )


def test_required_app_functions_exist() -> None:
    functions = _functions_by_name(_app_tree())

    assert "_engine_for_model" in functions
    assert "render_sidebar_stream_controls" in functions
    assert "render_detail_controls" in functions
    assert "render_stream_metrics_card" in functions
    assert "render_detail_metrics" in functions
    assert "render_metrics_dashboard" in functions


def test_model_load_fallback_records_loaded_model_id() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    tree = _app_tree()
    detector_select_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Detector model"
    ]

    assert len(detector_select_calls) == 1
    assert any(
        keyword.arg == "key"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "MODEL_SELECT_KEY"
        for keyword in detector_select_calls[0].keywords
    )
    assert "sync_model_selector_before_widget(" in source
    assert "record_model_fallback(" in source
    assert "record_loaded_model(st.session_state, model_id=loaded_model_id)" in source


@pytest.mark.parametrize(
    ("function_name", "required_call"),
    [
        ("render_sidebar_stream_controls", "snapshot_stream_controls"),
        ("render_detail_controls", "snapshot_stream_controls"),
        ("_engine_for_model", "snapshot_for_rebuild"),
        ("render_stream_metrics_card", "snapshot_stream_metrics"),
        ("render_detail_metrics", "snapshot_stream_metrics"),
    ],
)
def test_app_consumers_call_required_snapshot_functions(
    function_name: str,
    required_call: str,
) -> None:
    functions = _functions_by_name(_app_tree())

    assert _contains_call(functions[function_name], required_call)


def test_app_has_no_live_frame_streamlit_polling_path() -> None:
    tree = _app_tree()
    forbidden_calls = {
        "snapshot_stream_frame",
        "update_stream_frame_cache",
        "image",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for keyword in decorator.keywords:
                assert not (
                    keyword.arg == "run_every"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == 0.01
                )


def test_metrics_use_one_auto_refresh_fragment() -> None:
    functions = _functions_by_name(_app_tree())
    auto_refresh_functions = []

    for name, function in functions.items():
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not (
                isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "fragment"
            ):
                continue
            if any(
                keyword.arg == "run_every"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0.25
                for keyword in decorator.keywords
            ):
                auto_refresh_functions.append(name)

    assert auto_refresh_functions == ["render_metrics_dashboard"]
    assert _contains_call(
        functions["render_metrics_dashboard"],
        "render_stream_metrics_card",
    )
    assert _contains_call(
        functions["render_metrics_dashboard"],
        "render_detail_metrics",
    )


def test_app_renders_and_registers_websocket_preview() -> None:
    tree = _app_tree()

    assert _contains_call(tree, "build_preview_component_html")
    assert _contains_call(tree, "html")
    assert _contains_call(tree, "replace_session")


def test_app_exposes_manual_server_webcam_discovery_and_add() -> None:
    tree = _app_tree()
    parent_by_child = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    discovery_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "discover_webcams"
    ]

    assert len(discovery_calls) == 1
    assert _contains_call(tree, "webcam")
    assert "Refresh cameras" in APP_PATH.read_text(encoding="utf-8")
    assert "Add camera" in APP_PATH.read_text(encoding="utf-8")

    ancestor = parent_by_child.get(discovery_calls[0])
    while ancestor is not None and not isinstance(ancestor, ast.If):
        ancestor = parent_by_child.get(ancestor)

    assert isinstance(ancestor, ast.If)
    refresh_calls = [
        node
        for node in ast.walk(ancestor.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Refresh cameras"
    ]
    assert len(refresh_calls) == 1


@pytest.mark.parametrize(
    "function_name",
    ["render_stream_metrics_card", "render_detail_metrics"],
)
def test_metrics_fragments_create_only_their_own_elements(function_name: str) -> None:
    function = _functions_by_name(_app_tree())[function_name]
    referenced_names = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }

    assert not referenced_names.intersection(
        {
            "stream_placeholders",
            "image_placeholder",
            "waiting_placeholder",
            "detail_metric_placeholders",
            "detail_runtime_placeholder",
        }
    )
    assert not _contains_call(function, "empty")


def test_app_does_not_read_mutable_context_attrs_directly() -> None:
    visitor = DirectContextReadVisitor()
    visitor.visit(_app_tree())

    assert visitor.violations == []
