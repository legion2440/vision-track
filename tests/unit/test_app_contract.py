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

    assert "_engine_for_backend" in functions
    assert "render_sidebar_stream_controls" in functions
    assert "render_detail_controls" in functions
    assert "render_stream_images" in functions
    assert "render_stream_metrics" in functions


@pytest.mark.parametrize(
    ("function_name", "required_call"),
    [
        ("render_sidebar_stream_controls", "snapshot_stream_controls"),
        ("render_detail_controls", "snapshot_stream_controls"),
        ("_engine_for_backend", "snapshot_for_rebuild"),
        ("render_stream_images", "snapshot_stream_frame"),
        ("render_stream_metrics", "snapshot_stream_metrics"),
    ],
)
def test_app_consumers_call_required_snapshot_functions(
    function_name: str,
    required_call: str,
) -> None:
    functions = _functions_by_name(_app_tree())

    assert _contains_call(functions[function_name], required_call)


def test_app_does_not_read_mutable_context_attrs_directly() -> None:
    visitor = DirectContextReadVisitor()
    visitor.visit(_app_tree())

    assert visitor.violations == []
