"""Unified diff parsing, in-memory patch application, AST semantic diffing,
dict-key diffing, identity/equality equivalence detection, SQL fixture diffing,
and fast symlink-shadow overlay management (V3).
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    is_binary: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    old_changed_lines: set[int] = field(default_factory=set)
    old_insert_anchors: set[int] = field(default_factory=set)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


def _strip_prefix(p: str) -> str | None:
    p = p.strip()
    if p == "/dev/null":
        return None
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def parse_patch(text: str) -> list[FilePatch]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    patches: list[FilePatch] = []
    cur: FilePatch | None = None

    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            if cur is not None:
                patches.append(cur)
            cur = FilePatch(old_path=None, new_path=None)
            parts = line.split(" ")
            if len(parts) >= 4:
                cur.old_path = _strip_prefix(parts[2])
                cur.new_path = _strip_prefix(parts[3])
            i += 1
            continue

        if cur is None and line.startswith("--- "):
            cur = FilePatch(old_path=None, new_path=None)

        if cur is not None:
            if line.startswith("new file mode"):
                cur.is_new = True
                i += 1
                continue
            if line.startswith("deleted file mode"):
                cur.is_deleted = True
                i += 1
                continue
            if line.startswith("rename from "):
                cur.is_rename = True
                cur.old_path = line[len("rename from "):].strip()
                i += 1
                continue
            if line.startswith("rename to "):
                cur.is_rename = True
                cur.new_path = line[len("rename to "):].strip()
                i += 1
                continue
            if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
                cur.is_binary = True
                i += 1
                continue
            if line.startswith("--- "):
                op = _strip_prefix(line[4:].split("\t")[0])
                cur.old_path = op
                if op is None:
                    cur.is_new = True
                i += 1
                continue
            if line.startswith("+++ "):
                np = _strip_prefix(line[4:].split("\t")[0])
                cur.new_path = np
                if np is None:
                    cur.is_deleted = True
                i += 1
                continue

            m = HUNK_RE.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                i += 1
                hlines: list[str] = []
                old_ln = old_start
                while i < n:
                    hl = lines[i]
                    if hl.startswith("diff --git ") or HUNK_RE.match(hl):
                        break
                    if hl.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
                        break
                    if hl.startswith(" "):
                        hlines.append(hl)
                        old_ln += 1
                    elif hl.startswith("-"):
                        hlines.append(hl)
                        cur.old_changed_lines.add(old_ln)
                        old_ln += 1
                    elif hl.startswith("+"):
                        hlines.append(hl)
                        if old_ln > 1:
                            cur.old_insert_anchors.add(old_ln - 1)
                        cur.old_insert_anchors.add(old_ln)
                    elif hl.startswith("\\"):
                        hlines.append(hl)
                    elif hl == "":
                        hlines.append(" ")
                        old_ln += 1
                    else:
                        break
                    i += 1
                cur.hunks.append(Hunk(old_start, old_count, new_start, new_count, hlines))
                continue
        i += 1

    if cur is not None:
        patches.append(cur)
    return patches


def apply_file_patch(orig_text: str | None, fp: FilePatch) -> tuple[bool, str | None]:
    if fp.is_binary:
        return False, None
    if fp.is_deleted:
        return True, None
    if fp.is_new:
        out_lines = []
        no_nl = False
        for h in fp.hunks:
            for hl in h.lines:
                if hl.startswith("+"):
                    out_lines.append(hl[1:])
                elif hl.startswith("\\"):
                    no_nl = True
        text = "\n".join(out_lines)
        if out_lines and not no_nl:
            text += "\n"
        return True, text

    if orig_text is None:
        return False, None
    if not fp.hunks:
        return True, orig_text

    orig_ends_nl = orig_text.endswith("\n")
    orig_lines = orig_text.splitlines()
    out: list[str] = []
    pos = 0
    final_ends_nl = orig_ends_nl

    for h in fp.hunks:
        target_pos = max(0, h.old_start - 1) if h.old_start > 0 else 0
        expected_old = [hl[1:] for hl in h.lines if hl.startswith((" ", "-"))]
        if orig_lines[target_pos:target_pos + len(expected_old)] != expected_old:
            found = None
            for delta in range(-5, 6):
                cand = target_pos + delta
                if 0 <= cand and orig_lines[cand:cand + len(expected_old)] == expected_old:
                    found = cand
                    break
            if found is None:
                return False, None
            target_pos = found

        if target_pos < pos:
            return False, None
        out.extend(orig_lines[pos:target_pos])
        pos = target_pos

        for j, hl in enumerate(h.lines):
            if hl.startswith(" "):
                out.append(hl[1:])
                pos += 1
            elif hl.startswith("-"):
                pos += 1
            elif hl.startswith("+"):
                out.append(hl[1:])
            elif hl.startswith("\\"):
                if j > 0 and h.lines[j - 1].startswith("+"):
                    final_ends_nl = False
                elif j > 0 and h.lines[j - 1].startswith("-"):
                    final_ends_nl = True

    out.extend(orig_lines[pos:])
    if not out:
        return True, ""
    res = "\n".join(out)
    if final_ends_nl:
        res += "\n"
    return True, res


# ---------------------------------------------------------------------------
# AST Semantic Diff Analysis for Python files (V3)
# ---------------------------------------------------------------------------
@dataclass
class ClassBodyEdit:
    cls_qual: str
    attr_name: str | None
    is_dict_assign: bool = False
    # If key set is unchanged and ONLY values of known keys changed, list key_reprs (e.g. "S:u")
    dict_value_keys: tuple[str, ...] | None = None


@dataclass
class IdentityOpEdit:
    """Represents a file whose ENTIRE AST diff is swapping `is`<->`==` or `is not`<->`!=`."""
    rel_path: str
    enclosing_class: str | None
    kind: str  # "none" or "enum"
    self_attr: str | None  # e.g. "_table" or "normalization_strategy"
    type_hint_name: str | None  # e.g. "Table" for `self._table: Table | None`
    enum_cls_name: str | None  # e.g. "NormalizationStrategy"
    all_class_assigns_safe: bool = False


@dataclass
class PySemanticDiff:
    syntax_error: bool = False
    is_inert: bool = False
    func_lines: set[int] = field(default_factory=set)
    class_body_edits: list[ClassBodyEdit] = field(default_factory=list)
    module_level_symbols: set[str] = field(default_factory=set)
    has_import_edit: bool = False
    has_unknown_top_edit: bool = False
    only_test_methods: list[tuple[str, str]] | None = None
    identity_op_edit: IdentityOpEdit | None = None


def _coding_cookie(src: str) -> str | None:
    for line in src.splitlines()[:2]:
        m = CODING_RE.match(line)
        if m:
            return m.group(1).lower()
    return None


def _ast_dict_key_repr(node: ast.AST | None) -> str | None:
    """Convert an AST dict key into the exact key_repr format used by TrackedDict."""
    if node is None:
        return None  # **spread
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return f"S:{node.value}"
    if isinstance(node, ast.Attribute):
        # e.g., TokenType.SHOW -> E:TokenType.SHOW, exp.Add -> T:Add
        if isinstance(node.value, ast.Name):
            if node.value.id == "exp":
                return f"T:{node.attr}"
            if node.value.id == "TokenType":
                return f"E:TokenType.{node.attr}"
    if isinstance(node, ast.Name):
        return f"T:{node.id}"
    return None


def _diff_ast_dicts(old_val: ast.AST | None, new_val: ast.AST | None) -> tuple[bool, tuple[str, ...] | None]:
    """Compare two AST values. If both are ast.Dict with IDENTICAL key structure and
    only values for specific representable keys changed, return (True, tuple_of_key_reprs).
    Otherwise return (is_dict, None)."""
    if not (isinstance(old_val, ast.Dict) and isinstance(new_val, ast.Dict)):
        return False, None
    if len(old_val.keys) != len(new_val.keys):
        return True, None
    changed_keys: list[str] = []
    for ok, ov, nk, nv in zip(old_val.keys, old_val.values, new_val.keys, new_val.values):
        if ok is None or nk is None:
            # **spread entry must be identical
            if ok is not None or nk is not None:
                return True, None
            if ast.dump(ov, include_attributes=False) != ast.dump(nv, include_attributes=False):
                return True, None
            continue
        if ast.dump(ok, include_attributes=False) != ast.dump(nk, include_attributes=False):
            return True, None
        if ast.dump(ov, include_attributes=False) != ast.dump(nv, include_attributes=False):
            kr = _ast_dict_key_repr(ok)
            if kr is None:
                return True, None
            changed_keys.append(kr)
    if not changed_keys:
        return True, None
    return True, tuple(sorted(set(changed_keys)))


def _find_class_attr_stmt(tree: ast.AST, cls_qual: str, attr_name: str) -> ast.AST | None:
    """Locate the assigned value AST node for Class.attr_name in tree."""
    parts = cls_qual.split(".")
    cur_body = getattr(tree, "body", [])
    cls_node = None
    for p in parts:
        found = None
        for stmt in cur_body:
            if isinstance(stmt, ast.ClassDef) and stmt.name == p:
                found = stmt
                break
        if found is None:
            return None
        cls_node = found
        cur_body = found.body
    if cls_node is None:
        return None
    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id == attr_name:
                return stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == attr_name:
                return stmt.value
    return None


def _extract_non_none_type_name(ann: ast.AST | None) -> str | None:
    """From `Table | None` or `t.Optional[Table]`, extract 'Table'."""
    if ann is None:
        return None
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        parts = [ann.left, ann.right]
        non_none = [p for p in parts if not (isinstance(p, ast.Constant) and p.value is None)]
        if len(non_none) == 1 and isinstance(non_none[0], ast.Name):
            return non_none[0].id
    if isinstance(ann, ast.Subscript):
        val = ann.value
        name = val.id if isinstance(val, ast.Name) else (val.attr if isinstance(val, ast.Attribute) else None)
        if name == "Optional" and isinstance(ann.slice, ast.Name):
            return ann.slice.id
    if isinstance(ann, ast.Name):
        return ann.id
    return None


def _detect_pure_identity_op_edit(rel_path: str, old_ast: ast.AST, new_ast: ast.AST) -> IdentityOpEdit | None:
    """Soundly check if the ENTIRE AST diff between old_ast and new_ast is swapping a single
    `ast.Is` <-> `ast.Eq` or `ast.IsNot` <-> `ast.NotEq` comparison."""
    diffs: list[tuple[ast.Compare, ast.Compare, ast.ClassDef | None]] = []

    def walk_pair(a: ast.AST, b: ast.AST, cur_cls: ast.ClassDef | None) -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.ClassDef):
            cur_cls = a
        if isinstance(a, ast.Compare) and len(a.ops) == 1 and len(b.ops) == 1:
            op_a, op_b = type(a.ops[0]), type(b.ops[0])
            if op_a is not op_b:
                swap_ok = (
                    {op_a, op_b} == {ast.Is, ast.Eq}
                    or {op_a, op_b} == {ast.IsNot, ast.NotEq}
                )
                if not swap_ok:
                    return False
                if (
                    ast.dump(a.left, include_attributes=False) == ast.dump(b.left, include_attributes=False)
                    and len(a.comparators) == 1 and len(b.comparators) == 1
                    and ast.dump(a.comparators[0], include_attributes=False) == ast.dump(b.comparators[0], include_attributes=False)
                ):
                    diffs.append((a, b, cur_cls))
                    return True
                return False
        for field_name in a._fields:
            va = getattr(a, field_name)
            vb = getattr(b, field_name)
            if isinstance(va, ast.AST) and isinstance(vb, ast.AST):
                if not walk_pair(va, vb, cur_cls):
                    return False
            elif isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    return False
                for ia, ib in zip(va, vb):
                    if isinstance(ia, ast.AST) and isinstance(ib, ast.AST):
                        if not walk_pair(ia, ib, cur_cls):
                            return False
                    elif ia != ib:
                        return False
            elif va != vb:
                return False
        return True

    if not walk_pair(old_ast, new_ast, None) or len(diffs) != 1:
        return None

    cmp_a, _, cls_node = diffs[0]
    lhs = cmp_a.left
    rhs = cmp_a.comparators[0]

    # Case 1: self.<attr> is None  <->  self.<attr> == None
    for side_x, side_const in ((lhs, rhs), (rhs, lhs)):
        if isinstance(side_const, ast.Constant) and side_const.value is None:
            if (
                isinstance(side_x, ast.Attribute)
                and isinstance(side_x.value, ast.Name)
                and side_x.value.id == "self"
                and cls_node is not None
            ):
                attr = side_x.attr
                type_hint = None
                for node in ast.walk(cls_node):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == attr
                    ):
                        type_hint = _extract_non_none_type_name(node.annotation)
                        break
                if type_hint:
                    return IdentityOpEdit(
                        rel_path=rel_path,
                        enclosing_class=cls_node.name,
                        kind="none",
                        self_attr=attr,
                        type_hint_name=type_hint,
                        enum_cls_name=None,
                        all_class_assigns_safe=True,
                    )

    # Case 2: self.<attr> is EnumCls.MEMBER  <->  self.<attr> == EnumCls.MEMBER
    for side_x, side_enum in ((lhs, rhs), (rhs, lhs)):
        if (
            isinstance(side_enum, ast.Attribute)
            and isinstance(side_enum.value, ast.Name)
            and isinstance(side_x, ast.Attribute)
            and isinstance(side_x.value, ast.Name)
            and side_x.value.id == "self"
            and cls_node is not None
        ):
            attr = side_x.attr
            enum_cls = side_enum.value.id
            # Verify every assignment to self.<attr> in cls_node assigns EnumCls(...) or self.<UPPER_CONST>
            assigns_ok = True
            assign_count = 0
            for node in ast.walk(cls_node):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (
                            isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "self"
                            and tgt.attr == attr
                        ):
                            assign_count += 1
                            v = node.value
                            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == enum_cls:
                                continue
                            if (
                                isinstance(v, ast.Attribute)
                                and isinstance(v.value, ast.Name)
                                and v.value.id in ("self", enum_cls)
                            ):
                                continue
                            assigns_ok = False
            if assign_count > 0 and assigns_ok:
                return IdentityOpEdit(
                    rel_path=rel_path,
                    enclosing_class=cls_node.name,
                    kind="enum",
                    self_attr=attr,
                    type_hint_name=None,
                    enum_cls_name=enum_cls,
                    all_class_assigns_safe=True,
                )

    return None


def _collect_enclosing_ranges(tree: ast.AST):
    funcs: list[tuple[int, int, str, str | None, ast.AST]] = []
    classes: list[tuple[int, int, str, ast.ClassDef]] = []

    def visit(node: ast.AST, cls_stack: list[str], fn_stack: list[str]):
        if isinstance(node, ast.ClassDef):
            q = ".".join(cls_stack + [node.name])
            classes.append((node.lineno, getattr(node, "end_lineno", node.lineno), q, node))
            for child in node.body:
                visit(child, cls_stack + [node.name], fn_stack)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            q = ".".join(cls_stack + fn_stack + [node.name])
            c_outer = ".".join(cls_stack) if cls_stack else None
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = getattr(node, "end_lineno", node.lineno)
            funcs.append((start, end, q, c_outer, node))
            for child in node.body:
                visit(child, cls_stack, fn_stack + [node.name])
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, cls_stack, fn_stack)

    for stmt in getattr(tree, "body", []):
        visit(stmt, [], [])
    return funcs, classes


def _stmt_lines_in_func(fn_node: ast.AST, target_lines: set[int]) -> set[int]:
    """Soundly expand changed/anchor lines inside a function to cover:
    - ALL lines of any enclosing simple statement (handles multi-line expressions/ternaries/returns)
    - ALL lines of any enclosing compound statement header (e.g. multi-line `if (A and B and C):`
      where flipping `and`->`or` changes short-circuiting so tests that only ran line A now reach C!)
    """
    out = set(target_lines)
    COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)

    def _header_end(node: ast.AST) -> int:
        if isinstance(node, (ast.If, ast.While)):
            return getattr(node.test, "end_lineno", node.lineno)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return getattr(node.iter, "end_lineno", node.lineno)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return max((getattr(item.context_expr, "end_lineno", node.lineno) for item in node.items), default=node.lineno)
        if isinstance(node, ast.Match):
            return getattr(node.subject, "end_lineno", node.lineno)
        return node.lineno

    def _walk_stmt(node: ast.AST, parent_if_header: range | None = None):
        if not isinstance(node, ast.stmt) or node is fn_node:
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.stmt):
                    _walk_stmt(ch, None)
            return

        s = getattr(node, "lineno", None)
        e = getattr(node, "end_lineno", s)
        if s is None or e is None:
            return
        if not any(s <= ln <= e for ln in target_lines):
            return

        if isinstance(node, COMPOUND):
            he = _header_end(node)
            hdr_range = range(s, he + 1)
            # If any target line falls inside the compound header, add EVERY line of the header!
            if any(s <= ln <= he for ln in target_lines):
                out.update(hdr_range)
                # If this is an elif, also include parent if's header lines!
                if parent_if_header is not None:
                    out.update(parent_if_header)
            else:
                # Target is inside body/orelse: add first line of enclosing header as safety anchor
                out.add(s)

            for b in getattr(node, "body", []):
                _walk_stmt(b, None)
            orelse = getattr(node, "orelse", [])
            if isinstance(node, ast.If) and len(orelse) == 1 and isinstance(orelse[0], ast.If):
                # elif chain: pass down combined header lines
                combined = range(min(parent_if_header.start, s) if parent_if_header else s, he + 1)
                _walk_stmt(orelse[0], combined)
            else:
                for b in orelse:
                    _walk_stmt(b, None)
            for h in getattr(node, "handlers", []):
                for b in getattr(h, "body", []):
                    _walk_stmt(b, None)
            for b in getattr(node, "finalbody", []):
                _walk_stmt(b, None)
            for c in getattr(node, "cases", []):
                for b in getattr(c, "body", []):
                    _walk_stmt(b, None)
        else:
            # Simple statement (Assign, Return, Expr, Raise, Assert, etc.): add ALL lines [s..e]!
            out.update(range(s, e + 1))

    for stmt in getattr(fn_node, "body", []):
        _walk_stmt(stmt, None)
    return out


def analyze_python_patch(old_src: str | None, new_src: str | None, fp: FilePatch, is_test_file: bool = False) -> PySemanticDiff:
    diff = PySemanticDiff()
    if new_src is None:
        diff.has_unknown_top_edit = True
        return diff

    try:
        new_ast = ast.parse(new_src, filename=fp.path)
    except SyntaxError:
        diff.syntax_error = True
        return diff

    if old_src is None:
        diff.has_unknown_top_edit = True
        return diff

    try:
        old_ast = ast.parse(old_src, filename=fp.path)
    except SyntaxError:
        diff.has_unknown_top_edit = True
        return diff

    if _coding_cookie(old_src) == _coding_cookie(new_src):
        if ast.dump(old_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False):
            diff.is_inert = True
            return diff

    # Check for pure identity-op swap (`is`<->`==`, `is not`<->`!=`)
    diff.identity_op_edit = _detect_pure_identity_op_edit(fp.path, old_ast, new_ast)

    touched = set(fp.old_changed_lines) | set(fp.old_insert_anchors)
    if not touched:
        diff.has_unknown_top_edit = True
        return diff

    funcs, classes = _collect_enclosing_ranges(old_ast)
    funcs_sorted = sorted(funcs, key=lambda x: (x[1] - x[0], x[0]))
    classes_sorted = sorted(classes, key=lambda x: (x[1] - x[0], x[0]))

    touched_test_methods: set[tuple[str, str]] = set()
    all_inside_test_methods = is_test_file
    seen_cls_edits: set[tuple[str, str | None]] = set()

    for ln in sorted(touched):
        matched_fn = None
        for fstart, fend, fq, fcls, fnode in funcs_sorted:
            if fstart <= ln <= fend:
                matched_fn = (fstart, fend, fq, fcls, fnode)
                break

        if matched_fn is not None:
            fstart, fend, fq, fcls, fnode = matched_fn
            body_first = fnode.body[0].lineno if getattr(fnode, "body", None) else fstart + 1
            if ln < body_first:
                diff.func_lines.update(_stmt_lines_in_func(fnode, {ln, body_first}))
                if fcls:
                    if (fcls, fnode.name) not in seen_cls_edits:
                        seen_cls_edits.add((fcls, fnode.name))
                        diff.class_body_edits.append(ClassBodyEdit(fcls, fnode.name))
                else:
                    diff.module_level_symbols.add(fnode.name)
                all_inside_test_methods = False
            else:
                diff.func_lines.update(_stmt_lines_in_func(fnode, {ln}))
                if ln not in fp.old_changed_lines:
                    diff.func_lines.add(body_first)
                if is_test_file:
                    if fcls and fnode.name.startswith("test_"):
                        touched_test_methods.add((fcls, fnode.name))
                    else:
                        all_inside_test_methods = False
            continue

        all_inside_test_methods = False

        matched_cls = None
        for cstart, cend, cq, cnode in classes_sorted:
            if cstart <= ln <= cend:
                matched_cls = (cstart, cend, cq, cnode)
                break

        if matched_cls is not None:
            cstart, cend, cq, cnode = matched_cls
            attr_name = None
            old_val_node = None
            for stmt in cnode.body:
                s = stmt.lineno
                e = getattr(stmt, "end_lineno", s)
                if s <= ln <= e:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        attr_name = stmt.targets[0].id
                        old_val_node = stmt.value
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        attr_name = stmt.target.id
                        old_val_node = stmt.value
                    break
            if (cq, attr_name) not in seen_cls_edits:
                seen_cls_edits.add((cq, attr_name))
                is_dict = False
                dkeys = None
                if attr_name and old_val_node is not None:
                    new_val_node = _find_class_attr_stmt(new_ast, cq, attr_name)
                    is_dict, dkeys = _diff_ast_dicts(old_val_node, new_val_node)
                diff.class_body_edits.append(
                    ClassBodyEdit(cq, attr_name, is_dict_assign=is_dict, dict_value_keys=dkeys)
                )
            continue

        matched_top = False
        for stmt in old_ast.body:
            s = stmt.lineno
            e = getattr(stmt, "end_lineno", s)
            if s <= ln <= e:
                matched_top = True
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    diff.has_import_edit = True
                elif isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            diff.module_level_symbols.add(tgt.id)
                        else:
                            diff.has_unknown_top_edit = True
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    diff.module_level_symbols.add(stmt.target.id)
                else:
                    diff.has_unknown_top_edit = True
                break
        if not matched_top:
            diff.has_unknown_top_edit = True

    if is_test_file and all_inside_test_methods and touched_test_methods:
        diff.only_test_methods = sorted(touched_test_methods)

    return diff


# ---------------------------------------------------------------------------
# SQL Fixture Pair Diffing (exact mirror of tests/helpers.py)
# ---------------------------------------------------------------------------
def _filter_sql_comments(s: str) -> str:
    return "\n".join(line for line in s.splitlines() if line and not line.startswith("--"))


def _extract_sql_meta(sql: str) -> tuple[str, dict[str, str]]:
    meta: dict[str, str] = {}
    sql_lines = sql.split("\n")
    i = 0
    while i < len(sql_lines) and sql_lines[i].startswith("#"):
        if ":" in sql_lines[i]:
            key, val = sql_lines[i].split(":", maxsplit=1)
            meta[key.lstrip("#").strip()] = val.strip()
        i += 1
    return "\n".join(sql_lines[i:]), meta


def parse_sql_fixture_pairs_text(text: str) -> list[tuple[dict[str, str], str, str]]:
    statements = _filter_sql_comments(text).split(";")
    size = len(statements)
    pairs = []
    for i in range(0, size, 2):
        if i + 1 < size:
            sql = statements[i].strip()
            sql, meta = _extract_sql_meta(sql)
            expected = statements[i + 1].strip()
            pairs.append((meta, sql, expected))
    return pairs


def parse_sql_fixtures_text(text: str) -> list[str]:
    out = []
    lines = []
    for line in _filter_sql_comments(text).splitlines():
        lines.append(line)
        if not line.startswith("#"):
            out.append("\n".join(lines))
            lines.clear()
    return out


def diff_sql_fixture_pairs(old_text: str, new_text: str) -> tuple[bool, list[int]]:
    old_pairs = parse_sql_fixture_pairs_text(old_text)
    new_pairs = parse_sql_fixture_pairs_text(new_text)
    if old_pairs == new_pairs:
        return True, []
    if len(old_pairs) == len(new_pairs):
        changed = [i for i, (a, b) in enumerate(zip(old_pairs, new_pairs)) if a != b]
        return False, changed
    old_set = set((tuple(sorted(m.items())), s, e) for m, s, e in old_pairs)
    changed = [
        i for i, (m, s, e) in enumerate(new_pairs)
        if (tuple(sorted(m.items())), s, e) not in old_set
    ]
    return False, changed or list(range(len(new_pairs)))


def diff_sql_fixtures_single(old_text: str, new_text: str) -> tuple[bool, list[int]]:
    old_items = parse_sql_fixtures_text(old_text)
    new_items = parse_sql_fixtures_text(new_text)
    if old_items == new_items:
        return True, []
    if len(old_items) == len(new_items):
        return False, [i for i, (a, b) in enumerate(zip(old_items, new_items)) if a != b]
    old_set = set(old_items)
    changed = [i for i, x in enumerate(new_items) if x not in old_set]
    return False, changed or list(range(len(new_items)))


class OverlayWorkspace:
    def __init__(self, base_root: Path, patched_files: dict[str, str | None]):
        self.base_root = base_root
        self.patched_files = patched_files
        self.root: Path | None = None

    def __enter__(self) -> Path:
        self.root = Path(tempfile.mkdtemp(prefix="ov_", dir="/tmp"))
        real_dirs: set[Path] = {Path(".")}
        deleted_rels = {rel for rel, content in self.patched_files.items() if content is None}
        for rel in self.patched_files:
            p = Path(rel).parent
            while True:
                real_dirs.add(p)
                if p == Path("."):
                    break
                p = p.parent

        for d in sorted(real_dirs, key=lambda x: len(x.parts)):
            dst_dir = self.root / d
            dst_dir.mkdir(parents=True, exist_ok=True)
            src_dir = self.base_root / d
            if src_dir.is_dir():
                for entry in src_dir.iterdir():
                    rel = (d / entry.name).as_posix()
                    if rel.startswith("./"):
                        rel = rel[2:]
                    if rel in deleted_rels or rel in self.patched_files:
                        continue
                    if (d / entry.name) in real_dirs:
                        continue
                    os.symlink(entry, dst_dir / entry.name)

        for rel, content in self.patched_files.items():
            if content is None:
                continue
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

        return self.root

    def __exit__(self, *exc):
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None
        return False
