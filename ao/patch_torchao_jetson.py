#!/usr/bin/env python3
"""Patch installed torchao for Jetson (PyTorch built with USE_DISTRIBUTED=0).

torchao v0.12.0 imports torch.distributed in many places. On Jetson,
PyTorch is built with USE_DISTRIBUTED=0 so these all fail. Also FP8
requires SM 8.9+ (Jetson Orin is SM 8.7).

Nuclear approach:
  1. Stub ALL .py files in float8/ (entire package unusable on Jetson)
  2. Remove _c10d_functional references from nf4tensor.py
  3. Guard nf4tensor import in dtypes/__init__.py
  4. Guard floatx imports in dtypes/
  5. Scan ALL torchao .py files and wrap any torch.distributed / torchao.float8
     imports in try/except with stub definitions
"""
import pathlib
import re
import site


def _extract_names(orig: str) -> list:
    """Extract class/def/CONSTANT names from Python source for stubs."""
    names = set()
    for m in re.finditer(r'^class\s+(\w+)', orig, re.MULTILINE):
        names.add(m.group(1))
    for m in re.finditer(r'^def\s+(\w+)', orig, re.MULTILINE):
        names.add(m.group(1))
    for m in re.finditer(r'^([A-Z_][A-Z_0-9]*)\s*[:=]', orig, re.MULTILINE):
        names.add(m.group(1))
    return sorted(names)


def _stub_float8(ao: pathlib.Path) -> list:
    """Replace all .py files in float8/ with stubs."""
    patched = []
    float8_dir = ao / "float8"
    if not float8_dir.exists():
        return patched

    for py_file in float8_dir.glob("*.py"):
        orig = py_file.read_text()
        names = _extract_names(orig)
        stub = ['"""[Jetson stub] %s — float8 disabled."""' % py_file.name]
        for name in names:
            stub.append(f"{name} = None")
        py_file.write_text("\n".join(stub) + "\n")
        patched.append(str(py_file))

    return patched


def _remove_c10d_functional(ao: pathlib.Path) -> list:
    """Remove c10d_functional op references from ALL torchao .py files.

    These are module-level dict entries like:
        torch.ops._c10d_functional.all_gather_into_tensor.default: handler,
        c10d_functional.all_gather_into_tensor.default,
    They fail because distributed ops aren't registered without USE_DISTRIBUTED.
    """
    patched = []
    for py_file in ao.rglob("*.py"):
        orig = py_file.read_text()
        if "c10d_functional" not in orig:
            continue
        new_text = re.sub(
            r'^\s*(?:torch\.ops\.)?_?c10d_functional\.[^\n]+\n',
            '', orig, flags=re.MULTILINE,
        )
        if new_text != orig:
            py_file.write_text(new_text)
            patched.append(str(py_file))
    return patched


def _guard_dtypes_init(ao: pathlib.Path) -> list:
    """Guard nf4tensor import in dtypes/__init__.py."""
    patched = []
    f = ao / "dtypes" / "__init__.py"
    if not f.exists():
        return patched
    orig = f.read_text()
    if "nf4tensor" in orig and "try:" not in orig:
        new_text = re.sub(
            r'^(from\s+\.nf4tensor\s+import\s+.+)$',
            'try:\n    \\1\nexcept Exception:\n    NF4Tensor = None\n    to_nf4 = None',
            orig, flags=re.MULTILINE,
        )
        if new_text != orig:
            f.write_text(new_text)
            patched.append(str(f))
    return patched


def _guard_floatx(ao: pathlib.Path) -> list:
    """Guard float8 imports in floatx/__init__.py and affine_quantized_tensor_ops.py."""
    patched = []

    f = ao / "dtypes" / "floatx" / "__init__.py"
    if f.exists():
        orig = f.read_text()
        if "try:" not in orig:
            lines = orig.splitlines()
            new_lines = []
            for line in lines:
                if "float8_layout" in line and "import" in line:
                    new_lines.append("try:")
                    new_lines.append("    " + line)
                    new_lines.append("except Exception:")
                    new_lines.append("    Float8Layout = None")
                else:
                    new_lines.append(line)
            f.write_text("\n".join(new_lines) + "\n")
            patched.append(str(f))

    f = ao / "dtypes" / "affine_quantized_tensor_ops.py"
    if f.exists():
        orig = f.read_text()
        pattern = re.compile(
            r'(from\s+torchao\.dtypes\.floatx\.\S+\s+import\s*\()'
            r'(.*?)'
            r'(\))',
            re.DOTALL,
        )

        def wrap_import(m):
            full = m.group(0)
            body = m.group(2)
            names = re.findall(r'(\w+)', body)
            stubs = []
            for name in names:
                if "_check" in name:
                    stubs.append(f"    def {name}(*a, **k): return False")
                else:
                    stubs.append(f"    {name} = None")
            indented = "    " + full.replace("\n", "\n    ")
            return (
                "try:\n" + indented + "\n"
                + "except Exception:\n" + "\n".join(stubs)
            )

        new_text = pattern.sub(wrap_import, orig)
        if new_text != orig:
            f.write_text(new_text)
            patched.append(str(f))

    return patched


def _guard_distributed_imports(ao: pathlib.Path) -> list:
    """Scan ALL torchao .py files and wrap torch.distributed / torchao.float8 imports."""
    patched = []
    float8_dir = ao / "float8"

    # Patterns that trigger the guard
    triggers = [
        "from torch.distributed",
        "import torch.distributed",
        "from torchao.float8",
    ]

    for py_file in ao.rglob("*.py"):
        # Skip float8/ (already stubbed)
        try:
            py_file.relative_to(float8_dir)
            continue
        except ValueError:
            pass

        orig = py_file.read_text()

        # Quick check: any trigger present?
        if not any(t in orig for t in triggers):
            continue

        lines = orig.splitlines()
        new_lines = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]

            is_trigger = any(stripped.startswith(t) for t in triggers)

            if is_trigger:
                # Check if already wrapped
                if i > 0 and new_lines and new_lines[-1].rstrip().endswith("try:"):
                    new_lines.append(line)
                    i += 1
                    continue

                # Collect the full import (may span multiple lines with parens)
                import_lines = [line]
                if "(" in line and ")" not in line:
                    i += 1
                    while i < len(lines) and ")" not in lines[i]:
                        import_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        import_lines.append(lines[i])

                # Extract imported names for stubs
                full_text = " ".join(il.strip() for il in import_lines)
                m = re.search(r'import\s+(.+)', full_text)
                if m:
                    raw = m.group(1).replace("(", "").replace(")", "")
                    imported_names = [n.strip().split(" as ")[-1].strip()
                                      for n in raw.split(",") if n.strip()]
                else:
                    imported_names = []

                new_lines.append(indent + "try:")
                for il in import_lines:
                    new_lines.append(indent + "    " + il.lstrip())
                new_lines.append(indent + "except Exception:")
                if imported_names:
                    for name in imported_names:
                        if name and name.isidentifier():
                            new_lines.append(indent + f"    {name} = None")
                    if not any(n and n.isidentifier() for n in imported_names):
                        new_lines.append(indent + "    pass")
                else:
                    new_lines.append(indent + "    pass")
                changed = True
            else:
                new_lines.append(line)
            i += 1

        if changed:
            py_file.write_text("\n".join(new_lines) + "\n")
            patched.append(str(py_file))

    return patched


def _fix_isinstance_none(ao: pathlib.Path) -> list:
    """Fix isinstance() calls where the type arg may be None due to our stubs.

    Transforms:
        if isinstance(model, Float8Linear):
    Into:
        if Float8Linear is not None and isinstance(model, Float8Linear):
    """
    patched = []
    # Names we may have stubbed to None
    stub_names = {"Float8Linear", "NF4Tensor", "DTensor"}

    for py_file in ao.rglob("*.py"):
        orig = py_file.read_text()
        changed = False
        new_text = orig

        for name in stub_names:
            # Match isinstance(X, Name) where not already guarded
            pattern = re.compile(
                r'(?<!is not None and )isinstance\((\w+),\s*' + re.escape(name) + r'\)'
            )
            replacement = f'{name} is not None and isinstance(\\1, {name})'
            result = pattern.sub(replacement, new_text)
            if result != new_text:
                new_text = result
                changed = True

        if changed:
            py_file.write_text(new_text)
            patched.append(str(py_file))

    return patched


def _fix_intmm_cuda_graph(ao: pathlib.Path) -> list:
    """Fix safe_int_mm __repr__() call that breaks CUDA Graph capture.

    torchao/kernel/intmm.py has:
        if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
    The __repr__() call triggers device-to-host copy which is illegal during
    CUDA Graph capture. Replace with a try/except guard.
    """
    patched = []
    f = ao / "kernel" / "intmm.py"
    if not f.exists():
        return patched
    orig = f.read_text()

    old = '"FakeTensor" in input.__repr__()'
    if old not in orig:
        return patched

    new = '_is_fake_tensor(input)'
    new_text = orig.replace(old, new)

    # Add helper function after imports
    helper = (
        '\ndef _is_fake_tensor(t):\n'
        '    """Check for FakeTensor without __repr__ (breaks CUDA Graph capture)."""\n'
        '    try:\n'
        '        return "FakeTensor" in type(t).__name__\n'
        '    except Exception:\n'
        '        return False\n'
    )

    # Insert after the last import line
    lines = new_text.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1
    lines.insert(insert_idx, helper)
    new_text = "\n".join(lines) + "\n"

    f.write_text(new_text)
    patched.append(str(f))
    return patched


def main():
    sp = pathlib.Path(site.getsitepackages()[0])
    ao = sp / "torchao"

    if not ao.exists():
        print("torchao not found in site-packages, skipping patch.")
        return

    all_patched = []
    all_patched.extend(_stub_float8(ao))
    all_patched.extend(_remove_c10d_functional(ao))
    all_patched.extend(_guard_dtypes_init(ao))
    all_patched.extend(_guard_floatx(ao))
    all_patched.extend(_guard_distributed_imports(ao))
    all_patched.extend(_fix_isinstance_none(ao))
    all_patched.extend(_fix_intmm_cuda_graph(ao))

    for p in all_patched:
        print(f"  patched: {p}")
    print(f"torchao Jetson patches applied ({len(all_patched)} files).")


if __name__ == "__main__":
    main()
