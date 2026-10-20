from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .context_builder import _active_preprocessor_text, _local_feature_defines
from .models import write_json


RTL_EXTENSIONS = {".v", ".sv"}
HEADER_EXTENSIONS = {".vh", ".svh"}
SKIP_PARTS = {
    ".git",
    "build",
    "out",
    "obj_dir",
    "coverage",
    "uvmf_2022",
    "uvmf_template_output",
}
SV_KEYWORDS = {
    "module",
    "interface",
    "package",
    "program",
    "begin",
    "end",
    "if",
    "else",
    "case",
    "for",
    "while",
    "assign",
    "always",
    "always_ff",
    "always_comb",
    "function",
    "task",
    "logic",
    "wire",
    "reg",
    "input",
    "output",
    "inout",
    "generate",
}
DEPENDENCY_DEFINE_DENYLIST = {
    # prim_assert.sv defines this only inside a tool-dependent conditional and
    # deliberately undefines it again. Promoting it globally enables SVA in
    # old Ibex revisions and breaks their bundled Verilator 4.x.
    "INC_ASSERT",
}


@dataclass
class SourceResolution:
    sources: list[str]
    include_dirs: list[str]
    defines: list[str]
    packages: list[str]
    modules: list[str]
    unresolved: list[str]
    manifests_consulted: list[str]

    def to_dict(self) -> dict:
        return {
            "sources": self.sources,
            "include_dirs": self.include_dirs,
            "defines": self.defines,
            "packages": self.packages,
            "modules": self.modules,
            "unresolved": self.unresolved,
            "manifests_consulted": self.manifests_consulted,
        }


class RTLSourceResolver:
    """Small dependency resolver scoped to the selected RTL component.

    It intentionally does not attempt full elaboration.  Repository profiles
    only bound the package/module index so a leaf-module Mini-TB does not scan
    generated verification trees in CVA6 or Caliptra.
    """

    def __init__(self, repo_root: Path, repo: str = "", target_file: str = ""):
        self.repo_root = repo_root.resolve()
        self.repo = repo
        self.search_roots = _source_roots(self.repo_root, repo, target_file)
        self.rtl_files = list(self._files(RTL_EXTENSIONS))
        self.header_files = list(self._files(HEADER_EXTENSIONS))
        self.module_index: dict[str, Path] = {}
        self.package_index: dict[str, Path] = {}
        self.include_index: dict[str, list[Path]] = {}
        for path in self.rtl_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            declarations = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.DOTALL)
            for name in re.findall(r"\bmodule\s+(?:automatic\s+)?([A-Za-z_]\w*)", declarations):
                self.module_index.setdefault(name, path)
            for name in re.findall(r"\bpackage\s+([A-Za-z_]\w*)", declarations):
                self.package_index.setdefault(name, path)
        for path in [*self.rtl_files, *self.header_files]:
            self.include_index.setdefault(path.name, []).append(path)

    def resolve(self, target_file: str, target_module: str) -> SourceResolution:
        target = (self.repo_root / target_file).resolve()
        target.relative_to(self.repo_root)
        if not target.is_file():
            raise FileNotFoundError(f"target source does not exist: {target_file}")

        selected: set[Path] = set()
        package_files: set[Path] = set()
        module_files: set[Path] = set()
        include_dirs: set[Path] = {target.parent}
        unresolved: set[str] = set()
        queue = [target]
        while queue and len(selected) < 120:
            path = queue.pop(0)
            if path in selected:
                continue
            selected.add(path)
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            text = _active_preprocessor_text(
                raw_text, {"VERILATOR", *_dependency_feature_defines(raw_text)}
            )
            if re.search(r"\bpackage\s+", text):
                package_files.add(path)
            else:
                module_files.add(path)

            package_names = set(re.findall(r"\bimport\s+([A-Za-z_]\w*)::", text))
            package_names.update(re.findall(r"\b([A-Za-z_]\w*)::[A-Za-z_]\w*", text))
            for name in sorted(package_names):
                dependency = self.package_index.get(name)
                if dependency:
                    queue.append(dependency)
                else:
                    unresolved.add(f"package:{name}")

            for include in re.findall(r"`include\s+[\"<]([^\">]+)[\">]", text):
                matches = self.include_index.get(Path(include).name, [])
                if matches:
                    best = min(matches, key=lambda item: _path_distance(path.parent, item.parent))
                    include_dirs.add(best.parent)
                    # ``include`` is textual inclusion, even when the fragment
                    # happens to use a .sv suffix.  Compiling the included file
                    # again as a standalone translation unit can duplicate
                    # declarations or place module-local statements (for
                    # example prim_util_memload.sv's ``initial`` block) at file
                    # scope.  The include directory is sufficient for the
                    # compiler to consume it through its owner.
                else:
                    unresolved.add(f"include:{include}")

            for module_name in self._instantiated_modules(text):
                dependency = self.module_index.get(module_name)
                if dependency and dependency not in selected:
                    queue.append(dependency)

        manifests, manifest_defines, manifest_include_dirs = self._manifest_hints(target)
        include_dirs.update(manifest_include_dirs)
        # A wrapper may define a feature before exposing matching conditional
        # ports, while the wrapped module is compiled as a separate source.
        # Promote only macros that are both defined and positively consumed;
        # this excludes conventional `ifndef include guards.
        local_feature_defines: set[str] = set()
        for path in selected:
            text = path.read_text(encoding="utf-8", errors="ignore")
            local_feature_defines.update(_dependency_feature_defines(text))
        ordered_packages = self._order_packages(package_files)
        ordered_modules = sorted(module_files - set(ordered_packages), key=lambda path: (path == target, str(path)))
        if target in ordered_modules:
            ordered_modules.remove(target)
        ordered_modules.append(target)
        sources = [*ordered_packages, *ordered_modules]
        defines = manifest_defines | local_feature_defines
        if self.repo == "ibex":
            # MiniTBs elaborate an individual synthesizable module rather than
            # the full ibex_core hierarchy.  Historical Ibex revisions contain
            # simulation-only diagnostics with absolute hierarchy references;
            # those references are intentionally absent from a module harness.
            defines.add("SYNTHESIS")
        return SourceResolution(
            sources=[str(path.relative_to(self.repo_root)) for path in _dedupe(sources)],
            include_dirs=[str(path.relative_to(self.repo_root)) for path in sorted(include_dirs) if path.is_relative_to(self.repo_root)],
            defines=sorted(defines),
            packages=[str(path.relative_to(self.repo_root)) for path in ordered_packages],
            modules=[str(path.relative_to(self.repo_root)) for path in ordered_modules],
            unresolved=sorted(unresolved),
            manifests_consulted=[str(path.relative_to(self.repo_root)) for path in manifests],
        )

    def write(self, resolution: SourceResolution, path: Path) -> None:
        write_json(path, resolution.to_dict())

    def _instantiated_modules(self, text: str) -> set[str]:
        clean = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.DOTALL)
        candidates = set(
            re.findall(
                r"(?:^|[;:]|\bgenerate\b|\bbegin\b)\s*([A-Za-z_]\w*)\s*"
                r"(?:#\s*\(.*?\)\s*)?[A-Za-z_]\w*\s*\(",
                clean,
                flags=re.DOTALL | re.MULTILINE,
            )
        )
        # In a named generate block (``begin : label``), the anchored pattern
        # sees the label before it sees the following instance.  Filtering a
        # broader declaration-shaped match through the module index recovers
        # those dependencies without admitting arbitrary identifiers.
        candidates.update(
            re.findall(
                r"\b([A-Za-z_]\w*)\s+(?:#\s*\(.*?\)\s*)?[A-Za-z_]\w*\s*\(",
                clean,
                flags=re.DOTALL | re.MULTILINE,
            )
        )
        return {name for name in candidates if name not in SV_KEYWORDS and name in self.module_index}

    def _order_packages(self, packages: set[Path]) -> list[Path]:
        remaining = set(packages)
        ordered: list[Path] = []
        while remaining:
            progress = False
            for path in sorted(remaining):
                text = path.read_text(encoding="utf-8", errors="ignore")
                imports = set(re.findall(r"\bimport\s+([A-Za-z_]\w*)::", text))
                blockers = {
                    dependency
                    for name in imports
                    if (dependency := self.package_index.get(name)) in remaining and dependency != path
                }
                if not blockers:
                    ordered.append(path)
                    remaining.remove(path)
                    progress = True
                    break
            if not progress:
                ordered.extend(sorted(remaining))
                break
        return ordered

    def _manifest_hints(self, target: Path) -> tuple[list[Path], set[str], set[Path]]:
        manifests = self._manifest_files()
        consulted: list[Path] = []
        defines: set[str] = set()
        include_dirs: set[Path] = set()
        target_name = target.name
        for path in manifests:
            if path.stat().st_size > 500_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if target_name not in text and target.parent.name not in text:
                continue
            consulted.append(path)
            for value in re.findall(r"(?:\+define\+|-D)([A-Za-z_]\w*)", text):
                defines.add(value)
            for value in re.findall(r"(?:\+incdir\+|-I)([^\s]+)", text):
                cleaned = _expand_manifest_path(value, self.repo_root, path)
                candidate = Path(cleaned)
                if not candidate.is_absolute():
                    candidate = (path.parent / candidate).resolve()
                if candidate.is_dir() and candidate.is_relative_to(self.repo_root):
                    include_dirs.add(candidate)
            if len(consulted) >= 4:
                break
        return consulted, defines, include_dirs

    def _manifest_files(self) -> list[Path]:
        names = {"Bender.yml", "compile.yml"}
        suffixes = {".f", ".vf", ".core"}
        if not self.repo:
            candidates = self.repo_root.rglob("*")
        else:
            locations = {self.repo_root, *self.search_roots}
            for root in tuple(self.search_roots):
                locations.update(parent for parent in root.parents if parent.is_relative_to(self.repo_root))
            collected: list[Path] = []
            for location in sorted(locations):
                if location == self.repo_root:
                    collected.extend(location.iterdir())
                elif location.is_dir():
                    collected.extend(location.rglob("*"))
            candidates = iter(collected)
        output = [
            path
            for path in candidates
            if path.is_file() and (path.suffix in suffixes or path.name in names)
        ]
        return sorted(set(output))[:200]

    def _files(self, suffixes: set[str]) -> Iterable[Path]:
        seen: set[Path] = set()
        for root in self.search_roots:
            for path in root.rglob("*"):
                if path in seen or not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                relative_parts = set(path.relative_to(self.repo_root).parts)
                if relative_parts & SKIP_PARTS:
                    continue
                seen.add(path)
                yield path


def _source_roots(repo_root: Path, repo: str, target_file: str) -> list[Path]:
    configured: list[Path] = []
    if repo == "ibex":
        configured = [
            repo_root / "rtl",
            repo_root / "examples" / "sim" / "rtl",
            # Ibex moved the lowRISC primitive library across releases.  The
            # exact parent is also required as an include directory for
            # `include "prim_assert.sv" and related assertion macros.
            repo_root / "shared" / "rtl",
            repo_root / "vendor" / "lowrisc_ip" / "prim" / "rtl",
            repo_root / "vendor" / "lowrisc_ip" / "ip" / "prim" / "rtl",
            # Older vendored lowRISC revisions generate abstract primitives
            # from a small wrapper plus the generic implementation. FuseSoC
            # normally supplies these files; isolated MiniTB elaboration must
            # index their checked-in equivalents directly.
            repo_root / "vendor" / "lowrisc_ip" / "ip" / "prim_generic" / "rtl",
            repo_root / "vendor" / "lowrisc_ip" / "prim_generic" / "rtl",
            repo_root / "vendor" / "lowrisc_ip" / "prim_xilinx" / "rtl",
            repo_root / "dv" / "uvm" / "core_ibex" / "common" / "prim",
            repo_root / "vendor" / "lowrisc_ip" / "dv" / "sv" / "dv_utils",
        ]
    elif repo == "cva6":
        configured = [repo_root / "core"]
    elif repo == "caliptra-rtl":
        parts = Path(target_file).parts
        if parts[:4] == ("src", "riscv_core", "veer_el2", "rtl"):
            configured = [repo_root.joinpath(*parts[:4])]
        elif len(parts) >= 2 and parts[0] == "src":
            configured = [repo_root / "src" / parts[1]]

    roots = [path.resolve() for path in configured if path.is_dir()]
    target = (repo_root / target_file).resolve() if target_file else None
    if target and target.is_file() and not any(target.is_relative_to(root) for root in roots):
        roots.insert(0, target.parent)
    return roots or [repo_root]


def _dependency_feature_defines(text: str) -> set[str]:
    return _local_feature_defines(text) - DEPENDENCY_DEFINE_DENYLIST


def _expand_manifest_path(value: str, repo_root: Path, manifest: Path) -> str:
    replacements = {
        "${RTL_ROOT}": str(repo_root),
        "$RTL_ROOT": str(repo_root),
        "${CALIPTRA_ROOT}": str(repo_root),
        "$CALIPTRA_ROOT": str(repo_root),
    }
    compile_root = manifest.parent.parent if manifest.name == "compile.yml" else manifest.parent
    replacements.update(
        {
            "${COMPILE_ROOT}": str(compile_root),
            "$COMPILE_ROOT": str(compile_root),
        }
    )
    cleaned = value
    for marker, replacement in replacements.items():
        cleaned = cleaned.replace(marker, replacement)
    return cleaned


def _path_distance(left: Path, right: Path) -> int:
    left_parts = left.parts
    right_parts = right.parts
    common = 0
    for lpart, rpart in zip(left_parts, right_parts):
        if lpart != rpart:
            break
        common += 1
    return (len(left_parts) - common) + (len(right_parts) - common)


def _dedupe(values: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
