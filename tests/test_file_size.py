"""
Test that Python files don't exceed 500 lines.

Long files are difficult for AI code editors to understand and modify.
Files should be kept concise and focused on a single responsibility.
Following the 150-500 line sweet spot improves maintainability and
AI-assisted development.

Reference: https://medium.com/@eamonn.faherty_58176/right-sizing-your-python-files-the-150-500-line-sweet-spot-for-ai-code-editors-340d550dcea4
"""

from pathlib import Path


def count_code_lines(filepath: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    code_lines = 0
    in_multiline_string = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track multiline strings (docstrings)
        if '"""' in stripped or "'''" in stripped:
            quote = '"""' if '"""' in stripped else "'''"
            count = stripped.count(quote)
            if count == 1:
                in_multiline_string = not in_multiline_string
            # If count is 2 or more, it's a single-line docstring
        
        # Skip blank lines and comments
        if stripped and not stripped.startswith("#") and not in_multiline_string:
            code_lines += 1
    
    return code_lines


def test_python_files_max_500_lines():
    """
    Ensure all Python files in dect/ are at most 500 lines long.
    
    Files exceeding 500 lines indicate:
    - Too many responsibilities in a single module
    - Need for refactoring into smaller, focused modules
    - Potential violation of single responsibility principle
    
    We use a soft buffer of 550 lines to avoid being too strict
    about files that are close to the limit.
    """
    max_lines = 550
    project_root = Path(__file__).parent.parent
    violations = []
    
    # Check dect/ directory
    dect_dir = project_root / "dect"
    if not dect_dir.exists():
        return
    
    for py_file in dect_dir.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in py_file.parts:
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            
            if line_count > max_lines:
                relative_path = py_file.relative_to(project_root)
                violations.append(f"{relative_path}: {line_count} lines (exceeds {max_lines})")
        except Exception:
            continue
    
    if violations:
        violation_list = "\n  - ".join(violations)
        assert False, f"Files exceeding {max_lines} lines:\n  - {violation_list}"


def test_python_files_report():
    """Report line counts for all Python files (informational only)."""
    project_root = Path(__file__).parent.parent
    dect_dir = project_root / "dect"
    
    if not dect_dir.exists():
        return
    
    file_counts = []
    
    for py_file in sorted(dect_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            
            relative_path = py_file.relative_to(project_root)
            file_counts.append((line_count, str(relative_path)))
        except Exception:
            continue
    
    # Sort by line count descending
    file_counts.sort(reverse=True)
    
    print("\n=== File Size Report ===")
    print(f"{'Lines':>6}  File")
    print("-" * 50)
    for lines, path in file_counts:
        status = ""
        if lines > 500:
            status = " [OVER LIMIT]"
        elif lines > 400:
            status = " [warning]"
        print(f"{lines:>6}  {path}{status}")
    print("-" * 50)
    print(f"Total files: {len(file_counts)}")
    print(f"Max lines: {file_counts[0][0] if file_counts else 0}")
