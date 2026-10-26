DB_PATH = "aidev.db"

# MSR paper artifact data
MSR_DATA_DIR = "data/msr_paper"

# File filtering
EXCLUDED_FILE_PATTERNS = [
    r"(^|/)tests?/",
    r"(^|/)spec/",
    r"_test\.\w+$",
    r"_spec\.\w+$",
    r"\.test\.\w+$",
    r"\.spec\.\w+$",
    r"(^|/)__(tests|test)__/",
    r"(^|/)(README|CHANGELOG|CHANGES|LICENSE|CONTRIBUTING)",
    r"(^|/)\.github/",
    r"(^|/)(ci|\.circleci|\.travis)/",
    r"(package-lock\.json|yarn\.lock|Pipfile\.lock|poetry\.lock)$",
    r"(^|/)docs?/",
]
BUSINESS_LOGIC_EXTENSIONS = {
    ".as", ".c", ".cc", ".cjs", ".cpp", ".cs", ".dart", ".erl", ".exs",
    ".fs", ".fsi", ".go", ".h", ".hpp", ".java", ".jl", ".js", ".kt",
    ".lua", ".mbt", ".php", ".py", ".pxd", ".pyx", ".qml", ".rb", ".rs",
    ".scm", ".sh", ".ts", ".tsx",
    # Added: ActionScript, C, Dart, MoonBit, QML — covers 10 previously skipped PRs
}

PATCH_TRUNCATE_LENGTH = 4000

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.4"
CLASSIFY_CONCURRENCY = 30
MIN_CONFIDENCE = 0.8
