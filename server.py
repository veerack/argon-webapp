import os
import sys
import json
import time
import uuid
import base64
import threading
import queue
import re
import subprocess
from pathlib import Path
from flask import Flask, Response, request, send_from_directory, jsonify

# Load environment variables
env_path = Path.home() / ".gemini" / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai is not installed.")
    sys.exit(1)

app = Flask(__name__, static_folder=".")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONVOS_FILE = DATA_DIR / "conversations.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
LATEST_WORK_FILE = DATA_DIR / "latest_argon_work.md"
LATEST_CONTEXT_FILE = DATA_DIR / "argon_latest_context.json"

DEFAULT_SETTINGS = {
    "auth_mode": "vertex",
    "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    "gcp_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "workspace_root": os.environ.get("STUDIO_WORKSPACE", ""),
    "vps_ssh_target": "",
    "vps_remote_dir": "",
    "custom_instruction": ""
}

SETTINGS_LOCK = threading.Lock()
CONVOS_LOCK = threading.Lock()

# Global runtime state
client = None
gcp_project = ""
gcp_location = "global"
api_key = ""
WORKSPACE_ROOT = ""

def load_settings():
    with SETTINGS_LOCK:
        if not SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_SETTINGS, f, indent=2)
            return dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                merged = dict(DEFAULT_SETTINGS)
                merged.update(saved)
                return merged
        except Exception as e:
            print(f"Error reading settings: {e}")
            return dict(DEFAULT_SETTINGS)

def save_settings_data(settings_dict):
    with SETTINGS_LOCK:
        tmp_file = SETTINGS_FILE.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        tmp_file.replace(SETTINGS_FILE)

def reinit_genai_client(settings_data):
    global client, gcp_project, gcp_location, api_key, WORKSPACE_ROOT
    WORKSPACE_ROOT = settings_data.get("workspace_root", "")
    auth_mode = settings_data.get("auth_mode", "vertex")
    gcp_project = (settings_data.get("gcp_project") or "").strip()
    gcp_location = (settings_data.get("gcp_location") or "global").strip()
    api_key = (settings_data.get("api_key") or "").strip()

    if not gcp_project and not api_key:
        client = None
        print("Notice: No Google Cloud Project ID or API key configured yet.")
        print("Please open http://localhost:4321 and configure your settings.")
        return

    if auth_mode == "vertex":
        if not gcp_project:
            raise ValueError("GCP Project ID is required for Vertex AI mode.")
        client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
    elif auth_mode == "api_key":
        if not api_key:
            raise ValueError("Gemini API Key is required for API key mode.")
        client = genai.Client(api_key=api_key)
    else:
        if gcp_project:
            client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        elif api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = None

# Initialize client on startup
initial_settings = load_settings()
try:
    reinit_genai_client(initial_settings)
except Exception as e:
    print(f"Warning: Initial client setup failed: {e}")

# --- Dynamic System Instruction Builder ---
def get_system_instruction():
    settings = load_settings()
    ws = settings.get("workspace_root", WORKSPACE_ROOT)
    vps_target = settings.get("vps_ssh_target", "")
    vps_dir = settings.get("vps_remote_dir", "")
    custom = settings.get("custom_instruction", "")

    lines = [
        f"You are working directly inside the project workspace located at: {ws}.",
        "You have full direct tools: read_file, edit_file, write_file, list_directory, search_code, and run_command.\n",
        "Terminal Execution (run_command):",
        "- You have an integrated live terminal via the 'run_command' tool.",
        f"- You can execute shell commands on the user's machine directly in '{ws}'.",
    ]
    if vps_target and vps_dir:
        lines.append(f"- If target='vps', commands execute on the remote host '{vps_target}' in directory '{vps_dir}'.")
    lines.extend([
        "- Use run_command autonomously to run scripts, compile code, test builds, or check outputs whenever needed.",
        "- Strict Constraint: Never run git commands under any circumstances (git commands are disabled by project policy).",
        "- Tool Calling Rule: Never output fake execution logs or 'Workspace actions completed' in your text response. Always execute real actions by invoking the function calls (run_command, read_file, edit_file, write_file), and only communicate with the user using clear, natural explanations.\n",
        "Autonomy & Execution Instructions:",
        "You have completely unlimited tool calls and tool rounds. Execute as many actions and rounds as necessary to complete entire tasks end-to-end without stopping prematurely.",
        "When the user gives high-level requests, autonomously identify the relevant file(s), read them using read_file to understand existing state, ",
        "modify or create files using edit_file/write_file, and execute terminal commands using run_command to verify your work. ",
        "Never ask the user to run commands or provide file paths unless genuinely unable to proceed after searching.",
        "Always provide your complete, detailed analysis and full implementation without truncating or cutting responses short."
    ])
    if custom and custom.strip():
        lines.append(f"\nUser Project Guidelines:\n{custom.strip()}")
    return "\n".join(lines)

# --- Workspace Safety & File Helpers ---
def get_safe_workspace_path(rel_path):
    rel_clean = (rel_path or "").strip().strip("'\"")
    if os.path.isabs(rel_clean):
        norm = os.path.normpath(rel_clean)
        if (WORKSPACE_ROOT and norm.lower().startswith(WORKSPACE_ROOT.lower())) or norm.lower().startswith("c:\\tools"):
            return norm
    rel = rel_clean.replace("/", "\\").lstrip("\\")
    full = os.path.normpath(os.path.join(WORKSPACE_ROOT, rel))
    if not ((WORKSPACE_ROOT and full.lower().startswith(WORKSPACE_ROOT.lower())) or full.lower().startswith("c:\\tools")):
        raise PermissionError(f"Access denied: path outside workspace ({rel_path})")
    return full

def read_workspace_file_content(rel_path, start_line=None, end_line=None):
    full = get_safe_workspace_path(rel_path)
    if not os.path.exists(full):
        return f"Error: File '{rel_path}' does not exist in {WORKSPACE_ROOT}."
    if os.path.isdir(full):
        return f"Error: '{rel_path}' is a directory, not a file."
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if start_line is not None or end_line is not None:
            s = max(1, int(start_line)) if start_line else 1
            e = min(total, int(end_line)) if end_line else total
            sliced = "".join(all_lines[s-1:e])
            return f"--- '{rel_path}' (lines {s}-{e} of {total}) ---\n" + sliced
        else:
            return "".join(all_lines)
    except Exception as e:
        return f"Error reading file '{rel_path}': {str(e)}"

def get_file_line_count(rel_path):
    try:
        full = get_safe_workspace_path(rel_path)
        if os.path.isfile(full):
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
    except Exception:
        pass
    return None

def write_workspace_file_content(rel_path, content):
    full = get_safe_workspace_path(rel_path)
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to '{rel_path}'."
    except Exception as e:
        return f"Error writing to file '{rel_path}': {str(e)}"

def edit_workspace_file_content(rel_path, target, replacement):
    full = get_safe_workspace_path(rel_path)
    if not os.path.exists(full):
        return f"Error: File '{rel_path}' does not exist."
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            current = f.read()
        if target not in current:
            return f"Error: Target text not found in '{rel_path}'."
        updated = current.replace(target, replacement, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(updated)
        return f"Successfully updated '{rel_path}'."
    except Exception as e:
        return f"Error editing '{rel_path}': {str(e)}"

def list_workspace_dir_entries(rel_path=""):
    full = get_safe_workspace_path(rel_path)
    if not os.path.exists(full):
        return f"Error: Directory '{rel_path}' does not exist."
    if not os.path.isdir(full):
        return f"Error: '{rel_path}' is a file, not a directory."
    try:
        entries = []
        for name in sorted(os.listdir(full)):
            if name in (".next", "node_modules", ".git", ".devbox_credentials", "cache", "__pycache__"):
                continue
            entry_path = os.path.join(full, name)
            is_dir = os.path.isdir(entry_path)
            entries.append(f"[{'DIR ' if is_dir else 'FILE'}] {name}")
        return "\n".join(entries) if entries else "(Empty directory)"
    except Exception as e:
        return f"Error listing directory '{rel_path}': {str(e)}"

SEARCH_CACHE = {}
SEARCH_CACHE_LOCK = threading.Lock()
SEARCH_CACHE_TTL = 60.0  # seconds

EXCLUDE_DIRS = {
    ".git", ".next", "node_modules", "cache", "__pycache__", ".devbox_credentials",
    "build", ".gradle", "target", ".idea", ".vscode", "dist", "out", "bin", "obj",
    "libraries", "versions", "world", "world_nether", "world_the_end", "logs", ".paper-remapped"
}

ALLOWED_CODE_EXTENSIONS = {
    ".java", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".yml", ".yaml",
    ".toml", ".xml", ".css", ".scss", ".html", ".sql", ".sh", ".py", ".txt",
    ".properties", ".cfg", ".conf", ".gradle", ".env", ".cmd", ".bat", ".rs", ".go", ".c", ".cpp", ".h"
}

def search_workspace_code_entries(query, ext="", stop_check=None):
    if not query or not query.strip():
        return "Error: Empty search query."

    clean_query = query.strip()
    cache_key = (clean_query.lower(), ext.lower(), WORKSPACE_ROOT)
    now = time.time()

    with SEARCH_CACHE_LOCK:
        if cache_key in SEARCH_CACHE:
            cached_time, cached_res = SEARCH_CACHE[cache_key]
            if now - cached_time < SEARCH_CACHE_TTL:
                return cached_res

    # Fast path: Remote VPS grep if workspace is on V:\ or SSH target is configured
    is_vps_ws = WORKSPACE_ROOT.lower().startswith("v:") or bool(load_settings().get("vps_ssh_target"))
    if is_vps_ws:
        try:
            settings = load_settings()
            vps_host = settings.get("vps_ssh_target", "vps")
            vps_dir = settings.get("vps_remote_dir", "/home/ownerv")
            if vps_host:
                rel_ws = ""
                if WORKSPACE_ROOT.lower().startswith("v:"):
                    rel_ws = os.path.relpath(WORKSPACE_ROOT, "V:\\").replace("\\", "/")
                target_remote = vps_dir if (not rel_ws or rel_ws == ".") else f"{vps_dir.rstrip('/')}/{rel_ws}"

                safe_q = clean_query.replace("'", "'\\''")
                include_flag = f"--include=*{ext}" if ext else ""
                exclude_flags = "--exclude-dir={.git,.next,node_modules,build,.gradle,target,.idea,.vscode,dist,out,bin,obj,cache,__pycache__,libraries,versions,world,world_nether,world_the_end,logs,.paper-remapped}"
                cmd = f"cd {target_remote} && grep -rIn -I {exclude_flags} {include_flag} -i '{safe_q}' . 2>/dev/null | head -n 40"

                res = subprocess.run(["ssh", "-T", vps_host, cmd], capture_output=True, text=True, timeout=10)
                out = (res.stdout or "").strip()
                if out:
                    lines = []
                    for line in out.splitlines():
                        clean_line = line[2:] if line.startswith("./") else line
                        lines.append(clean_line[:140])
                    result_str = "\n".join(lines)
                    if len(lines) >= 40:
                        result_str += "\n... (more matches truncated)"
                else:
                    result_str = f"No matches found for '{query}'."

                with SEARCH_CACHE_LOCK:
                    SEARCH_CACHE[cache_key] = (now, result_str)
                return result_str
        except Exception:
            pass

    # Local file walk fallback with strict pruning, extension check and timeout
    if not os.path.exists(WORKSPACE_ROOT):
        return f"Error: Workspace '{WORKSPACE_ROOT}' does not exist."

    results = []
    q_lower = clean_query.lower()
    start_search_time = time.time()
    max_search_duration = 10.0

    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        if stop_check and stop_check():
            return "Search canceled."
        if time.time() - start_search_time > max_search_duration:
            results.append("... [Search timed out after 10s; more matches may exist]")
            break

        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS and not d.startswith(".")]

        for fname in files:
            if stop_check and stop_check():
                return "Search canceled."

            fname_lower = fname.lower()
            if ext and not fname_lower.endswith(ext.lower()):
                continue

            f_ext = os.path.splitext(fname_lower)[1]
            if f_ext not in ALLOWED_CODE_EXTENSIONS and ext == "":
                continue

            full = os.path.join(root, fname)
            try:
                if os.path.getsize(full) > 1024 * 1024:
                    continue
                rel = os.path.relpath(full, WORKSPACE_ROOT)
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for lno, line in enumerate(f, 1):
                        if q_lower in line.lower():
                            results.append(f"{rel}:{lno}: {line.strip()[:100]}")
                            if len(results) >= 40:
                                res_text = "\n".join(results) + "\n... (more matches truncated)"
                                with SEARCH_CACHE_LOCK:
                                    SEARCH_CACHE[cache_key] = (now, res_text)
                                return res_text
            except Exception:
                continue

    result_str = "\n".join(results) if results else f"No matches found for '{query}'."
    with SEARCH_CACHE_LOCK:
        SEARCH_CACHE[cache_key] = (now, result_str)
    return result_str

def is_git_command(cmd_str):
    if re.search(r'(^|[\s;&|])git(\.exe)?([\s;&|]|$)', cmd_str, re.IGNORECASE):
        return True
    return False

def execute_terminal_command(command, target="local", timeout=90):
    cmd_clean = (command or "").strip()
    if not cmd_clean:
        return "error: command is empty."

    if is_git_command(cmd_clean):
        return "error: git commands are strictly prohibited in this environment."

    settings = load_settings()
    is_vps = (target == "vps") or cmd_clean.startswith("vps ")
    if cmd_clean.startswith("vps "):
        cmd_clean = cmd_clean[4:].strip()
        is_vps = True

    try:
        if is_vps:
            vps_host = settings.get("vps_ssh_target", "vps")
            vps_dir = settings.get("vps_remote_dir", "")
            if not vps_host:
                return "error: remote SSH target is not configured in settings."
            remote_cmd = f"cd {vps_dir} && {cmd_clean}" if vps_dir else cmd_clean
            res = subprocess.run(
                ["ssh", "-T", vps_host, remote_cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            if not os.path.exists(WORKSPACE_ROOT):
                return f"error: workspace directory '{WORKSPACE_ROOT}' does not exist on this machine."
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd_clean],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout
            )

        stdout = res.stdout or ""
        stderr = res.stderr or ""

        output_parts = []
        if stdout.strip():
            output_parts.append(stdout.strip())
        if stderr.strip():
            output_parts.append(f"[stderr]\n{stderr.strip()}")

        combined = "\n".join(output_parts).strip()
        if not combined:
            combined = "(command completed with no output)"

        if len(combined) > 8000:
            combined = combined[:4000] + f"\n\n... [truncated {len(combined) - 8000} characters] ...\n\n" + combined[-4000:]

        return f"[exit code: {res.returncode}]\n{combined}"

    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout} seconds."
    except Exception as e:
        return f"error executing command: {str(e)}"

# --- Gemini Tool Declarations ---
WORKSPACE_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="read_file",
        description="Read the contents of a file in the workspace. Optionally specify a line range (start_line, end_line).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Relative file path in workspace"),
                "start_line": types.Schema(type="INTEGER", description="Optional 1-based start line"),
                "end_line": types.Schema(type="INTEGER", description="Optional 1-based end line")
            },
            required=["path"]
        )
    ),
    types.FunctionDeclaration(
        name="edit_file",
        description="Edit an existing file in the workspace by exact string replacement.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Relative file path"),
                "target": types.Schema(type="STRING", description="Exact character sequence to be replaced"),
                "replacement": types.Schema(type="STRING", description="The replacement text to insert")
            },
            required=["path", "target", "replacement"]
        )
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Create a new file or completely overwrite an existing file in the workspace.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Relative file path"),
                "content": types.Schema(type="STRING", description="Full content of the file")
            },
            required=["path", "content"]
        )
    ),
    types.FunctionDeclaration(
        name="list_directory",
        description="List files and subdirectories in a directory of the workspace.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Relative directory path, or '' for workspace root")
            }
        )
    ),
    types.FunctionDeclaration(
        name="search_code",
        description="Search for symbols, function names, imports, or code patterns across the codebase.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Search string or symbol name"),
                "extension": types.Schema(type="STRING", description="Optional file extension filter, e.g. '.ts' or '.py'")
            },
            required=["query"]
        )
    ),
    types.FunctionDeclaration(
        name="run_command",
        description="Execute a terminal command on the machine. Runs in the workspace directory via PowerShell, or on the remote host via SSH if target='vps'. Use this autonomously to run scripts, compile code, test builds, or check outputs. Never run git commands.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "command": types.Schema(type="STRING", description="The exact shell command line string to execute."),
                "target": types.Schema(type="STRING", description="Execution target: 'local' (default, runs in workspace) or 'vps' (runs via SSH).")
            },
            required=["command"]
        )
    )
]

def execute_workspace_tool(name, args, stop_check=None):
    if name == "read_file":
        path = args.get("path", "")
        start_line = args.get("start_line") or args.get("StartLine")
        end_line = args.get("end_line") or args.get("EndLine")
        return read_workspace_file_content(path, start_line, end_line)
    elif name == "edit_file":
        return edit_workspace_file_content(args.get("path", ""), args.get("target", ""), args.get("replacement", ""))
    elif name == "write_file":
        return write_workspace_file_content(args.get("path", ""), args.get("content", ""))
    elif name == "list_directory":
        return list_workspace_dir_entries(args.get("path", ""))
    elif name == "search_code":
        return search_workspace_code_entries(args.get("query", ""), args.get("extension", ""), stop_check=stop_check)
    elif name == "run_command":
        return execute_terminal_command(args.get("command", ""), args.get("target", "local"))
    else:
        return f"Error: Unknown tool '{name}'."

# --- Conversations Storage ---
def load_conversations_data():
    with CONVOS_LOCK:
        if not CONVOS_FILE.exists():
            return []
        try:
            with open(CONVOS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return []

def save_conversations_data(convos):
    with CONVOS_LOCK:
        tmp_file = CONVOS_FILE.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(convos, f, indent=2, ensure_ascii=False)
        tmp_file.replace(CONVOS_FILE)

# --- HTTP Endpoints ---

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/models")
def get_models():
    settings = load_settings()
    ws = settings.get("workspace_root", WORKSPACE_ROOT)
    proj = settings.get("gcp_project", "")
    billing_url = f"https://console.cloud.google.com/billing?project={proj}" if proj else "https://console.cloud.google.com/billing"
    return jsonify({
        "models": [
            {
                "id": "gemini-4-argon",
                "name": "Argon",
                "tag": "Reasoning & Architecture",
                "badge": "Argon"
            },
            {
                "id": "gemini-4-carbon",
                "name": "Carbon",
                "tag": "Security & Code Auditor",
                "badge": "Carbon"
            }
        ],
        "project": proj,
        "billing_url": billing_url,
        "workspace_root": ws,
        "workspace_basename": os.path.basename(os.path.normpath(ws)) or ws,
        "workspace_mounted": os.path.exists(ws)
    })

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    settings = load_settings()
    ws = settings.get("workspace_root", WORKSPACE_ROOT)
    proj = settings.get("gcp_project", "")
    billing_url = f"https://console.cloud.google.com/billing?project={proj}" if proj else "https://console.cloud.google.com/billing"
    return jsonify({
        "settings": settings,
        "workspace_exists": os.path.exists(ws),
        "workspace_basename": os.path.basename(os.path.normpath(ws)) or ws,
        "billing_url": billing_url
    })

@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No settings provided"}), 400

    current = load_settings()
    for k in ["auth_mode", "gcp_project", "gcp_location", "api_key", "workspace_root", "vps_ssh_target", "vps_remote_dir", "custom_instruction"]:
        if k in payload:
            current[k] = payload[k]

    try:
        reinit_genai_client(current)
        save_settings_data(current)
        ws = current.get("workspace_root", "")
        proj = current.get("gcp_project", "")
        billing_url = f"https://console.cloud.google.com/billing?project={proj}" if proj else "https://console.cloud.google.com/billing"
        return jsonify({
            "success": True,
            "message": "settings updated successfully",
            "settings": current,
            "workspace_exists": os.path.exists(ws),
            "workspace_basename": os.path.basename(os.path.normpath(ws)) or ws,
            "billing_url": billing_url
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"error initializing client: {str(e)}"}), 400

@app.route("/api/settings/validate_path", methods=["POST"])
def api_validate_path():
    payload = request.get_json(force=True)
    p = payload.get("path", "").strip()
    if not p:
        return jsonify({"exists": False, "is_dir": False, "message": "path is empty"})
    exists = os.path.exists(p)
    is_dir = os.path.isdir(p) if exists else False
    return jsonify({
        "exists": exists,
        "is_dir": is_dir,
        "normalized": os.path.normpath(p) if exists else p,
        "basename": os.path.basename(os.path.normpath(p)) if exists else p
    })

@app.route("/api/workspace/files", methods=["GET"])
def get_workspace_files():
    file_list = []
    if not os.path.exists(WORKSPACE_ROOT):
        return jsonify({"error": f"Workspace '{WORKSPACE_ROOT}' is not accessible", "files": []})

    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        dirs[:] = [d for d in dirs if d not in (".next", "node_modules", ".git", "cache", "__pycache__", ".devbox_credentials")]
        for fname in files:
            if fname.endswith(('.ts', '.tsx', '.js', '.jsx', '.py', '.json', '.md', '.css', '.env', '.sh', '.sql', '.html', '.txt')):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, WORKSPACE_ROOT).replace("\\", "/")
                file_list.append(rel)

    return jsonify({"root": WORKSPACE_ROOT, "files": sorted(file_list)})

@app.route("/api/workspace/file", methods=["GET"])
def get_workspace_file():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "Missing path parameter"}), 400
    try:
        full = get_safe_workspace_path(path)
        if not os.path.exists(full):
            return jsonify({"error": f"File '{path}' does not exist"}), 404
        if os.path.isdir(full):
            return jsonify({"error": f"'{path}' is a directory"}), 400
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.splitlines()
        rel_display = os.path.relpath(full, WORKSPACE_ROOT).replace("\\", "/") if (WORKSPACE_ROOT and full.lower().startswith(WORKSPACE_ROOT.lower())) else full
        return jsonify({
            "path": rel_display,
            "full_path": full,
            "content": content,
            "lines": lines,
            "lines_count": len(lines)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/workspace/apply", methods=["POST"])
def apply_to_workspace():
    data = request.get_json(force=True)
    path = data.get("path")
    content = data.get("content")
    mode = data.get("mode", "write")
    target = data.get("target")
    replacement = data.get("replacement")

    if not path:
        return jsonify({"error": "Missing path"}), 400

    if mode == "edit" and target is not None:
        result = edit_workspace_file_content(path, target, replacement or "")
    else:
        if content is None:
            return jsonify({"error": "Missing content"}), 400
        result = write_workspace_file_content(path, content)

    if result.startswith("Error:"):
        return jsonify({"success": False, "message": result}), 400
    return jsonify({"success": True, "message": result, "path": path})

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    convos = load_conversations_data()
    summaries = []
    for c in convos:
        summaries.append({
            "id": c.get("id"),
            "title": c.get("title", "Untitled Chat"),
            "model": c.get("model", "gemini-4-argon"),
            "thinking_level": c.get("thinking_level", "high"),
            "messageCount": len(c.get("messages", [])),
            "createdAt": c.get("createdAt"),
            "updatedAt": c.get("updatedAt", c.get("createdAt", time.time()))
        })
    summaries.sort(key=lambda x: -x.get("updatedAt", 0))
    return jsonify(summaries)

@app.route("/api/conversations/<cid>", methods=["GET"])
def get_conversation(cid):
    convos = load_conversations_data()
    for c in convos:
        if c.get("id") == cid:
            return jsonify(c)
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/api/conversations", methods=["POST"])
def save_conversation():
    payload = request.get_json(force=True)
    cid = payload.get("id")
    convos = load_conversations_data()
    now = time.time()

    existing = None
    for c in convos:
        if c.get("id") == cid:
            existing = c
            break

    if existing:
        existing["title"] = payload.get("title", existing.get("title", "Untitled Chat"))
        existing["model"] = payload.get("model", existing.get("model", "gemini-4-argon"))
        existing["thinking_level"] = payload.get("thinking_level", existing.get("thinking_level", "high"))
        new_msgs = payload.get("messages")
        if new_msgs is not None:
            if len(new_msgs) == 0 and len(existing.get("messages", [])) > 0 and not payload.get("clear_all", False):
                pass
            else:
                existing["messages"] = new_msgs
        existing["updatedAt"] = now
        saved_convo = existing
    else:
        new_id = cid or str(uuid.uuid4())
        saved_convo = {
            "id": new_id,
            "title": payload.get("title", "New Chat"),
            "model": payload.get("model", "gemini-4-argon"),
            "thinking_level": payload.get("thinking_level", "high"),
            "messages": payload.get("messages", []),
            "createdAt": now,
            "updatedAt": now
        }
        convos.append(saved_convo)

    save_conversations_data(convos)

    if saved_convo.get("messages"):
        last_m = saved_convo["messages"][-1]
        try:
            with open(LATEST_WORK_FILE, "w", encoding="utf-8") as f:
                f.write(f"# Latest Argon/Carbon Work ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                f.write(f"**Chat:** {saved_convo.get('title')}\n")
                f.write(f"**Model:** {saved_convo.get('model')}\n\n")
                f.write(f"### Last Response\n\n{last_m.get('text', '')}\n")

            with open(LATEST_CONTEXT_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "updatedAt": now,
                    "conversation_id": saved_convo.get("id"),
                    "title": saved_convo.get("title"),
                    "model": saved_convo.get("model"),
                    "last_message": last_m
                }, f, indent=2)
        except Exception as e:
            print(f"Error writing latest work file: {e}")

    return jsonify(saved_convo)

@app.route("/api/conversations/<cid>", methods=["DELETE"])
def delete_conversation(cid):
    convos = load_conversations_data()
    filtered = [c for c in convos if c.get("id") != cid]
    save_conversations_data(filtered)
    return jsonify({"success": True})

@app.route("/api/terminal/run", methods=["POST"])
def run_terminal():
    data = request.get_json(force=True)
    command = data.get("command", "")
    target = data.get("target", "local")
    result = execute_terminal_command(command, target)
    return jsonify({"command": command, "target": target, "result": result})

# --- Active Generation Background Manager ---
ACTIVE_JOBS = {}
ACTIVE_JOBS_LOCK = threading.Lock()

def get_or_create_job(cid):
    with ACTIVE_JOBS_LOCK:
        if cid not in ACTIVE_JOBS:
            ACTIVE_JOBS[cid] = {
                "cid": cid,
                "status": "idle",
                "events": [],
                "subscribers": [],
                "started_at": 0,
                "stop_requested": False
            }
        return ACTIVE_JOBS[cid]

def emit_job_event(cid, event_data):
    with ACTIVE_JOBS_LOCK:
        job = ACTIVE_JOBS.get(cid)
        if not job:
            return
        job["events"].append(event_data)
        dead = []
        for q in job["subscribers"]:
            try:
                q.put_nowait(event_data)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in job["subscribers"]:
                job["subscribers"].remove(q)

def run_generation_worker(cid, model, thinking_level, effective_instruction, formatted_contents, enable_tools):
    job = get_or_create_job(cid)
    with ACTIVE_JOBS_LOCK:
        job["status"] = "running"
        job["events"] = []
        job["started_at"] = time.time()
        job["stop_requested"] = False

    config_args = {
        "system_instruction": effective_instruction,
        "max_output_tokens": 65536
    }
    if enable_tools:
        config_args["tools"] = [types.Tool(function_declarations=WORKSPACE_TOOL_DECLARATIONS)]

    if thinking_level and thinking_level != "off":
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=True
        )
    else:
        config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(**config_args)

    convos = load_conversations_data()
    convo = next((c for c in convos if c.get("id") == cid), None)
    if not convo:
        convo = {
            "id": cid,
            "title": "New Chat",
            "model": model,
            "thinking_level": thinking_level,
            "messages": [],
            "createdAt": time.time(),
            "updatedAt": time.time()
        }
        convos.append(convo)

    msgs = convo.setdefault("messages", [])
    if not msgs or msgs[-1].get("role") != "model" or not msgs[-1].get("is_generating"):
        msgs.append({
            "role": "model",
            "model": model,
            "text": "",
            "thoughts": "",
            "toolCalls": [],
            "is_generating": True,
            "started_at": job["started_at"],
            "duration": None,
            "usage": None
        })
    model_msg = msgs[-1]
    model_msg["is_generating"] = True
    model_msg["started_at"] = job["started_at"]
    convo["updatedAt"] = time.time()
    save_conversations_data(convos)

    total_input_tokens = 0
    total_output_tokens = 0
    current_contents = list(formatted_contents)
    round_count = 0
    last_function_calls = []
    last_save_time = time.time()

    def persist_progress(force=False):
        nonlocal last_save_time
        now = time.time()
        if force or (now - last_save_time >= 1.5):
            all_c = load_conversations_data()
            for c in all_c:
                if c.get("id") == cid and c.get("messages"):
                    c["messages"][-1] = dict(model_msg)
                    c["updatedAt"] = now
                    break
            save_conversations_data(all_c)
            last_save_time = now

    try:
        while True:
            if job.get("stop_requested"):
                break

            round_count += 1
            function_calls_to_execute = []
            accumulated_model_parts = []

            max_retries = 5
            retry_delays = [10, 20, 35, 60, 60]
            turn_succeeded = False

            for attempt in range(max_retries):
                if job.get("stop_requested"):
                    break
                try:
                    function_calls_to_execute = []
                    accumulated_model_parts = []
                    response = client.models.generate_content_stream(
                        model=model,
                        contents=current_contents,
                        config=config
                    )

                    for chunk in response:
                        if job.get("stop_requested"):
                            break
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            cand = chunk.candidates[0]
                            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                                for part in cand.content.parts:
                                    accumulated_model_parts.append(part)
                                    is_thought = getattr(part, "thought", False)
                                    part_text = getattr(part, "text", "") or ""

                                    if part_text:
                                        if is_thought:
                                            model_msg["thoughts"] = (model_msg.get("thoughts") or "") + part_text
                                            emit_job_event(cid, {"type": "thought", "text": part_text})
                                        else:
                                            model_msg["text"] = (model_msg.get("text") or "") + part_text
                                            emit_job_event(cid, {"type": "content", "text": part_text})
                                        persist_progress(force=False)

                                    fn_call = getattr(part, "function_call", None)
                                    if fn_call:
                                        fn_name = fn_call.name
                                        fn_args = dict(fn_call.args) if hasattr(fn_call, "args") else {}
                                        function_calls_to_execute.append((fn_name, fn_args))

                        usage = getattr(chunk, "usage_metadata", None)
                        if usage:
                            if getattr(usage, "prompt_token_count", 0):
                                total_input_tokens = max(total_input_tokens, usage.prompt_token_count)
                            if getattr(usage, "candidates_token_count", 0):
                                total_output_tokens += usage.candidates_token_count

                    turn_succeeded = True
                    break
                except Exception as ex:
                    err_str = str(ex)
                    is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate limit" in err_str.lower()
                    if is_quota and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        notice = f"\n\n> Quota limit encountered (429). Waiting {delay}s for rolling quota to reset...\n\n"
                        model_msg["thoughts"] = (model_msg.get("thoughts") or "") + notice
                        emit_job_event(cid, {"type": "thought", "text": notice})
                        persist_progress(force=True)
                        time.sleep(delay)
                    else:
                        raise

            if not turn_succeeded or job.get("stop_requested"):
                break

            last_function_calls = function_calls_to_execute
            if not function_calls_to_execute:
                break

            tool_response_parts = []
            for fn_name, fn_args in function_calls_to_execute:
                if job.get("stop_requested"):
                    break
                emit_job_event(cid, {"type": "tool_call", "name": fn_name, "args": fn_args})
                exec_result = execute_workspace_tool(fn_name, fn_args, stop_check=lambda: job.get("stop_requested", False))
                if job.get("stop_requested"):
                    break
                preview = exec_result[:400] + ("..." if len(exec_result) > 400 else "")

                # Line range & diff calculation
                start_line = fn_args.get("start_line") or fn_args.get("StartLine")
                end_line = fn_args.get("end_line") or fn_args.get("EndLine")
                target_path = fn_args.get("path", "")
                lines_added = None
                lines_removed = None

                if fn_name == "read_file" and not end_line and target_path:
                    try:
                        line_count = get_file_line_count(target_path)
                        if line_count:
                            start_line = start_line or 1
                            end_line = line_count
                    except Exception:
                        pass
                elif fn_name == "edit_file":
                    t_str = fn_args.get("target", "")
                    r_str = fn_args.get("replacement", "")
                    import difflib
                    t_lines = t_str.splitlines(keepends=True) if t_str else []
                    r_lines = r_str.splitlines(keepends=True) if r_str else []
                    diff = list(difflib.ndiff(t_lines, r_lines))
                    lines_added = sum(1 for d in diff if d.startswith('+ '))
                    lines_removed = sum(1 for d in diff if d.startswith('- '))
                elif fn_name == "write_file":
                    c_str = fn_args.get("content", "")
                    lines_added = len(c_str.splitlines()) if c_str else 0
                    lines_removed = 0

                tc_item = {
                    "name": fn_name,
                    "args": fn_args,
                    "result_preview": preview,
                    "output": exec_result[:8000] if fn_name == "run_command" else preview,
                    "path": target_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines_added": lines_added,
                    "lines_removed": lines_removed
                }

                emit_job_event(cid, {
                    "type": "tool_result",
                    "name": fn_name,
                    "result_preview": preview,
                    "output": exec_result[:8000] if fn_name == "run_command" else preview,
                    "full_length": len(exec_result),
                    "path": target_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines_added": lines_added,
                    "lines_removed": lines_removed
                })

                model_msg.setdefault("toolCalls", []).append(tc_item)
                persist_progress(force=True)

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": exec_result}
                    )
                )

            current_contents.append(types.Content(role="model", parts=accumulated_model_parts))
            current_contents.append(types.Content(role="user", parts=tool_response_parts))

        # Synthesis fallback if needed (if tool loop ended without final response text)
        if last_function_calls and not model_msg.get("text") and not job.get("stop_requested"):
            current_contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="Synthesize everything you found and accomplished, answer the user thoroughly, and output your complete response and any code changes.")]
            ))
            synthesis_config_args = dict(config_args)
            synthesis_config_args.pop("tools", None)
            synthesis_config = types.GenerateContentConfig(**synthesis_config_args)

            for s_attempt in range(max_retries):
                if job.get("stop_requested"):
                    break
                try:
                    synth_stream = client.models.generate_content_stream(
                        model=model,
                        contents=current_contents,
                        config=synthesis_config
                    )
                    for chunk in synth_stream:
                        if job.get("stop_requested"):
                            break
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            cand = chunk.candidates[0]
                            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                                for part in cand.content.parts:
                                    is_thought = getattr(part, "thought", False)
                                    part_text = getattr(part, "text", "") or ""
                                    if part_text:
                                        if is_thought:
                                            model_msg["thoughts"] = (model_msg.get("thoughts") or "") + part_text
                                            emit_job_event(cid, {"type": "thought", "text": part_text})
                                        else:
                                            model_msg["text"] = (model_msg.get("text") or "") + part_text
                                            emit_job_event(cid, {"type": "content", "text": part_text})
                                        persist_progress(force=False)
                        usage = getattr(chunk, "usage_metadata", None)
                        if usage:
                            if getattr(usage, "prompt_token_count", 0):
                                total_input_tokens = max(total_input_tokens, usage.prompt_token_count)
                            if getattr(usage, "candidates_token_count", 0):
                                total_output_tokens += usage.candidates_token_count
                    break
                except Exception as s_ex:
                    err_str = str(s_ex)
                    is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate limit" in err_str.lower()
                    if is_quota and s_attempt < max_retries - 1:
                        s_delay = retry_delays[s_attempt]
                        notice = f"\n\n> Synthesis quota reached (429). Waiting {s_delay}s for quota to reset...\n\n"
                        model_msg["thoughts"] = (model_msg.get("thoughts") or "") + notice
                        emit_job_event(cid, {"type": "thought", "text": notice})
                        persist_progress(force=True)
                        time.sleep(s_delay)
                    else:
                        raise

        cost_usd = ((total_input_tokens * 2.00) + (total_output_tokens * 10.00)) / 1_000_000
        cost_eur = cost_usd * 0.93
        duration = round(time.time() - job["started_at"], 1)

        done_payload = {
            "type": "done",
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": round(cost_usd, 5),
            "cost_eur": round(cost_eur, 5),
            "duration": duration
        }
        model_msg["is_generating"] = False
        model_msg["duration"] = duration
        model_msg["usage"] = done_payload
        persist_progress(force=True)

        emit_job_event(cid, done_payload)

        # Export latest work
        try:
            with open(LATEST_WORK_FILE, "w", encoding="utf-8") as f:
                f.write(f"# Latest Argon/Carbon Work ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                f.write(f"**Chat:** {convo.get('title')}\n")
                f.write(f"**Model:** {model}\n\n")
                f.write(f"### Last Response\n\n{model_msg.get('text', '')}\n")
            with open(LATEST_CONTEXT_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "updatedAt": time.time(),
                    "conversation_id": cid,
                    "title": convo.get("title"),
                    "model": model,
                    "last_message": model_msg
                }, f, indent=2)
        except Exception as e:
            print(f"Error exporting latest work: {e}")

    except Exception as e:
        err_msg = str(e)
        model_msg["is_generating"] = False
        model_msg["text"] = (model_msg.get("text") or "") + f"\n\n[Error: {err_msg}]"
        persist_progress(force=True)
        emit_job_event(cid, {"type": "error", "message": err_msg})
    finally:
        with ACTIVE_JOBS_LOCK:
            job["status"] = "idle"
            job["stop_requested"] = False

# --- Chat & Agentic Function Calling Endpoints ---

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    cid = data.get("conversation_id")
    if not cid:
        cid = str(uuid.uuid4())

    model = data.get("model", "gemini-4-argon")
    thinking_level = data.get("thinking_level", "high")
    system_instruction = data.get("system_instruction")
    raw_contents = data.get("contents", [])
    enable_tools = data.get("enable_tools", True)

    if not raw_contents:
        return jsonify({"error": "No messages provided"}), 400

    base_instruction = get_system_instruction()
    effective_instruction = f"{base_instruction}\n\n{system_instruction}" if system_instruction else base_instruction

    formatted_contents = []
    for msg in raw_contents:
        role = msg.get("role", "user")
        if role not in ("user", "model"):
            role = "user"

        text = msg.get("text", "") or ""

        # Strip any hallucinated repetition from prior contaminated turns
        if role == "model" and "Workspace actions completed in this step:" in text:
            text = text.split("Workspace actions completed in this step:")[0].strip()

        # If a model turn has no text (e.g. was interrupted), skip it so user messages flow naturally
        if role == "model" and not text:
            continue

        parts = []
        if text:
            parts.append(types.Part.from_text(text=text))
        for f in msg.get("files", []):
            fname = f.get("name", "attachment")
            mime = f.get("mime", "application/octet-stream")
            b64_data = f.get("data")
            ftext = f.get("text")
            if b64_data:
                try:
                    parts.append(types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime))
                except Exception:
                    pass
            elif ftext is not None:
                parts.append(types.Part.from_text(text=f"\n--- Attached file: {fname} ---\n{ftext}\n--- End of {fname} ---\n"))

        if parts:
            if formatted_contents and formatted_contents[-1].role == role:
                formatted_contents[-1].parts.extend(parts)
            else:
                formatted_contents.append(types.Content(role=role, parts=parts))

    job = get_or_create_job(cid)

    # Stop any previous generation and clear old events synchronously so previous turn is never replayed!
    with ACTIVE_JOBS_LOCK:
        job["stop_requested"] = True
        job["status"] = "idle"
        for q in list(job.get("subscribers", [])):
            try:
                q.put_nowait({"type": "error", "message": "Starting new turn."})
            except Exception:
                pass
        job["subscribers"].clear()
        job["events"] = []  # Cleared synchronously so stream_subscriber sees an empty event list!
        job["status"] = "running"
        job["stop_requested"] = False
        job["started_at"] = time.time()

    worker_thread = threading.Thread(
        target=run_generation_worker,
        args=(cid, model, thinking_level, effective_instruction, formatted_contents, enable_tools),
        daemon=True
    )
    worker_thread.start()

    def stream_subscriber():
        q = queue.Queue()
        with ACTIVE_JOBS_LOCK:
            existing_events = list(job["events"])
            job["subscribers"].append(q)

        for ev in existing_events:
            yield f"data: {json.dumps(ev)}\n\n"

        try:
            while True:
                try:
                    ev = q.get(timeout=1.5)
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    with ACTIVE_JOBS_LOCK:
                        still_running = (job["status"] == "running")
                    if not still_running and q.empty():
                        break
                    yield ": ping\n\n"
        finally:
            with ACTIVE_JOBS_LOCK:
                if q in job["subscribers"]:
                    job["subscribers"].remove(q)

    return Response(
        stream_subscriber(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@app.route("/api/chat/stream", methods=["GET"])
def stream_existing_chat():
    cid = request.args.get("cid")
    if not cid:
        return jsonify({"error": "Missing cid"}), 400

    job = get_or_create_job(cid)

    def stream_subscriber():
        q = queue.Queue()
        with ACTIVE_JOBS_LOCK:
            existing_events = list(job["events"])
            job["subscribers"].append(q)

        for ev in existing_events:
            yield f"data: {json.dumps(ev)}\n\n"

        try:
            while True:
                try:
                    ev = q.get(timeout=1.5)
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    with ACTIVE_JOBS_LOCK:
                        still_running = (job["status"] == "running")
                    if not still_running and q.empty():
                        break
                    yield ": ping\n\n"
        finally:
            with ACTIVE_JOBS_LOCK:
                if q in job["subscribers"]:
                    job["subscribers"].remove(q)

    return Response(
        stream_subscriber(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@app.route("/api/chat/abort", methods=["POST"])
def abort_chat():
    data = request.get_json(force=True)
    cid = data.get("conversation_id")
    if cid:
        with ACTIVE_JOBS_LOCK:
            job = ACTIVE_JOBS.get(cid)
            if job:
                job["stop_requested"] = True
                job["status"] = "idle"
                for q in list(job.get("subscribers", [])):
                    try:
                        q.put_nowait({"type": "error", "message": "Stopped by user."})
                    except Exception:
                        pass
    return jsonify({"success": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4321))
    print(f"=====================================================")
    print(f"  Gemini 4 Argon & Carbon Studio Server Running")
    print(f"  URL: http://localhost:{port}")
    print(f"  Auth Mode: {initial_settings.get('auth_mode')}")
    print(f"  Project: {initial_settings.get('gcp_project')}")
    print(f"  Workspace: {WORKSPACE_ROOT} (Mounted: {os.path.exists(WORKSPACE_ROOT)})")
    print(f"=====================================================")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
