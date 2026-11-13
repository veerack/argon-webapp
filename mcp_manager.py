import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MCP_CONFIG_FILE = DATA_DIR / "mcp_config.json"

def sanitize_schema_for_gemini(schema):
    """
    Recursively sanitize JSON Schema for Google GenAI SDK:
    - Flatten oneOf / anyOf to the primary branch
    - Convert enum values to strings
    - Strip unsupported schema keywords ($schema, $id, definitions)
    - Ensure type='object' has properties dict
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    res = {}
    for k, v in schema.items():
        if k in ("$schema", "$id", "definitions", "$defs"):
            continue
        if k in ("oneOf", "anyOf") and isinstance(v, list) and v:
            primary = sanitize_schema_for_gemini(v[0])
            for pk, pv in primary.items():
                if pk not in res:
                    res[pk] = pv
            continue
        if k == "enum" and isinstance(v, list):
            res["enum"] = [str(x) for x in v]
            continue
        if k == "properties" and isinstance(v, dict):
            res["properties"] = {pk: sanitize_schema_for_gemini(pv) for pk, pv in v.items()}
            continue
        if k == "items" and isinstance(v, dict):
            res["items"] = sanitize_schema_for_gemini(v)
            continue
        res[k] = v

    if "type" not in res:
        res["type"] = "object"
    if res["type"] == "object" and "properties" not in res:
        res["properties"] = {}
    return res

class MCPServerInstance:
    def __init__(self, server_id, config):
        self.server_id = server_id
        self.config = config
        self.process = None
        self.lock = threading.Lock()
        self.req_id = 0
        self.tools = []
        self.initialized = False
        self.last_error = None

    def _start_process(self):
        if self.process and self.process.poll() is None:
            return True

        command = self.config.get("command", "node")
        args = self.config.get("args", [])
        env = dict(os.environ)
        if self.config.get("env"):
            env.update(self.config["env"])

        cmd_list = [command] + list(args)
        cwd = self.config.get("cwd") or (str(Path(args[0]).parent) if args else None)

        try:
            self.process = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
                bufsize=1
            )
            self.initialized = False
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = f"Failed to start process: {str(e)}"
            return False

    def _send_rpc(self, method, params=None, timeout=20):
        if not self._start_process():
            raise RuntimeError(self.last_error or "Process not running")

        with self.lock:
            self.req_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self.req_id,
                "method": method
            }
            if params is not None:
                payload["params"] = params

            msg = json.dumps(payload) + "\n"
            try:
                self.process.stdin.write(msg)
                self.process.stdin.flush()
            except Exception as e:
                self.process = None
                raise RuntimeError(f"Error writing to MCP server stdin: {e}")

            # Read response
            start_t = time.time()
            while True:
                if time.time() - start_t > timeout:
                    raise TimeoutError(f"MCP RPC {method} timed out after {timeout}s")

                line = self.process.stdout.readline()
                if not line:
                    exit_code = self.process.poll()
                    raise RuntimeError(f"MCP server process exited unexpectedly (code {exit_code})")

                line = line.strip()
                if not line:
                    continue

                try:
                    resp = json.loads(line)
                    if resp.get("id") == self.req_id:
                        if "error" in resp:
                            raise RuntimeError(f"MCP Error: {resp['error']}")
                        return resp.get("result", {})
                except json.JSONDecodeError:
                    continue

    def initialize(self):
        if self.initialized and self.process and self.process.poll() is None:
            return True

        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "gemini-4-studio",
                "version": "1.0.0"
            }
        }
        res = self._send_rpc("initialize", init_params)
        self.initialized = True

        # Send notifications/initialized if needed
        try:
            notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            self.process.stdin.write(notif)
            self.process.stdin.flush()
        except Exception:
            pass

        return True

    def list_tools(self, force_refresh=False):
        if self.tools and not force_refresh:
            return self.tools

        try:
            self.initialize()
            res = self._send_rpc("tools/list", {})
            self.tools = res.get("tools", [])
            return self.tools
        except Exception as e:
            self.last_error = str(e)
            return []

    def call_tool(self, tool_name, arguments):
        try:
            self.initialize()
            res = self._send_rpc("tools/call", {"name": tool_name, "arguments": arguments or {}}, timeout=45)
            content = res.get("content", [])
            output_parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        output_parts.append(c.get("text", ""))
                    elif c.get("type") == "image":
                        output_parts.append(f"[Image content: {c.get('mimeType', 'image/png')}]")
                    else:
                        output_parts.append(str(c))
                else:
                    output_parts.append(str(c))
            result_text = "\n".join(output_parts)
            if res.get("isError"):
                return f"MCP Tool Error ({tool_name}):\n{result_text}"
            return result_text if result_text.strip() else "(Tool completed with no output)"
        except Exception as e:
            return f"Error executing MCP tool '{tool_name}': {str(e)}"

    def close(self):
        with self.lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                self.process = None
                self.initialized = False

class MCPManager:
    def __init__(self):
        self.instances = {}
        self.lock = threading.Lock()
        self.tool_to_server_map = {}
        self.active_declarations = []
        self._load_or_discover_config()

    def _discover_local_servers(self):
        discovered = {}

        # 1. Blockbench MCP in C:\Users\pisto\mcps\blockbench-mcp
        home = Path.home()
        bb_dir = home / "mcps" / "blockbench-mcp"
        bb_dist = bb_dir / "dist" / "index.js"
        if bb_dist.exists():
            discovered["blockbench"] = {
                "name": "Blockbench 3D",
                "description": "Blockbench live 3D modeling, cube creation, texturing, rigging, and animations.",
                "command": "node",
                "args": [str(bb_dist).replace("\\", "/")],
                "env": {"BLOCKBENCH_MCP_PORT": "8787"},
                "cwd": str(bb_dir).replace("\\", "/"),
                "enabled": True
            }

        # 2. Check other folders in ~/mcps
        mcps_root = home / "mcps"
        if mcps_root.exists():
            for sub in mcps_root.iterdir():
                if sub.is_dir() and sub.name not in ("blockbench-mcp",):
                    mcp_json = sub / ".mcp.json"
                    if mcp_json.exists():
                        try:
                            with open(mcp_json, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                            for sname, sdata in cfg.get("mcpServers", {}).items():
                                if sname not in discovered:
                                    discovered[sname] = {
                                        "name": sname,
                                        "description": f"Local MCP server in {sub.name}",
                                        "command": sdata.get("command", "node"),
                                        "args": sdata.get("args", []),
                                        "env": sdata.get("env", {}),
                                        "cwd": str(sub).replace("\\", "/"),
                                        "enabled": True
                                    }
                        except Exception:
                            pass

        # 3. Check ~/.gemini/config/mcp_config.json
        gemini_mcp = home / ".gemini" / "config" / "mcp_config.json"
        if gemini_mcp.exists():
            try:
                with open(gemini_mcp, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                for sname, sdata in cfg.get("mcpServers", {}).items():
                    if sname not in discovered and sname != "rea": # Exclude rea if not needed or keep optional
                        discovered[sname] = {
                            "name": sname,
                            "description": f"Gemini configured MCP server ({sname})",
                            "command": sdata.get("command", "node"),
                            "args": sdata.get("args", []),
                            "env": sdata.get("env", {}),
                            "enabled": False
                        }
            except Exception:
                pass

        return discovered

    def _load_or_discover_config(self):
        discovered = self._discover_local_servers()
        saved = {}
        if MCP_CONFIG_FILE.exists():
            try:
                with open(MCP_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f).get("mcpServers", {})
            except Exception:
                saved = {}

        merged = {}
        # Merge discovered servers with saved enabled states
        for sid, sdata in discovered.items():
            merged[sid] = dict(sdata)
            if sid in saved and "enabled" in saved[sid]:
                merged[sid]["enabled"] = saved[sid]["enabled"]

        for sid, sdata in saved.items():
            if sid not in merged:
                merged[sid] = dict(sdata)

        self._save_config(merged)
        self._sync_instances(merged)

    def _save_config(self, servers_dict):
        try:
            with open(MCP_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": servers_dict}, f, indent=2)
        except Exception as e:
            print(f"Error saving MCP config: {e}")

    def _sync_instances(self, servers_dict):
        with self.lock:
            # Remove instances no longer configured
            for sid in list(self.instances.keys()):
                if sid not in servers_dict:
                    self.instances[sid].close()
                    del self.instances[sid]

            # Update or create instances
            for sid, sdata in servers_dict.items():
                if sid not in self.instances:
                    self.instances[sid] = MCPServerInstance(sid, sdata)
                else:
                    self.instances[sid].config = sdata

    def get_servers_status(self):
        res = []
        for sid, inst in self.instances.items():
            is_enabled = inst.config.get("enabled", False)
            tools_count = len(inst.tools) if inst.tools else 0
            if is_enabled and not inst.tools:
                try:
                    tools = inst.list_tools()
                    tools_count = len(tools)
                except Exception:
                    pass
            res.append({
                "id": sid,
                "name": inst.config.get("name", sid),
                "description": inst.config.get("description", ""),
                "enabled": is_enabled,
                "tools_count": tools_count,
                "error": inst.last_error
            })
        return res

    def toggle_server(self, server_id, enabled=None):
        if server_id not in self.instances:
            return False, f"Server '{server_id}' not found."

        inst = self.instances[server_id]
        if enabled is None:
            new_state = not inst.config.get("enabled", False)
        else:
            new_state = bool(enabled)

        inst.config["enabled"] = new_state
        if not new_state:
            inst.close()

        # Update config file
        all_configs = {sid: i.config for sid, i in self.instances.items()}
        self._save_config(all_configs)
        self.refresh_tools_mapping()
        return True, new_state

    def rescan(self):
        self._load_or_discover_config()
        self.refresh_tools_mapping()
        return self.get_servers_status()

    def refresh_tools_mapping(self):
        with self.lock:
            self.tool_to_server_map = {}
            for sid, inst in self.instances.items():
                if inst.config.get("enabled", False):
                    try:
                        tools = inst.list_tools()
                        for t in tools:
                            tname = t["name"]
                            # Map original name
                            self.tool_to_server_map[tname] = (sid, t)
                            # Also map with server prefix: e.g. blockbench_add_cube
                            prefixed = f"{sid}_{tname}"
                            self.tool_to_server_map[prefixed] = (sid, t)
                    except Exception as e:
                        print(f"Error fetching tools for MCP server {sid}: {e}")

    def get_gemini_declarations(self):
        from google.genai import types
        self.refresh_tools_mapping()
        declarations = []
        seen = set()

        for tname, (sid, tdata) in self.tool_to_server_map.items():
            # Use original tool name as primary
            orig_name = tdata["name"]
            if orig_name in seen:
                continue
            seen.add(orig_name)

            desc = tdata.get("description", "") or orig_name
            # Add server attribution
            server_name = self.instances[sid].config.get("name", sid)
            full_desc = f"[{server_name} MCP Tool] {desc}"
            schema = sanitize_schema_for_gemini(tdata.get("inputSchema", {}))

            try:
                decl = types.FunctionDeclaration(
                    name=orig_name,
                    description=full_desc[:1024],
                    parameters=schema
                )
                declarations.append(decl)
            except Exception as e:
                print(f"Error creating FunctionDeclaration for MCP tool '{orig_name}': {e}")

        return declarations

    def has_tool(self, tool_name):
        return tool_name in self.tool_to_server_map

    def call_tool(self, tool_name, arguments):
        if tool_name not in self.tool_to_server_map:
            return f"Error: MCP tool '{tool_name}' is not registered."

        sid, tdata = self.tool_to_server_map[tool_name]
        orig_name = tdata["name"]
        inst = self.instances.get(sid)
        if not inst:
            return f"Error: MCP server '{sid}' is not running."

        return inst.call_tool(orig_name, arguments)

# Global singleton
mcp_mgr = MCPManager()
