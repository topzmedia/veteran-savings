"""
AI Ad Agency — Web Dashboard
Run with: python -m uvicorn ai_ad_agency.web.app:app --reload --port 8000
Then open: http://localhost:8000
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file so OPENAI_API_KEY etc. are available as env vars
try:
    from dotenv import load_dotenv
    # Search upward from this file for a .env
    _env_candidates = [
        Path(__file__).parent.parent.parent / ".env",  # project root
        Path(__file__).parent.parent / ".env",          # ai_ad_agency/
        Path.home() / "aiagency" / ".env",              # ~/aiagency/.env
        Path.home() / ".env",
    ]
    for _p in _env_candidates:
        if _p.exists():
            load_dotenv(_p, override=False)
            break
except ImportError:
    pass

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # ai_ad_agency/
CONFIGS_DIR = BASE_DIR / "configs"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

ROOT_PATH = os.environ.get("ROOT_PATH", "")

app = FastAPI(title="AI Ad Agency", root_path=ROOT_PATH, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# In-memory job tracker
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _job_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

def _run_generation(job_id: str, config_name: str, pipeline: str, count: int) -> None:
    config_path = CONFIGS_DIR / config_name
    output_dir = OUTPUTS_DIR / job_id

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["log"] = []

    def log(msg: str) -> None:
        with _jobs_lock:
            _jobs[job_id]["log"].append(msg)

    log(f"Starting {pipeline} generation — count={count}")
    log(f"Offer: {config_name.replace('.json','').replace('_',' ').title()}")

    try:
        sys.path.insert(0, str(BASE_DIR.parent))
        from ai_ad_agency.utils.config import load_app_config
        from ai_ad_agency.models.schemas import OfferConfig

        app_config = load_app_config(str(BASE_DIR / "configs" / "app_config.json"))
        with open(config_path) as f:
            offer_data = json.load(f)
        offer = OfferConfig(**offer_data)
        output_dir.mkdir(parents=True, exist_ok=True)

        hooks = []

        # ── Hooks ──────────────────────────────────────────────────────────
        if pipeline in ("hooks", "all"):
            log("Generating hooks...")
            from ai_ad_agency.providers.llm_provider import build_llm_provider
            from ai_ad_agency.agents.hook_agent import run_hook_agent
            llm = build_llm_provider(app_config.providers.llm)
            hooks = run_hook_agent(
                llm, offer,
                total_hooks=min(count, 30),
                output_dir=str(output_dir / "hooks"),
            )
            log(f"✓ Generated {len(hooks)} hooks")

        # ── Scripts ────────────────────────────────────────────────────────
        if pipeline in ("scripts", "all"):
            log("Generating scripts...")
            from ai_ad_agency.providers.llm_provider import build_llm_provider
            from ai_ad_agency.agents.script_agent import run_script_agent
            llm = build_llm_provider(app_config.providers.llm)
            if not hooks:
                from ai_ad_agency.agents.hook_agent import run_hook_agent
                hooks = run_hook_agent(llm, offer, total_hooks=5, output_dir=str(output_dir / "hooks"))
            scripts = run_script_agent(
                llm, hooks, offer,
                scripts_per_hook=2,
                output_dir=str(output_dir / "scripts"),
            )
            log(f"✓ Generated {len(scripts)} scripts")

        # ── Images ─────────────────────────────────────────────────────────
        if pipeline in ("images", "all"):
            log("Generating images...")
            from ai_ad_agency.providers.image_provider import build_image_provider
            from ai_ad_agency.agents.image_agent import ImageAgent
            img_provider = build_image_provider(app_config.providers.image)
            agent = ImageAgent(app_config, img_provider)
            images = agent.generate_batch(
                offer,
                count=min(count, 20),
                output_dir=str(output_dir / "images"),
            )
            log(f"✓ Generated {len(images)} images")

        # ── Video / Talking Actors ──────────────────────────────────────────
        if pipeline in ("video", "all"):
            log("Generating video scripts & actor jobs...")
            from ai_ad_agency.providers.llm_provider import build_llm_provider
            from ai_ad_agency.agents.hook_agent import run_hook_agent
            from ai_ad_agency.agents.script_agent import run_script_agent
            from ai_ad_agency.agents.avatar_catalog_agent import AvatarCatalogAgent
            from ai_ad_agency.agents.talking_actor_agent import TalkingActorAgent
            from ai_ad_agency.providers.avatar_provider import build_avatar_provider

            llm = build_llm_provider(app_config.providers.llm)
            if not hooks:
                hooks = run_hook_agent(llm, offer, total_hooks=3, output_dir=str(output_dir / "hooks"))
            scripts = run_script_agent(llm, hooks, offer, scripts_per_hook=1, output_dir=str(output_dir / "scripts"))
            log(f"✓ Generated {len(scripts)} video scripts")

            avatar_provider = build_avatar_provider(app_config.providers.avatar)
            catalog = AvatarCatalogAgent(app_config, avatar_provider)
            catalog.sync()
            avatars = catalog.select_diverse_batch(count=min(count, 5))
            avatar_ids = [a.avatar_id for a in avatars]
            log(f"✓ Selected {len(avatar_ids)} AI actors")

            actor_agent = TalkingActorAgent(app_config, avatar_provider, catalog)
            jobs_list = actor_agent.create_jobs_from_scripts(scripts, avatar_ids)
            rendered = actor_agent.generate_batch(jobs_list, output_dir=str(output_dir / "videos"))
            done = sum(1 for j in rendered if str(j.render_status) in ("completed", "RenderStatus.COMPLETED"))
            log(f"✓ Rendered {done}/{len(rendered)} talking actor videos")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_dir"] = str(output_dir)
        log("All done! Click 'View Files' to see your outputs.")

    except Exception as exc:
        import traceback
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
        log(f"Error: {exc}")
        log(traceback.format_exc().split('\n')[-2])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    configs = sorted([f.name for f in CONFIGS_DIR.glob("*.json") if f.name != "app_config.json"])
    jobs = []
    with _jobs_lock:
        for jid, jdata in sorted(_jobs.items(), reverse=True):
            jobs.append({"id": jid, **jdata})
    return templates.TemplateResponse("index.html", {
        "request": request,
        "configs": configs,
        "jobs": jobs,
    })


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    config: str = Form(...),
    pipeline: str = Form(...),
    count: int = Form(10),
):
    job_id = _job_id()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "config": config,
            "pipeline": pipeline,
            "count": count,
            "log": [],
            "output_dir": None,
            "created_at": datetime.utcnow().isoformat(),
        }
    background_tasks.add_task(_run_generation, job_id, config, pipeline, count)
    return JSONResponse({"job_id": job_id})


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job)


@app.get("/api/job/{job_id}/files")
async def list_outputs(job_id: str):
    out_dir = OUTPUTS_DIR / job_id
    if not out_dir.exists():
        return JSONResponse({"files": []})
    files = []
    for f in sorted(out_dir.rglob("*")):
        if f.is_file() and not f.name.endswith((".pyc", ".pyo")):
            rel = f.relative_to(OUTPUTS_DIR)
            files.append({
                "name": f.name,
                "path": str(rel),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "ext": f.suffix.lower(),
                "category": f.parent.name,
            })
    return JSONResponse({"files": files})


@app.get("/api/job/{job_id}/content")
async def job_content(job_id: str):
    """Return hook and script text for inline display."""
    out_dir = OUTPUTS_DIR / job_id
    if not out_dir.exists():
        return JSONResponse({"hooks": [], "scripts": []})
    hooks = []
    scripts = []

    hooks_dir = out_dir / "hooks"
    if hooks_dir.exists():
        for f in sorted(hooks_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", item.get("hook_text", str(item))) if isinstance(item, dict) else str(item)
                        hooks.append({"text": text, "file": f.name})
                elif isinstance(data, dict):
                    text = data.get("text", data.get("hook_text", str(data)))
                    hooks.append({"text": text, "file": f.name})
            except Exception:
                pass

    scripts_dir = out_dir / "scripts"
    if scripts_dir.exists():
        for f in sorted(scripts_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                scripts.append({"data": data, "file": f.name})
            except Exception:
                pass

    return JSONResponse({"hooks": hooks, "scripts": scripts})


@app.get("/download/{path:path}")
async def download(path: str):
    file_path = OUTPUTS_DIR / path
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(file_path), filename=file_path.name)
