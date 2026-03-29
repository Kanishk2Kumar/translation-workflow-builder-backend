import hashlib
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from db import get_pool
from executor import execute_workflow

router = APIRouter(prefix="/workflow", tags=["workflow"])


class RunWorkflowResponse(BaseModel):
    execution_id: str
    status: str
    output: dict
    logs: list
    cache_hit: bool = False


# ─── Text extraction ──────────────────────────────────────────────────────────

def extract_text(filename: str, contents: bytes) -> str:
    if filename.endswith(".txt"):
        return contents.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(contents))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="PDF support requires pypdf. Run: pip install pypdf",
            )

    if filename.endswith(".docx"):
        try:
            import io
            import docx
            doc = docx.Document(io.BytesIO(contents))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="DOCX support requires python-docx. Run: pip install python-docx",
            )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {filename}. Supported: .txt, .pdf, .docx",
    )


# ─── Document hash helpers ────────────────────────────────────────────────────

def compute_hash(contents: bytes, target_language: str) -> str:
    """
    Hash = SHA256(file bytes + target language).
    Same file in different target languages = different cache entries.
    """
    h = hashlib.sha256()
    h.update(contents)
    h.update(target_language.encode())
    return h.hexdigest()


async def get_cached_execution(
    pool, document_hash: str, workflow_id: str
) -> dict | None:
    """
    Look up a previous successful execution with the same document hash
    for the same workflow. Returns the output payload or None.
    """
    row = await pool.fetchrow(
        """
        SELECT id, output
        FROM executions
        WHERE workflow_id = $1
          AND status = 'success'
          AND input::jsonb->>'document_hash' = $2
        ORDER BY started_at DESC
        LIMIT 1
        """,
        workflow_id,
        document_hash,
    )
    if not row or not row["output"]:
        return None
    return {
        "execution_id": str(row["id"]),
        "output": json.loads(row["output"]) if isinstance(row["output"], str) else row["output"],
    }


# ─── POST /workflow/{workflow_id}/run ─────────────────────────────────────────

@router.post("/{workflow_id}/run", response_model=RunWorkflowResponse)
async def run_workflow(
    workflow_id: str,
    file: UploadFile = File(...),
    target_language: str = Form(default="hi"),
):
    pool = get_pool()
    contents = await file.read()

    # 1. Compute document hash
    document_hash = compute_hash(contents, target_language)

    # 2. Check cache — same document + same language + same workflow?
    cached = await get_cached_execution(pool, document_hash, workflow_id)
    if cached:
        return RunWorkflowResponse(
            execution_id=cached["execution_id"],
            status="success",
            output=cached["output"],
            logs=[],
            cache_hit=True,
        )

    # 3. Extract text from file
    raw_text = extract_text(file.filename, contents)

    # 4. Fetch workflow from DB
    row = await pool.fetchrow(
        "SELECT id, nodes, edges FROM workflows WHERE id = $1",
        workflow_id,
    )
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found",
        )

    nodes = json.loads(row["nodes"]) if isinstance(row["nodes"], str) else row["nodes"]
    edges = json.loads(row["edges"]) if isinstance(row["edges"], str) else row["edges"]

    if not nodes:
        raise HTTPException(status_code=400, detail="Workflow has no nodes")

    # 5. Create execution record — store hash in input for future cache lookups
    execution_id = str(uuid.uuid4())
    await pool.execute(
        """
        INSERT INTO executions (id, workflow_id, status, input, started_at)
        VALUES ($1, $2, 'running', $3, $4)
        """,
        execution_id,
        workflow_id,
        json.dumps({
            "filename": file.filename,
            "target_language": target_language,
            "document_hash": document_hash,
        }),
        datetime.utcnow(),
    )

    # 6. Build initial context
    initial_context = {
        "raw_text": raw_text,
        "target_language": target_language,
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "source_filename": file.filename,
        "document_hash": document_hash,
    }

    # 7. Execute workflow
    try:
        final_context = await execute_workflow(
            nodes=nodes,
            edges=edges,
            initial_context=initial_context,
        )
    except Exception as e:
        await pool.execute(
            """
            UPDATE executions
            SET status = 'failed', logs = $1, completed_at = $2
            WHERE id = $3
            """,
            json.dumps([{"error": str(e)}]),
            datetime.utcnow(),
            execution_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}",
        )

    final_output = final_context.get("final_output", {})
    logs = final_context.get("_logs", [])

    return RunWorkflowResponse(
        execution_id=execution_id,
        status="success",
        output=final_output,
        logs=logs,
        cache_hit=False,
    )


# ─── GET /execution/{execution_id}/status ────────────────────────────────────

@router.get("/execution/{execution_id}/status")
async def get_execution_status(execution_id: str):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, status, input, output, logs, started_at, completed_at
        FROM executions
        WHERE id = $1
        """,
        execution_id,
    )
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Execution '{execution_id}' not found",
        )

    return {
        "execution_id": str(row["id"]),
        "status": row["status"],
        "input": json.loads(row["input"]) if row["input"] else None,
        "output": json.loads(row["output"]) if row["output"] else None,
        "logs": json.loads(row["logs"]) if row["logs"] else [],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }