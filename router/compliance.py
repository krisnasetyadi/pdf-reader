# router/compliance.py
"""
Generic "Reference Framework Gap Analysis" endpoints.

Skill-agnostic by design: `skill_id` selects the behavior/output schema.
"compliance_gap_check" (Skill 1) is validated first via ISO 27001/9001 —
ISO is just the first `framework_name` a caller happens to use, nothing
here is ISO-specific. "scenario_regulatory_impact" (Skill 2) is scaffolded
so the endpoint/schema shape exists, but is NOT validated for real
decisions yet (see run_gap_analysis below).
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from typing import List, Optional
import asyncio
import logging
import uuid

from models import GapAnalysisRequest, GapAnalysisResponse, GapAnalysisRun, GapAnalysisItem
from processor import processor
import storage as supabase_storage
from router.auth import get_current_user, UserRecord

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_SKILLS = {"compliance_gap_check", "scenario_regulatory_impact"}

SCENARIO_SCAFFOLD_DISCLAIMER = (
    "Skill scenario_regulatory_impact masih scaffold (belum divalidasi untuk keputusan nyata). "
    "Ini BUKAN nasihat pajak/hukum resmi — konsultasikan ke konsultan pajak/akuntan bersertifikat "
    "sebelum mengambil keputusan berdasarkan hasil ini."
)


def _summarize(items: List[dict]) -> dict:
    summary = {"met": 0, "partial": 0, "not_met": 0, "unknown": 0}
    for item in items:
        status = item.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _can_access_run(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own runs."""
    if user.role == "admin":
        return True
    if not row:
        return False
    return row.get("owner_id") == user.user_id


@router.post("/analysis/gap-analysis", response_model=GapAnalysisResponse)
async def run_gap_analysis(
    body: GapAnalysisRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Run a Skill against reference (+ optional target) collections and
    persist the result as a run. Generic across skills — not ISO-specific."""
    if body.skill_id not in SUPPORTED_SKILLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown skill_id. Supported: {sorted(SUPPORTED_SKILLS)}",
        )
    if not body.reference_collection_ids:
        raise HTTPException(status_code=400, detail="reference_collection_ids is required")

    # Scope collection access — admins can reach every PDF collection;
    # everyone else only their own (same rule router/agnostic.py enforces
    # for hybrid search). Without this, a non-admin could point
    # reference/target ids at another user's private collection and read
    # its content back out via the "evidence" quotes in the response.
    is_admin = user.role == "admin"
    allowed_pdf_ids = set(await asyncio.to_thread(
        supabase_storage.list_collection_ids_for_user, user.user_id, is_admin
    ))
    requested_ids = set(body.reference_collection_ids)
    if body.target_collection_id:
        requested_ids.add(body.target_collection_id)
    if not requested_ids.issubset(allowed_pdf_ids):
        raise HTTPException(status_code=403, detail="One or more collections are not accessible to you")

    run_id = str(uuid.uuid4())
    items: List[dict] = []

    if body.skill_id == "compliance_gap_check":
        if not body.target_collection_id:
            raise HTTPException(
                status_code=400,
                detail="target_collection_id is required for compliance_gap_check",
            )
        # run_compliance_gap_check makes several sequential blocking LLM
        # calls — off the event loop, or one gap-analysis run freezes every
        # other concurrent request (health checks included) until it finishes.
        items, disclaimer = await asyncio.to_thread(
            processor.run_compliance_gap_check,
            reference_collection_ids=body.reference_collection_ids,
            target_collection_id=body.target_collection_id,
            framework_name=body.framework_name,
        )
    else:
        # scenario_regulatory_impact — scaffold only. The endpoint works
        # end-to-end (request validated, run persisted) so the structure
        # exists for later, but it deliberately returns no analysis yet:
        # per plan, this skill is not validated for real tax/legal decisions.
        disclaimer = SCENARIO_SCAFFOLD_DISCLAIMER

    supabase_storage.create_gap_analysis_run(
        run_id=run_id,
        skill_id=body.skill_id,
        reference_collection_ids=body.reference_collection_ids,
        framework_name=body.framework_name,
        target_collection_id=body.target_collection_id,
        scenario_input=body.scenario_input,
        owner_id=user.user_id,
        status="completed",
    )
    supabase_storage.save_gap_analysis_items(run_id, items)

    run_row = supabase_storage.get_gap_analysis_run(run_id) or {
        "run_id": run_id,
        "skill_id": body.skill_id,
        "framework_name": body.framework_name,
        "reference_collection_ids": body.reference_collection_ids,
        "target_collection_id": body.target_collection_id,
        "scenario_input": body.scenario_input,
        "status": "completed",
        "created_at": "",
    }

    return GapAnalysisResponse(
        run=GapAnalysisRun(**run_row),
        items=[GapAnalysisItem(**it) for it in items],
        summary=_summarize(items),
        disclaimer=disclaimer,
    )


@router.get("/analysis/gap-analysis/runs", response_model=List[GapAnalysisRun])
async def list_gap_analysis_runs(user: UserRecord = Depends(get_current_user)):
    """History of runs visible to the current user (own runs, or all for admins)."""
    rows = supabase_storage.list_gap_analysis_runs(
        owner_id=user.user_id, is_admin=(user.role == "admin")
    )
    return [GapAnalysisRun(**row) for row in rows]


@router.get("/analysis/gap-analysis/{run_id}", response_model=GapAnalysisResponse)
async def get_gap_analysis_run(run_id: str, user: UserRecord = Depends(get_current_user)):
    """Full result of a stored run — this is the 'view everything in-app'
    path (not just download): the frontend's full-table dialog reads from
    this endpoint, no re-analysis needed."""
    run_row = supabase_storage.get_gap_analysis_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _can_access_run(run_row, user):
        raise HTTPException(status_code=403, detail="Not allowed to access this run")

    items = supabase_storage.get_gap_analysis_items(run_id)
    return GapAnalysisResponse(
        run=GapAnalysisRun(**run_row),
        items=[GapAnalysisItem(**it) for it in items],
        summary=_summarize(items),
        disclaimer=None,
    )


@router.get("/analysis/gap-analysis/{run_id}/export")
async def export_gap_analysis_run(
    run_id: str,
    format: str = Query("markdown", pattern="^(markdown|pdf)$"),
    user: UserRecord = Depends(get_current_user),
):
    """Render a stored run into a downloadable report. Pure render of
    already-persisted data — no LLM call. Per plan, this is part of Skill 1's
    MVP (not deferred): download is an additional way to take the report
    outside the app, not the only way to see the full result."""
    run_row = supabase_storage.get_gap_analysis_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _can_access_run(run_row, user):
        raise HTTPException(status_code=403, detail="Not allowed to access this run")

    items = supabase_storage.get_gap_analysis_items(run_id)
    markdown = _render_markdown(run_row, items)

    if format == "markdown":
        return Response(
            content=markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="gap-analysis-{run_id}.md"'},
        )

    pdf_bytes = _render_pdf(run_row, items, markdown)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="gap-analysis-{run_id}.pdf"'},
    )


def _render_markdown(run_row: dict, items: List[dict]) -> str:
    lines = [
        f"# Laporan Gap Analysis — {run_row.get('framework_name', '')}",
        "",
        f"- Skill: {run_row.get('skill_id', '')}",
        f"- Dibuat: {run_row.get('created_at', '')}",
        f"- Reference collection: {', '.join(run_row.get('reference_collection_ids') or [])}",
        f"- Target collection: {run_row.get('target_collection_id') or '-'}",
        "",
        "> Catatan: laporan ini dihasilkan AI berdasarkan dokumen yang diupload user — "
        "bukan audit/sertifikasi resmi maupun nasihat pajak/hukum resmi.",
        "",
        "| Item | Status | Evidence | Rekomendasi |",
        "|---|---|---|---|",
    ]
    for item in items:
        evidence = (item.get("evidence") or "").replace("\n", " ").replace("|", "/")
        recommendation = (item.get("recommendation") or "").replace("\n", " ").replace("|", "/")
        lines.append(f"| {item.get('label', '')} | {item.get('status', '')} | {evidence} | {recommendation} |")
    return "\n".join(lines)


def _render_pdf(run_row: dict, items: List[dict], markdown_fallback: str) -> bytes:
    """Render the run as a PDF. Falls back to returning the markdown as raw
    bytes if reportlab isn't installed — library choice is an implementation
    detail, not a blocker for shipping this endpoint (per plan)."""
    try:
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()

        story = [
            Paragraph(f"Laporan Gap Analysis — {run_row.get('framework_name', '')}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Dibuat: {run_row.get('created_at', '')}", styles["Normal"]),
            Paragraph(
                "Catatan: laporan ini dihasilkan AI berdasarkan dokumen yang diupload user — "
                "bukan audit/sertifikasi resmi maupun nasihat pajak/hukum resmi.",
                styles["Italic"],
            ),
            Spacer(1, 12),
        ]

        table_data = [["Item", "Status", "Evidence", "Rekomendasi"]]
        for item in items:
            table_data.append([
                item.get("label", ""),
                item.get("status", ""),
                (item.get("evidence") or "")[:200],
                (item.get("recommendation") or "")[:200],
            ])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E6B4C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
        doc.build(story)
        return buffer.getvalue()
    except ImportError:
        logger.warning("reportlab not installed — export falling back to markdown bytes for PDF request")
        return markdown_fallback.encode("utf-8")
