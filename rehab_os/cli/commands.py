"""CLI commands for RehabOS."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rehab_os.config import get_settings

app = typer.Typer(
    name="rehab-os",
    help="Multi-agent clinical reasoning system for PT/OT/SLP",
    add_completion=False,
)
console = Console()


def get_orchestrator():
    """Get initialized orchestrator."""
    from rehab_os.llm import create_router_from_settings
    from rehab_os.agents import Orchestrator
    from rehab_os.knowledge import VectorStore

    settings = get_settings()
    llm = create_router_from_settings()
    vector_store = VectorStore(persist_directory=settings.chroma_persist_dir)

    return Orchestrator(llm=llm, knowledge_base=vector_store)


@app.command()
def consult(
    query: str = typer.Argument(..., help="Clinical consultation query"),
    discipline: str = typer.Option("PT", "--discipline", "-d", help="Discipline: PT, OT, SLP"),
    setting: str = typer.Option(
        "outpatient", "--setting", "-s", help="Care setting"
    ),
    patient_file: Optional[Path] = typer.Option(
        None, "--patient", "-p", help="JSON file with patient context"
    ),
    include_docs: bool = typer.Option(
        False, "--docs", help="Generate documentation"
    ),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA review"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Run a clinical consultation through the agent pipeline."""
    from rehab_os.models.output import ClinicalRequest
    from rehab_os.models.patient import PatientContext, Discipline, CareSetting

    # Parse discipline and setting
    try:
        discipline_enum = Discipline(discipline)
    except ValueError:
        console.print(f"[red]Invalid discipline: {discipline}. Use PT, OT, or SLP[/red]")
        raise typer.Exit(1)

    try:
        setting_enum = CareSetting(setting)
    except ValueError:
        console.print(f"[red]Invalid setting: {setting}[/red]")
        raise typer.Exit(1)

    # Load patient context if provided
    patient = None
    if patient_file:
        if not patient_file.exists():
            console.print(f"[red]Patient file not found: {patient_file}[/red]")
            raise typer.Exit(1)
        patient_data = json.loads(patient_file.read_text())
        patient = PatientContext.model_validate(patient_data)
    else:
        # Create minimal patient context from query
        patient = PatientContext(
            age=50,  # Default
            sex="other",
            chief_complaint=query,
            discipline=discipline_enum,
            setting=setting_enum,
        )

    # Create request
    request = ClinicalRequest(
        query=query,
        patient=patient,
        discipline=discipline_enum,
        setting=setting_enum,
        include_documentation=include_docs,
    )

    # Run consultation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing consultation...", total=None)

        orchestrator = get_orchestrator()
        response = asyncio.run(orchestrator.process(request, skip_qa=skip_qa))

        progress.update(task, completed=True)

    # Output results
    if output_json:
        console.print(response.model_dump_json(indent=2))
    else:
        _display_response(response)


def _display_response(response):
    """Display consultation response in rich format."""
    # Safety Panel
    safety = response.safety
    safety_color = "green" if safety.is_safe_to_treat else "red"
    console.print(
        Panel(
            f"[bold]Safe to Treat:[/bold] {safety.is_safe_to_treat}\n"
            f"[bold]Urgency:[/bold] {safety.urgency_level.value}\n"
            f"[bold]Summary:[/bold] {safety.summary}",
            title="Safety Assessment",
            border_style=safety_color,
        )
    )

    if safety.red_flags:
        table = Table(title="Red Flags")
        table.add_column("Finding")
        table.add_column("Urgency")
        table.add_column("Action")
        for rf in safety.red_flags:
            table.add_row(rf.finding, rf.urgency.value, rf.recommended_action)
        console.print(table)

    # Diagnosis
    if response.diagnosis:
        dx = response.diagnosis
        console.print(
            Panel(
                f"[bold]Primary:[/bold] {dx.primary_diagnosis}\n"
                f"[bold]ICD-10:[/bold] {', '.join(dx.icd_codes)}\n"
                f"[bold]Confidence:[/bold] {dx.confidence:.0%}\n"
                f"[bold]Rationale:[/bold] {dx.rationale}",
                title="Diagnosis",
            )
        )

    # Plan of Care
    if response.plan:
        plan = response.plan
        console.print(
            Panel(
                f"[bold]Clinical Summary:[/bold]\n{plan.clinical_summary}\n\n"
                f"[bold]Prognosis:[/bold] {plan.prognosis}\n"
                f"[bold]Visit Frequency:[/bold] {plan.visit_frequency}",
                title="Plan of Care",
            )
        )

        # Goals table
        if plan.smart_goals:
            table = Table(title="Goals")
            table.add_column("Timeframe")
            table.add_column("Goal")
            for goal in plan.smart_goals:
                table.add_row(goal.timeframe.value, goal.description)
            console.print(table)

        # Interventions
        if plan.interventions:
            table = Table(title="Interventions")
            table.add_column("Intervention")
            table.add_column("Rationale")
            for intervention in plan.interventions[:5]:
                table.add_row(intervention.name, intervention.rationale[:100] + "...")
            console.print(table)

    # QA Review
    if response.qa_review:
        qa = response.qa_review
        console.print(
            Panel(
                f"[bold]Quality Score:[/bold] {qa.overall_quality:.0%}\n"
                f"[bold]Strengths:[/bold] {', '.join(qa.strengths[:3])}\n"
                f"[bold]Suggestions:[/bold] {', '.join(qa.suggestions[:3])}",
                title="QA Review",
            )
        )

    # Disclaimer
    console.print(f"\n[dim]{response.disclaimer}[/dim]")


@app.command()
def evidence(
    query: str = typer.Argument(..., help="Evidence search query"),
    condition: str = typer.Option(..., "--condition", "-c", help="Clinical condition"),
    discipline: str = typer.Option("PT", "--discipline", "-d", help="Discipline"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
):
    """Search for clinical evidence."""
    from rehab_os.llm import create_router_from_settings
    from rehab_os.agents import EvidenceAgent
    from rehab_os.agents.evidence import EvidenceInput
    from rehab_os.agents.base import AgentContext
    from rehab_os.knowledge import VectorStore

    settings = get_settings()
    llm = create_router_from_settings()
    vector_store = VectorStore(persist_directory=settings.chroma_persist_dir)

    agent = EvidenceAgent(llm=llm, knowledge_base=vector_store)

    evidence_input = EvidenceInput(
        condition=condition,
        clinical_question=query,
    )
    context = AgentContext(discipline=discipline)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Searching evidence...", total=None)
        result = asyncio.run(agent.run(evidence_input, context))

    # Display results
    console.print(Panel(f"[bold]Query:[/bold] {result.query}", title="Evidence Search"))

    if result.synthesis:
        console.print(Panel(result.synthesis, title="Evidence Synthesis"))

    if result.evidence_items:
        table = Table(title=f"Evidence Items ({len(result.evidence_items)})")
        table.add_column("Source")
        table.add_column("Level")
        table.add_column("Content")
        for ev in result.evidence_items:
            table.add_row(
                ev.source,
                ev.evidence_level.value,
                ev.content[:200] + "..." if len(ev.content) > 200 else ev.content,
            )
        console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the REST API server."""
    import uvicorn

    console.print(f"Starting RehabOS API server on {host}:{port}")
    uvicorn.run(
        "rehab_os.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command()
def init_kb(
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Directory with guideline files"
    ),
    load_samples: bool = typer.Option(True, "--samples", help="Load sample guidelines"),
):
    """Initialize the knowledge base with guidelines."""
    from rehab_os.knowledge import VectorStore, GuidelineLoader, initialize_knowledge_base

    settings = get_settings()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing knowledge base...", total=None)

        vector_store, _ = asyncio.run(
            initialize_knowledge_base(
                persist_dir=settings.chroma_persist_dir,
                load_samples=load_samples,
            )
        )

        if directory:
            progress.update(task, description="Loading custom guidelines...")
            loader = GuidelineLoader(vector_store)
            count = asyncio.run(loader.load_from_directory(directory))
            console.print(f"Loaded {count} chunks from {directory}")

        progress.update(task, completed=True)

    console.print(f"[green]Knowledge base initialized: {vector_store.count} documents[/green]")


@app.command()
def health():
    """Check system health."""
    from rehab_os.llm import create_router_from_settings

    settings = get_settings()

    console.print("[bold]RehabOS Health Check[/bold]\n")

    # Check LLM
    llm = create_router_from_settings()
    health_status = asyncio.run(llm.health_check())

    table = Table(title="LLM Status")
    table.add_column("Provider")
    table.add_column("Status")
    for provider, status in health_status.items():
        status_str = "[green]OK[/green]" if status else "[red]UNAVAILABLE[/red]"
        table.add_row(provider, status_str)
    console.print(table)

    # Check knowledge base
    from rehab_os.knowledge import VectorStore

    try:
        vs = VectorStore(persist_directory=settings.chroma_persist_dir)
        console.print(f"[green]Knowledge Base: {vs.count} documents[/green]")
    except Exception as e:
        console.print(f"[red]Knowledge Base: Error - {e}[/red]")


@app.command()
def version():
    """Show version information."""
    from rehab_os import __version__

    console.print(f"RehabOS v{__version__}")


@app.command()
def analytics(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to analyze"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Export report to JSON file"
    ),
    show_candidates: bool = typer.Option(
        False, "--candidates", "-c", help="Show prompt tuning candidates"
    ),
):
    """Generate prompt effectiveness analytics report."""
    from datetime import datetime, timedelta
    from rehab_os.observability import PromptAnalytics

    settings = get_settings()
    log_dir = Path("data/logs")

    if not log_dir.exists():
        console.print("[yellow]No logs found. Run some consultations first.[/yellow]")
        return

    analytics_engine = PromptAnalytics(log_dir)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing logs...", total=None)

        report = analytics_engine.generate_report(start_time, end_time)
        report_dict = report.to_dict()

        progress.update(task, completed=True)

    # Display summary
    console.print(Panel.fit("[bold]Prompt Effectiveness Report[/bold]"))
    console.print(f"Period: {start_time.date()} to {end_time.date()}\n")

    # Summary stats
    summary = report_dict["summary"]
    console.print(f"Total Agent Runs: {summary['total_agent_runs']}")
    console.print(f"Total LLM Calls: {summary['total_llm_calls']}")
    console.print(f"Total Tokens: {summary['total_tokens']:,}")
    console.print(f"Estimated Cost: ${summary['estimated_cost']:.2f}\n")

    # Agent metrics table
    if report_dict["agents"]:
        table = Table(title="Agent Performance")
        table.add_column("Agent")
        table.add_column("Runs", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Confidence", justify="right")

        for name, metrics in report_dict["agents"].items():
            conf = metrics["avg_confidence"]
            conf_str = f"{conf:.2f}" if conf else "-"
            table.add_row(
                name,
                str(metrics["total_runs"]),
                f"{metrics['success_rate']:.0%}",
                f"{metrics['avg_latency_ms']:.0f}ms",
                f"{metrics['avg_tokens']:.0f}",
                conf_str,
            )
        console.print(table)

    # Attention needed
    attention = report_dict["attention_needed"]
    if any(attention.values()):
        console.print("\n[bold yellow]Attention Needed:[/bold yellow]")
        if attention["high_correction_rate"]:
            console.print(f"  High correction rate: {', '.join(attention['high_correction_rate'])}")
        if attention["high_latency"]:
            console.print(f"  High latency: {', '.join(attention['high_latency'])}")
        if attention["low_confidence"]:
            console.print(f"  Low confidence: {', '.join(attention['low_confidence'])}")
        if attention["high_fallback"]:
            console.print(f"  High fallback rate: {', '.join(attention['high_fallback'])}")

    # Show tuning candidates if requested
    if show_candidates:
        candidates = analytics_engine.get_prompt_tuning_candidates(start_time, end_time)
        if candidates:
            console.print("\n[bold]Prompt Tuning Candidates:[/bold]")
            for c in candidates:
                console.print(f"\n  [cyan]{c['agent']}[/cyan]")
                for reason in c["reasons"]:
                    console.print(f"    - {reason}")
        else:
            console.print("\n[green]No agents require immediate prompt tuning.[/green]")

    # Export if requested
    if output:
        with open(output, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        console.print(f"\n[green]Report exported to {output}[/green]")


@app.command()
def feedback_stats():
    """Show feedback annotation statistics."""
    from pathlib import Path

    feedback_file = Path("data/logs/feedback.jsonl")
    if not feedback_file.exists():
        console.print("[yellow]No feedback annotations found.[/yellow]")
        return

    # Load feedback entries
    entries = []
    with open(feedback_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        console.print("[yellow]No feedback annotations found.[/yellow]")
        return

    console.print(Panel.fit("[bold]Feedback Annotation Statistics[/bold]"))
    console.print(f"Total Annotations: {len(entries)}\n")

    # By status
    by_status: dict[str, int] = {}
    by_agent: dict[str, dict[str, int]] = {}

    for entry in entries:
        status = entry.get("status", "unknown")
        agent = entry.get("agent_name", "unknown")

        by_status[status] = by_status.get(status, 0) + 1

        if agent not in by_agent:
            by_agent[agent] = {}
        by_agent[agent][status] = by_agent[agent].get(status, 0) + 1

    # Status table
    table = Table(title="By Status")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in sorted(by_status.items()):
        table.add_row(status, str(count))
    console.print(table)

    # By agent table
    if by_agent:
        table = Table(title="By Agent")
        table.add_column("Agent")
        table.add_column("Accepted", justify="right")
        table.add_column("Rejected", justify="right")
        table.add_column("Needs Review", justify="right")

        for agent, counts in sorted(by_agent.items()):
            table.add_row(
                agent,
                str(counts.get("accepted", 0)),
                str(counts.get("rejected", 0)),
                str(counts.get("needs_review", 0)),
            )
        console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run dashboard on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
):
    """Launch the Streamlit evaluation dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        console.print("[red]Dashboard not found.[/red]")
        return

    console.print(f"[bold]Launching RehabOS Dashboard[/bold]")
    console.print(f"URL: http://{host}:{port}")
    console.print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_path),
                "--server.port", str(port),
                "--server.address", host,
                "--browser.gatherUsageStats", "false",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Dashboard failed to start: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


@app.command()
def scheduler(
    action: str = typer.Argument(..., help="Action: start, stop, status, run-task"),
    task_id: Optional[str] = typer.Option(None, "--task", "-t", help="Task ID for run-task"),
    setup_defaults: bool = typer.Option(False, "--setup", help="Setup default schedule"),
):
    """Manage the learning loop scheduler."""
    from rehab_os.learning.scheduler import LearningScheduler, create_learning_scheduler

    scheduler_instance = LearningScheduler()

    if setup_defaults:
        scheduler_instance.setup_default_schedule()
        console.print("[green]Default schedule configured.[/green]")

    if action == "status":
        tasks = scheduler_instance.list_tasks()

        if not tasks:
            console.print("[yellow]No scheduled tasks.[/yellow]")
            console.print("Run with --setup to configure default schedule.")
            return

        table = Table(title="Scheduled Tasks")
        table.add_column("Task ID")
        table.add_column("Type")
        table.add_column("Schedule")
        table.add_column("Last Run")
        table.add_column("Status")
        table.add_column("Enabled")

        for task in tasks:
            last_run = task.last_run.strftime("%Y-%m-%d %H:%M") if task.last_run else "Never"
            enabled = "[green]Yes[/green]" if task.enabled else "[red]No[/red]"
            status_color = {
                "completed": "green",
                "failed": "red",
                "running": "yellow",
                "pending": "white",
            }.get(task.last_status.value, "white")

            table.add_row(
                task.task_id,
                task.task_type.value,
                task.schedule,
                last_run,
                f"[{status_color}]{task.last_status.value}[/{status_color}]",
                enabled,
            )

        console.print(table)

    elif action == "run-task":
        if not task_id:
            console.print("[red]Task ID required for run-task action.[/red]")
            return

        task = scheduler_instance.get_task(task_id)
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return

        console.print(f"Running task: {task_id}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            prog_task = progress.add_task(f"Running {task_id}...", total=None)
            result = asyncio.run(scheduler_instance.run_task(task_id))
            progress.update(prog_task, completed=True)

        if result.status.value == "completed":
            console.print(f"[green]Task completed successfully.[/green]")
            if result.result:
                console.print(json.dumps(result.result, indent=2, default=str))
        else:
            console.print(f"[red]Task failed: {result.error}[/red]")

    elif action == "start":
        console.print("[bold]Starting learning loop scheduler...[/bold]")
        console.print("Press Ctrl+C to stop\n")

        try:
            asyncio.run(scheduler_instance.run_loop(check_interval=60))
        except KeyboardInterrupt:
            scheduler_instance.stop()
            console.print("\n[yellow]Scheduler stopped.[/yellow]")

    elif action == "stop":
        console.print("[yellow]Stop command requires scheduler to be running in background.[/yellow]")
        console.print("Use Ctrl+C to stop the running scheduler.")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: start, stop, status, run-task")


@app.command()
def optimize(
    agent_name: str = typer.Argument(..., help="Agent to optimize"),
    strategy: str = typer.Option("metric_based", "--strategy", "-s", help="Strategy: metric_based, feedback_based"),
    apply: bool = typer.Option(False, "--apply", help="Apply optimization with LLM"),
):
    """Generate and optionally apply prompt optimizations."""
    from rehab_os.learning.prompt_optimizer import PromptOptimizer

    optimizer = PromptOptimizer()

    console.print(f"[bold]Analyzing {agent_name}...[/bold]\n")

    # Generate optimization
    optimization = optimizer.generate_optimization(agent_name, strategy)

    if not optimization:
        console.print(f"[green]No optimization needed for {agent_name}.[/green]")
        return

    console.print(Panel.fit(f"[bold]Optimization for {agent_name}[/bold]"))
    console.print(f"Strategy: {optimization.optimization_type}")
    console.print(f"Reasoning: {optimization.reasoning}\n")

    console.print("[bold]Suggested Changes:[/bold]")
    for change in optimization.changes:
        console.print(f"  - {change}")

    console.print("\n[bold]Current Metrics:[/bold]")
    for key, value in optimization.before_metrics.items():
        if value is not None:
            console.print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    if optimization.predicted_improvement:
        console.print("\n[bold]Predicted Improvements:[/bold]")
        for key, value in optimization.predicted_improvement.items():
            console.print(f"  {key}: {value:+.2f}")

    if apply:
        console.print("\n[yellow]Applying optimization with LLM...[/yellow]")
        console.print("[yellow]Note: This requires an active LLM connection.[/yellow]")
        # Would call optimizer.apply_optimization_with_llm() here
        console.print("[red]LLM-based optimization not implemented in CLI yet.[/red]")


if __name__ == "__main__":
    app()
