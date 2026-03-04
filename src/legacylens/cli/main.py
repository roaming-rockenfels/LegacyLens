"""LegacyLens CLI — query legacy codebases with natural language."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(
    name="legacylens",
    help="RAG-powered CLI for querying legacy Fortran codebases.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to codebase directory to ingest"),
) -> None:
    """Ingest a legacy codebase into the vector database."""
    from legacylens.rag.ingest import ingest_directory

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"[red]Error:[/] {path} is not a directory")
        raise typer.Exit(1)

    stats = ingest_directory(str(resolved))
    if stats["chunks"] == 0:
        raise typer.Exit(1)


@app.command()
def batch_ingest(
    path: str = typer.Argument(..., help="Path to codebase directory to ingest"),
    no_reset: bool = typer.Option(False, "--no-reset", help="Don't delete/recreate Pinecone index"),
) -> None:
    """Ingest a legacy codebase using the Voyage Batch API (for large datasets)."""
    from legacylens.rag.batch_ingest import batch_ingest_directory

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"[red]Error:[/] {path} is not a directory")
        raise typer.Exit(1)

    stats = batch_ingest_directory(str(resolved), reset_index=not no_reset)
    if stats.get("chunks", 0) == 0 or stats.get("error"):
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    mode: str = typer.Option(
        "explain",
        "--mode",
        "-m",
        help="Response mode: explain, deps, docs, business_logic",
    ),
    no_answer: bool = typer.Option(
        False, "--no-answer", help="Only show retrieved chunks, skip LLM answer"
    ),
    show_code: bool = typer.Option(
        False, "--show-code/--no-code", help="Show source code snippets in results"
    ),
) -> None:
    """Ask a natural language question about the codebase."""
    from legacylens.rag.retrieve import retrieve, format_results
    from legacylens.rag.generate import generate_answer

    console.print(f"\n[bold]Query:[/] {question}\n")

    with console.status("Searching vector database..."):
        results = retrieve(question, top_k=top_k)

    if not results:
        console.print("[yellow]No results found. Have you ingested a codebase?[/]")
        raise typer.Exit(1)

    format_results(results, show_code=show_code)

    if no_answer:
        return

    console.print(Panel(f"[dim]Generating {mode} answer with Claude...[/]", border_style="blue"))

    answer = generate_answer(question, results, mode=mode)
    console.print()
    console.print(Panel(Markdown(answer), title="[bold green]LegacyLens Answer[/]", border_style="green"))


@app.command()
def explain(
    question: str = typer.Argument(..., help="What to explain about the codebase"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
) -> None:
    """Explain how code works in the legacy codebase."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    console.print(f"\n[bold]Explaining:[/] {question}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="explain")
    console.print(Panel(Markdown(answer), title="[bold green]Explanation[/]", border_style="green"))


@app.command()
def deps(
    unit_name: str = typer.Argument(..., help="Subroutine/function name to trace dependencies for"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to retrieve"),
) -> None:
    """Show dependency relationships for a subroutine or function."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    question = f"What are the dependencies and call relationships of {unit_name}? What does it call and what calls it?"
    console.print(f"\n[bold]Dependencies for:[/] {unit_name}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, pin_unit=unit_name)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="deps")
    console.print(Panel(Markdown(answer), title=f"[bold green]Dependencies: {unit_name}[/]", border_style="green"))


@app.command()
def document(
    unit_name: str = typer.Argument(..., help="Subroutine/function name to generate docs for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
) -> None:
    """Generate modern documentation for a subroutine or function."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    question = f"Generate comprehensive documentation for the {unit_name} routine"
    console.print(f"\n[bold]Generating docs for:[/] {unit_name}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, pin_unit=unit_name)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="docs")
    console.print(Panel(Markdown(answer), title=f"[bold green]Documentation: {unit_name}[/]", border_style="green"))


@app.command()
def logic(
    question: str = typer.Argument(..., help="What business/mathematical logic to extract"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
) -> None:
    """Extract and explain the mathematical/business logic."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    console.print(f"\n[bold]Extracting logic:[/] {question}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="business_logic")
    console.print(Panel(Markdown(answer), title="[bold green]Business Logic[/]", border_style="green"))


@app.command()
def view(
    file_path: str = typer.Argument(..., help="File path to view (from query results)"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Base directory of ingested codebase"),
) -> None:
    """View the full source of a file from query results."""
    from legacylens.rag.source_reader import resolve_source_path

    base = Path(data_dir) if data_dir else None
    full_path = resolve_source_path(file_path, base_dir=base)
    if full_path is None:
        console.print(f"[red]Error:[/] File not found: {file_path}")
        raise typer.Exit(1)

    content = full_path.read_text(encoding="utf-8", errors="replace")
    ext = full_path.suffix.lower()
    lexer = "fortran" if ext in (".f", ".f90", ".f95", ".f03") else "text"

    console.print(f"\n[bold]{file_path}[/] ({len(content.splitlines())} lines)\n")
    syntax = Syntax(content, lexer, line_numbers=True, theme="monokai")
    console.print(syntax)


@app.command()
def stats() -> None:
    """Show index statistics."""
    from legacylens.rag.storage import get_index_stats

    info = get_index_stats()
    console.print(f"[bold]Index Stats:[/]")
    console.print(f"  Total vectors: {info.get('total_vector_count', 'N/A')}")
    console.print(f"  Dimension: {info.get('dimension', 'N/A')}")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the API server on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
) -> None:
    """Start the FastAPI server with Swagger UI."""
    import uvicorn

    console.print(f"\n[bold green]Starting LegacyLens API server[/]")
    console.print(f"  Swagger UI: http://{host}:{port}/docs")
    console.print(f"  API base:   http://{host}:{port}\n")
    uvicorn.run("legacylens.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
