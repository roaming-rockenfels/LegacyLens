"""LegacyLens CLI — query legacy codebases with natural language."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(
    name="legacylens",
    help="RAG-powered CLI for querying legacy codebases.",
    no_args_is_help=True,
)
sources_app = typer.Typer(help="Manage ingested source codebases.")
app.add_typer(sources_app, name="sources")
console = Console()

_LEXER_MAP = {
    ".f": "fortran", ".f90": "fortran", ".f95": "fortran", ".f03": "fortran", ".for": "fortran",
    ".py": "python", ".pyi": "python",
}


def _resolve_source(source_name: str | None) -> tuple[str | None, list[str] | None, str | None]:
    """Resolve a --source flag to (namespace, languages, source_dir).

    Returns (None, None, None) if no source.
    """
    if not source_name:
        return None, None, None
    from legacylens.sources import get_source
    info = get_source(source_name)
    if info is None:
        console.print(f"[red]Error:[/] Source '{source_name}' not found. Run 'legacylens sources list'.")
        raise typer.Exit(1)
    return info.namespace, info.languages, info.path


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to codebase directory to ingest"),
    name: str = typer.Option(None, "--name", "-n", help="Source name (defaults to folder basename)"),
) -> None:
    """Ingest a legacy codebase into the vector database."""
    from legacylens.rag.ingest import ingest_directory
    from legacylens.sources import scan_directory, detect_languages, register_source

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"[red]Error:[/] {path} is not a directory")
        raise typer.Exit(1)

    source_name = name or resolved.name

    # Pre-scan for supported file types
    from legacylens.rag.ingest import create_default_registry
    registry = create_default_registry()
    extensions = scan_directory(str(resolved), known_extensions=registry.supported_extensions())
    if not extensions:
        console.print(f"[red]Error:[/] No supported files found in {path}")
        console.print(f"  Supported extensions: {', '.join(registry.supported_extensions())}")
        raise typer.Exit(1)

    languages = detect_languages(extensions)
    console.print(f"  Detected languages: {', '.join(languages)}")
    console.print(f"  File extensions: {extensions}")

    from legacylens.sources import _slugify
    namespace = _slugify(source_name)

    stats = ingest_directory(str(resolved), namespace=namespace)
    if stats["chunks"] == 0:
        raise typer.Exit(1)

    register_source(source_name, str(resolved), languages, extensions, stats["chunks"])
    console.print(f"  Registered source: [bold]{source_name}[/] (namespace: {namespace})")


@app.command()
def batch_ingest(
    path: str = typer.Argument(..., help="Path to codebase directory to ingest"),
    no_reset: bool = typer.Option(False, "--no-reset", help="Don't delete/recreate Pinecone index"),
    name: str = typer.Option(None, "--name", "-n", help="Source name (defaults to folder basename)"),
) -> None:
    """Ingest a legacy codebase using the Voyage Batch API (for large datasets)."""
    from legacylens.rag.batch_ingest import batch_ingest_directory
    from legacylens.sources import scan_directory, detect_languages, register_source, _slugify

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"[red]Error:[/] {path} is not a directory")
        raise typer.Exit(1)

    source_name = name or resolved.name
    namespace = _slugify(source_name)

    from legacylens.rag.ingest import create_default_registry
    registry = create_default_registry()
    extensions = scan_directory(str(resolved), known_extensions=registry.supported_extensions())
    languages = detect_languages(extensions)

    stats = batch_ingest_directory(str(resolved), reset_index=not no_reset, namespace=namespace)
    if stats.get("chunks", 0) == 0 or stats.get("error"):
        raise typer.Exit(1)

    register_source(source_name, str(resolved), languages, extensions, stats.get("chunks", 0))
    console.print(f"  Registered source: [bold]{source_name}[/] (namespace: {namespace})")


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
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Ask a natural language question about the codebase."""
    from legacylens.rag.retrieve import retrieve, format_results
    from legacylens.rag.generate import generate_answer

    namespace, languages, source_dir = _resolve_source(source)
    console.print(f"\n[bold]Query:[/] {question}\n")

    with console.status("Searching vector database..."):
        results = retrieve(question, top_k=top_k, namespace=namespace, languages=languages)

    if not results:
        console.print("[yellow]No results found. Have you ingested a codebase?[/]")
        raise typer.Exit(1)

    format_results(results, show_code=show_code)

    if no_answer:
        return

    console.print(Panel(f"[dim]Generating {mode} answer with Claude...[/]", border_style="blue"))

    answer = generate_answer(question, results, mode=mode, source_dir=source_dir)
    console.print()
    console.print(Panel(Markdown(answer), title="[bold green]LegacyLens Answer[/]", border_style="green"))


@app.command()
def explain(
    question: str = typer.Argument(..., help="What to explain about the codebase"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Explain how code works in the legacy codebase."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    namespace, languages, source_dir = _resolve_source(source)
    console.print(f"\n[bold]Explaining:[/] {question}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, namespace=namespace, languages=languages)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="explain", source_dir=source_dir)
    console.print(Panel(Markdown(answer), title="[bold green]Explanation[/]", border_style="green"))


@app.command()
def deps(
    unit_name: str = typer.Argument(..., help="Subroutine/function name to trace dependencies for"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to retrieve"),
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Show dependency relationships for a subroutine or function."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    namespace, languages, source_dir = _resolve_source(source)
    question = f"What are the dependencies and call relationships of {unit_name}? What does it call and what calls it?"
    console.print(f"\n[bold]Dependencies for:[/] {unit_name}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, pin_unit=unit_name, namespace=namespace, languages=languages)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="deps", source_dir=source_dir)
    console.print(Panel(Markdown(answer), title=f"[bold green]Dependencies: {unit_name}[/]", border_style="green"))


@app.command()
def document(
    unit_name: str = typer.Argument(..., help="Subroutine/function name to generate docs for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Generate modern documentation for a subroutine or function."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    namespace, languages, source_dir = _resolve_source(source)
    question = f"Generate comprehensive documentation for the {unit_name} routine"
    console.print(f"\n[bold]Generating docs for:[/] {unit_name}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, pin_unit=unit_name, namespace=namespace, languages=languages)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="docs", source_dir=source_dir)
    console.print(Panel(Markdown(answer), title=f"[bold green]Documentation: {unit_name}[/]", border_style="green"))


@app.command()
def logic(
    question: str = typer.Argument(..., help="What business/mathematical logic to extract"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Extract and explain the mathematical/business logic."""
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    namespace, languages, source_dir = _resolve_source(source)
    console.print(f"\n[bold]Extracting logic:[/] {question}\n")

    with console.status("Searching..."):
        results = retrieve(question, top_k=top_k, namespace=namespace, languages=languages)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(1)

    answer = generate_answer(question, results, mode="business_logic", source_dir=source_dir)
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
    lexer = _LEXER_MAP.get(ext, "text")

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
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    source: str = typer.Option(None, "--source", "-s", help="Source name to query"),
) -> None:
    """Interactive multi-turn chat about the codebase."""
    from legacylens.rag.session import ChatSession

    namespace, languages, source_dir = _resolve_source(source)
    # Note: ChatSession doesn't yet support namespace/languages passthrough,
    # but the source flag is validated here for consistency.

    console.print(
        Panel(
            "[bold]LegacyLens Chat[/]\nType [cyan]exit[/] or [cyan]quit[/] to end, "
            "[cyan]reset[/] to clear history.",
            border_style="blue",
        )
    )

    with ChatSession(top_k=top_k) as session:
        while True:
            try:
                question = console.input("\n[bold]You:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            if question.lower() == "reset":
                session.reset()
                console.print("[dim]Session reset.[/]")
                continue

            with console.status("Thinking..."):
                answer = session.ask(question)

            console.print()
            console.print(
                Panel(Markdown(answer), title="[bold green]LegacyLens[/]", border_style="green")
            )

    console.print("\n[dim]Goodbye![/]")


@sources_app.command("list")
def sources_list() -> None:
    """List all ingested sources."""
    from rich.table import Table
    from legacylens.sources import list_sources

    all_sources = list_sources()
    if not all_sources:
        console.print("[yellow]No sources registered. Use 'legacylens ingest' to add one.[/]")
        return

    table = Table(title="Ingested Sources")
    table.add_column("Name", style="bold")
    table.add_column("Path")
    table.add_column("Languages")
    table.add_column("Chunks", justify="right")
    table.add_column("Ingested At")

    for s in all_sources:
        table.add_row(s.name, s.path, ", ".join(s.languages), str(s.chunk_count), s.ingested_at[:19])

    console.print(table)


@sources_app.command("info")
def sources_info(
    name: str = typer.Argument(..., help="Source name"),
) -> None:
    """Show detailed information about an ingested source."""
    from legacylens.sources import get_source

    info = get_source(name)
    if info is None:
        console.print(f"[red]Error:[/] Source '{name}' not found.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Source: {info.name}[/]")
    console.print(f"  Path:       {info.path}")
    console.print(f"  Namespace:  {info.namespace}")
    console.print(f"  Languages:  {', '.join(info.languages)}")
    console.print(f"  Chunks:     {info.chunk_count}")
    console.print(f"  Ingested:   {info.ingested_at}")
    console.print(f"  Extensions:")
    for ext, count in sorted(info.extensions.items()):
        console.print(f"    {ext}: {count} files")


@sources_app.command("rm")
def sources_rm(
    name: str = typer.Argument(..., help="Source name to remove"),
) -> None:
    """Remove an ingested source (deletes namespace from Pinecone + registry entry)."""
    from legacylens.sources import get_source, remove_source

    info = get_source(name)
    if info is None:
        console.print(f"[red]Error:[/] Source '{name}' not found.")
        raise typer.Exit(1)

    # Delete namespace from Pinecone
    try:
        from legacylens.rag.storage import delete_namespace
        delete_namespace(info.namespace)
        console.print(f"  Deleted Pinecone namespace: {info.namespace}")
    except Exception as e:
        console.print(f"  [yellow]Warning:[/] Failed to delete namespace: {e}")

    remove_source(name)
    console.print(f"  Removed source: [bold]{name}[/]")


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
