"""
ARIA — Automated Requirements Intelligence Agent
Energy Transmission Data Platform Requirements Gathering

Usage:
    python aria_agent.py                    # Interactive CLI session
    python aria_agent.py --output ./output  # Save requirements JSON to directory
    python aria_agent.py --session session.json  # Resume a saved session

Dependencies:
    pip install anthropic rich
"""

import json
import re
import os
import sys
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT — load from file or use inline fallback
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

def load_system_prompt() -> str:
    """Load system prompt from file, falling back to inline version."""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # Minimal inline fallback — use full system_prompt.txt in production
    return """
You are ARIA, a senior data platform architect specializing in energy transmission.
Conduct structured discovery interviews with business users to capture requirements
for data platform use cases. Ask no more than 2 questions at a time. Cover:
1. Business owner and problem framing
2. Source systems and data landscape (PI, SCADA, EMS, SAP, GIS, etc.)
3. Data consumers and access patterns
4. Compliance requirements (NERC CIP, FERC, market-sensitive data)
5. Success metrics and constraints

When the interview is complete, output a structured JSON requirements document
inside <requirements> tags followed by a plain-English executive summary.
"""


# ─────────────────────────────────────────────────────────────
# USE CASE ID GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_use_case_id(output_dir: Optional[Path] = None) -> str:
    """Generate a sequential use case ID like UC-2025-001."""
    year = date.today().year
    counter_file = (output_dir or Path(".")) / ".uc_counter.json"
    
    counters = {}
    if counter_file.exists():
        try:
            counters = json.loads(counter_file.read_text())
        except Exception:
            pass
    
    key = str(year)
    counters[key] = counters.get(key, 0) + 1
    
    try:
        counter_file.write_text(json.dumps(counters))
    except Exception:
        pass
    
    return f"UC-{year}-{counters[key]:03d}"


# ─────────────────────────────────────────────────────────────
# REQUIREMENTS EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_requirements(text: str) -> Optional[dict]:
    """Extract and parse JSON from <requirements> tags in the LLM response."""
    match = re.search(r"<requirements>(.*?)</requirements>", text, re.DOTALL)
    if not match:
        return None
    
    raw_json = match.group(1).strip()
    
    # Strip markdown code fences if present
    raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
    raw_json = re.sub(r"\s*```$", "", raw_json)
    
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse requirements JSON: {e}")
        # Return raw string so caller can handle
        return {"_raw": raw_json, "_parse_error": str(e)}


def extract_summary(text: str) -> str:
    """Extract the executive summary (text outside <requirements> tags)."""
    # Remove the requirements block
    summary = re.sub(r"<requirements>.*?</requirements>", "", text, flags=re.DOTALL)
    return summary.strip()


# ─────────────────────────────────────────────────────────────
# SESSION PERSISTENCE
# ─────────────────────────────────────────────────────────────

class Session:
    """Manages conversation history and session persistence."""
    
    def __init__(self, session_file: Optional[Path] = None):
        self.session_file = session_file
        self.history: list[dict] = []
        self.requirements: Optional[dict] = None
        self.started_at = datetime.now().isoformat()
        self.completed = False
        
        if session_file and session_file.exists():
            self._load()
    
    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def save(self):
        if not self.session_file:
            return
        data = {
            "started_at": self.started_at,
            "completed": self.completed,
            "history": self.history,
            "requirements": self.requirements,
        }
        self.session_file.write_text(json.dumps(data, indent=2))
    
    def _load(self):
        try:
            data = json.loads(self.session_file.read_text())
            self.started_at = data.get("started_at", self.started_at)
            self.completed = data.get("completed", False)
            self.history = data.get("history", [])
            self.requirements = data.get("requirements")
            print(f"Resumed session from {self.session_file} "
                  f"({len(self.history)//2} turns completed)")
        except Exception as e:
            print(f"Warning: Could not load session: {e}")


# ─────────────────────────────────────────────────────────────
# ARIA AGENT
# ─────────────────────────────────────────────────────────────

class ARIAAgent:
    """
    ARIA — Automated Requirements Intelligence Agent
    
    Conducts structured requirements interviews with business users
    for energy transmission data platform use cases.
    """
    
    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 2048
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        session: Optional[Session] = None,
        verbose: bool = False,
    ):
        self.client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from env
        self.system_prompt = load_system_prompt()
        self.output_dir = output_dir or Path(".")
        self.session = session or Session()
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _call_api(self, messages: list[dict]) -> str:
        """Make a Claude API call and return the text response."""
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=self.system_prompt,
            messages=messages,
        )
        return response.content[0].text
    
    def start_interview(self) -> str:
        """Generate the opening message from ARIA."""
        if self.session.history:
            # Resuming — don't repeat the intro
            return self.session.history[-1]["content"]
        
        opening = self._call_api([{
            "role": "user",
            "content": (
                "Hello, I'm ready to capture a new data platform use case. "
                "Please introduce yourself and begin the requirements interview."
            )
        }])
        
        self.session.add_turn("user", 
            "Hello, I'm ready to capture a new data platform use case. "
            "Please introduce yourself and begin the requirements interview.")
        self.session.add_turn("assistant", opening)
        self.session.save()
        return opening
    
    def send_message(self, user_message: str) -> tuple[str, Optional[dict]]:
        """
        Send a user message and get ARIA's response.
        
        Returns:
            (response_text, requirements_dict_or_None)
            requirements_dict is populated when the interview is complete.
        """
        self.session.add_turn("user", user_message)
        
        response = self._call_api(self.session.history)
        self.session.add_turn("assistant", response)
        
        # Check if requirements were generated
        requirements = extract_requirements(response)
        
        if requirements:
            self.session.requirements = requirements
            self.session.completed = True
            self._save_requirements(requirements, response)
        
        self.session.save()
        return response, requirements
    
    def _save_requirements(self, requirements: dict, full_response: str):
        """Save the requirements document to files."""
        # Assign use case ID if not present
        if "use_case_id" not in requirements or requirements["use_case_id"] == "UC-YYYY-NNN":
            requirements["use_case_id"] = generate_use_case_id(self.output_dir)
        
        uc_id = requirements.get("use_case_id", "UC-UNKNOWN")
        safe_id = uc_id.replace("/", "-")
        
        # Save structured JSON
        json_path = self.output_dir / f"{safe_id}_requirements.json"
        json_path.write_text(json.dumps(requirements, indent=2))
        
        # Save full interview transcript
        transcript_path = self.output_dir / f"{safe_id}_transcript.md"
        self._write_transcript(transcript_path, requirements, full_response)
        
        if self.verbose:
            print(f"\nSaved: {json_path}")
            print(f"Saved: {transcript_path}")
        
        return json_path, transcript_path
    
    def _write_transcript(self, path: Path, requirements: dict, final_response: str):
        """Write the full interview transcript as a Markdown file."""
        summary = extract_summary(final_response)
        uc_name = requirements.get("use_case_name", "Unknown Use Case")
        uc_id = requirements.get("use_case_id", "")
        
        lines = [
            f"# Requirements Interview Transcript",
            f"## {uc_id}: {uc_name}",
            f"",
            f"*Generated by ARIA on {date.today().isoformat()}*",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            summary,
            f"",
            f"---",
            f"",
            f"## Interview Transcript",
            f"",
        ]
        
        for turn in self.session.history:
            role_label = "**Business User**" if turn["role"] == "user" else "**ARIA**"
            # Skip the initial system setup message
            content = turn["content"]
            if "Please introduce yourself and begin" in content and turn["role"] == "user":
                continue
            lines.append(f"### {role_label}")
            lines.append(f"")
            lines.append(content)
            lines.append(f"")
        
        path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────

def print_message(role: str, content: str, console=None):
    """Print a formatted message to the console."""
    if RICH_AVAILABLE and console:
        if role == "assistant":
            # Strip <requirements> block from display — it's long
            display = re.sub(
                r"<requirements>.*?</requirements>", 
                "\n*[Requirements document generated — see output files]*\n",
                content, flags=re.DOTALL
            )
            panel = Panel(
                Markdown(display),
                title="[bold cyan]ARIA[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
            console.print(panel)
        else:
            console.print(f"\n[dim]You:[/dim] {content}\n")
    else:
        if role == "assistant":
            print(f"\n{'─'*60}")
            print("ARIA:")
            # Strip requirements block for readability
            display = re.sub(
                r"<requirements>.*?</requirements>", 
                "\n[Requirements document generated — see output files]\n",
                content, flags=re.DOTALL
            )
            print(display)
            print(f"{'─'*60}\n")
        else:
            print(f"\nYou: {content}")


def run_cli(output_dir: Path, session_file: Optional[Path], verbose: bool):
    """Run the interactive CLI interview session."""
    console = Console() if RICH_AVAILABLE else None
    
    if RICH_AVAILABLE:
        console.print(Rule("[bold cyan]ARIA — Requirements Intelligence Agent[/bold cyan]"))
        console.print(
            "[dim]Type your responses below. "
            "Press Ctrl+C or type 'exit' to quit.[/dim]\n"
        )
    else:
        print("=" * 60)
        print("ARIA — Requirements Intelligence Agent")
        print("Type 'exit' to quit")
        print("=" * 60)
    
    session = Session(session_file)
    agent = ARIAAgent(output_dir=output_dir, session=session, verbose=verbose)
    
    # Start or resume
    opening = agent.start_interview()
    print_message("assistant", opening, console)
    
    # Interview loop
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("[bold]You[/bold]")
            else:
                user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession saved. Run again with --session to resume.")
            break
        
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Session ended. Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        response, requirements = agent.send_message(user_input)
        print_message("assistant", response, console)
        
        if requirements:
            if RICH_AVAILABLE:
                console.print(
                    f"\n[bold green]✓ Requirements document saved to {output_dir}[/bold green]"
                )
                console.print(
                    "[dim]Start a new session to capture another use case.[/dim]\n"
                )
            else:
                print(f"\n✓ Requirements saved to {output_dir}")
            break
    
    if verbose and session.requirements:
        print("\nFinal requirements summary:")
        print(json.dumps(session.requirements, indent=2))


# ─────────────────────────────────────────────────────────────
# PROGRAMMATIC API (for integration)
# ─────────────────────────────────────────────────────────────

class ARIASession:
    """
    High-level API for integrating ARIA into other applications.
    
    Example:
        session = ARIASession()
        greeting = session.start()
        
        while not session.is_complete:
            response = session.reply(user_message)
            print(response.text)
        
        requirements = session.requirements
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self._output_dir = Path(output_dir) if output_dir else Path("./requirements")
        self._session = Session()
        self._agent = ARIAAgent(output_dir=self._output_dir, session=self._session)
        self.is_complete = False
        self.requirements: Optional[dict] = None
    
    def start(self) -> str:
        """Start the interview. Returns ARIA's opening message."""
        return self._agent.start_interview()
    
    def reply(self, message: str) -> "ARIAResponse":
        """Send a user message. Returns an ARIAResponse object."""
        text, requirements = self._agent.send_message(message)
        if requirements:
            self.is_complete = True
            self.requirements = requirements
        return ARIAResponse(text=text, requirements=requirements)
    
    @property
    def history(self) -> list[dict]:
        return self._session.history


class ARIAResponse:
    """Response from an ARIA interview turn."""
    
    def __init__(self, text: str, requirements: Optional[dict] = None):
        self.text = text
        self.requirements = requirements
        self.is_final = requirements is not None
        
        # Strip <requirements> from display text
        self.display_text = re.sub(
            r"<requirements>.*?</requirements>", "", text, flags=re.DOTALL
        ).strip()
        
        if requirements:
            self.executive_summary = extract_summary(text)
        else:
            self.executive_summary = None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARIA — Energy Transmission Requirements Intelligence Agent"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("./requirements"),
        help="Directory to save requirements JSON and transcripts (default: ./requirements)"
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Path to a saved session JSON file to resume"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print additional debug information"
    )
    
    args = parser.parse_args()
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Get your API key at: https://console.anthropic.com")
        sys.exit(1)
    
    run_cli(
        output_dir=args.output,
        session_file=args.session,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
