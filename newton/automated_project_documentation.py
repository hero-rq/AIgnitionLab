# Filename: automated_project_documentation.py

import random
import string
import sys
import os

from copy import copy, deepcopy
from pathlib import Path
from abc import abstractmethod
from contextlib import contextmanager

# These come from your own utilities:
# from utils import *
# from tools import *
# from inference import *
# from common_imports import *
# from agents import get_score

#########################################################
## STDOUT SUPPRESSION
#########################################################
@contextmanager
def suppress_stdout():
    """
    Context manager to cleanly suppress standard output, 
    typically used when executing code we do not want 
    to clutter the console with.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#########################################################
## Basic Command Abstraction
#########################################################
class Command:
    """
    Abstract representation of a 'command' used to 
    edit or replace blocks of LaTeX-like documentation 
    lines. Derived classes handle specific behaviors.
    """
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        pass

#########################################################
## ARXIV-STYLE SEARCH TOOL (Optional)
#########################################################
class ArxivSearch:
    """
    Stubbed class for searching Arxiv. 
    In real usage, you'd implement search and retrieval.
    """
    def find_papers_by_str(self, query, N=10):
        return f"[FAKE] Arxiv search results for query='{query}'"

    def retrieve_full_paper_text(self, paper_id):
        return f"[FAKE] Full text for arXiv paper ID='{paper_id}'"

class DocSearch(Command):
    """
    Demonstrates searching for relevant references/papers 
    to incorporate into project documentation. 
    """
    def __init__(self):
        super().__init__()
        self.search_engine = ArxivSearch()
        self.num_papers_per_search = 10
        self.cmd_type = "SEARCH-doc"

    def docstring(self) -> str:
        return (
            "============= DOCUMENTATION SEARCH TOOL =============\n"
            "You can search for project references or relevant research. "
            "Use the command:\n"
            "```SUMMARY\n<search query>\n```\n"
            "to find relevant papers (or references). Use:\n"
            "```FULL_TEXT\n<arxiv paper id>\n```\n"
            "to retrieve the full text for that paper ID. Make sure you read the full text before citing."
        )

    def execute_command(self, *args) -> str:
        """
        Executes either a SUMMARY or FULL_TEXT command for 
        retrieving references or full content.
        """
        if args[0] == "SUMMARY":
            return self.search_engine.find_papers_by_str(args[1], self.num_papers_per_search)
        elif args[0] == "FULL_TEXT":
            return self.search_engine.retrieve_full_paper_text(args[1])
        raise Exception("Invalid Documentation Search Command")

    def matches_command(self, cmd_str) -> bool:
        if "```SUMMARY" in cmd_str: 
            return True
        elif "```FULL_TEXT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Extracts the relevant query or ID from the command string.
        """
        sum_text = extract_prompt(args[0], "SUMMARY").split("\n")
        full_text = extract_prompt(args[0], "FULL_TEXT").split("\n")

        if len(sum_text) == 0 and len(full_text) == 0:
            return False, None
        if len(sum_text) > 0:
            return True, ("SUMMARY", sum_text,)
        if len(full_text) > 0:
            return True, ("FULL_TEXT", full_text,)

#########################################################
## REPLACE COMMAND: Entirely Overwrite Documentation
#########################################################
class DocReplace(Command):
    """
    Replaces all existing LaTeX lines with a brand-new
    block of content. 
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "DOC-replace"

    def docstring(self) -> str:
        return (
            "============= DOCUMENTATION REPLACE TOOL =============\n"
            "Use this tool to replace all existing documentation.\n"
            "Command format:\n"
            "```REPLACE\n<new documentation>\n```\n"
            "The system will compile your new documentation. "
            "If it fails, the replacement is reverted."
        )

    def execute_command(self, *args) -> str:
        """
        Actually perform the replacement. 
        args[0] -> new LaTeX lines
        """
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the text indicates a REPLACE command.
        """
        if "```REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Extract the new LaTeX text for replacement and 
        attempt to compile or validate it.
        """
        new_latex = extract_prompt(args[0], "REPLACE")
        latex_ret = compile_latex(new_latex, compile=args[1])  # compile=args[1] is a bool
        if "[CODE EXECUTION ERROR]" in latex_ret:
            return False, (None, latex_ret,)
        return True, (new_latex.split("\n"), latex_ret)

#########################################################
## EDIT COMMAND: Replace Lines [N:M] in Documentation
#########################################################
class DocEdit(Command):
    """
    Replaces lines N through M in the existing documentation 
    with new lines.
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "DOC-edit"

    def docstring(self) -> str:
        return (
            "============= DOCUMENTATION EDIT TOOL =============\n"
            "Use the command:\n"
            "```EDIT N M\n<new lines>\n```\n"
            "to replace lines [N through M] of the existing LaTeX-like docs.\n"
            "Changes are validated by a compile check."
        )

    def execute_command(self, *args) -> str:
        """
        Actually perform the line replacement.
        args format:
          (N, M, doc_lines, new_lines_to_insert, compile_bool)
        """
        try:
            args = args[0]
            current_latex = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))

            # Remove lines N through M
            for _ln in lines_to_replace:
                current_latex.pop(_ln)

            # Insert new lines
            for _line in lines_to_add:
                current_latex.insert(args[0], _line)

            new_latex = "\n".join(current_latex)
            latex_ret = compile_latex(new_latex, compile=args[4])

            if "error" in latex_ret.lower():
                return (False, None, latex_ret)
            return (True, current_latex, latex_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the text matches the EDIT command pattern.
        """
        if "```EDIT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Parse out the integer line indices (N, M) and 
        the block of new lines.
        """
        cmd_str, doc_lines = args[0], args[1]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0:
                return False, (None, None, None, None)

            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, (None, None, None, None)

            lines_to_edit = [int(v) for v in lines_to_edit]
            if len(text[1:]) == 0:
                return False, (None, None, None, None)

            return success, (
                lines_to_edit[0], 
                lines_to_edit[1], 
                doc_lines, 
                text[1:]
            )
        except Exception as e:
            return False, (None, None, None, None)

#########################################################
## SECTIONAL TIPS FOR PROJECT DOCUMENTATION (EXAMPLE)
#########################################################
doc_section_tips = {
    "overview": """
- Provide a concise overview of the project's purpose and 
  main challenges addressed.
- Summarize the approach or architecture in an accessible way.
- Emphasize the relevance/impact of the project in a few lines.
""",
    "setup": """
- Describe environment setup, dependencies, or relevant 
  hardware details (if critical).
- Summarize how to configure or run the code.
""",
    "implementation": """
- Outline critical parts of the code or architecture.
- Include relevant code snippets in LaTeX form if needed. 
- Use thorough descriptions of data structures, classes, 
  and methods.
""",
    "results_and_analysis": """
- Provide results from tests or experiments (like performance stats).
- Include comparisons to known baselines or expected outputs.
- If relevant, add diagrams or figures referencing included images.
- Discuss possible limitations or edge cases.
""",
    "future_extensions": """
- Brainstorm or propose additional features or improvements.
- Suggest ways to adapt the code or approach to new contexts.
"""
}

#########################################################
## Placeholder for "Compile" Function
#########################################################
def compile_latex(latex_content, compile=True):
    """
    Stubbed function simulating a LaTeX compile or 
    some form of doc validation. 
    """
    if not compile:
        return "[NO COMPILE] LaTeX accepted."
    # In real usage, you'd do something like:
    #   result = do_latex_compile(latex_content)
    #   if result.success: return "Compilation successful."
    #   else: return "[CODE EXECUTION ERROR]"
    return "Compilation successful."

#########################################################
## Placeholder Extract & Query (LLM) Functions
#########################################################
def extract_prompt(full_text, key_str):
    """
    Example function: extracts text from full_text between 
    triple backticks ``` and some KEY, e.g. 'REPLACE'.
    This is a stub; you might already have a more robust 
    version in your codebase.
    """
    # A super-simplistic approach:
    if key_str not in full_text:
        return ""
    # E.g. parse text after ```KEY and stop at next triple ```
    # Stub logic just returning everything after the key
    # up to next triple backticks.
    return full_text.split(key_str)[-1].replace("```", "").strip()

def query_model(model_str, system_prompt, prompt, temp=1.0, openai_api_key=None):
    """
    Stub for querying an LLM. 
    Return a string that includes your commanded format.
    """
    return f"[FAKE MODEL OUTPUT] {prompt}"

#########################################################
## MAIN PROJECT DOCUMENTATION MANAGER
#########################################################
class ProjectDocumentationManager:
    """
    A manager that orchestrates writing and refining 
    documentation for a software project or experiment, 
    using an LLM to automatically build LaTeX-like docs.
    """
    def __init__(
        self,
        llm_str,
        notes=None, 
        max_steps=10, 
        insights=None, 
        plan=None, 
        exp_code=None, 
        exp_results=None, 
        additional_review=None, 
        reference_docs=None, 
        topic=None, 
        openai_api_key=None, 
        compile_pdf=True
    ):
        # Basic config / fallback defaults
        self.llm_str = llm_str
        self.openai_api_key = openai_api_key
        self.compile_pdf = compile_pdf
        self.max_steps = max_steps

        self.notes = notes if notes else []
        self.insights = insights if insights else ""
        self.plan = plan if plan else ""
        self.exp_code = exp_code if exp_code else ""
        self.exp_results = exp_results if exp_results else ""
        self.additional_review = additional_review if additional_review else ""
        self.reference_docs = reference_docs if reference_docs else []
        self.topic = topic if topic else ""

        # For storing best solutions
        self.best_score = None
        self.best_report = []
        self.prev_working_doc = []
        self.doc_lines = []
        self.prev_doc_ret = ""

        self.commands = [DocReplace()]  # later we add DocEdit
        self.model = self.llm_str

        # For searching references
        self.section_references = {}

    ###################################################################
    ## CORE INTERFACE
    ###################################################################
    def initial_solve(self):
        """
        Initialize the doc creation with a first pass 
        (DocReplace) to build a minimal structure or scaffolding.
        """
        self.best_score = None
        init_report, init_return, self.best_score = self.gen_initial_report()
        self.best_report = [(copy(init_report), self.best_score, init_return)]
        self.doc_lines = init_report
        self.commands = [DocEdit(), DocReplace()]
        self.prev_working_doc = copy(self.doc_lines)

    def solve(self):
        """
        Iterative refinement of the documentation:
          - The system queries for a command (EDIT or REPLACE).
          - Executes it, re-checks, and potentially does 
            multiple attempts to produce a high-scoring doc.
        """
        num_attempts = 0
        max_papers = 1  # how many "best" variants to keep
        top_score = None
        best_pkg = None

        while True:
            # Use random best doc as starting
            self.doc_lines = copy(random.choice(self.best_report)[0])

            # LLM suggests next command
            model_resp = query_model(
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt="\nPlease enter a command (EDIT or REPLACE): ",
                temp=1.0,
                openai_api_key=self.openai_api_key
            )
            model_resp = self.clean_text(model_resp)

            cmd_str, doc_lines, prev_ret, score = self.process_command(model_resp)

            if score is not None:
                if top_score is None:
                    best_pkg = (copy(doc_lines), copy(prev_ret), copy(model_resp), copy(cmd_str))
                    top_score = score
                elif score > top_score:
                    best_pkg = (copy(doc_lines), copy(prev_ret), copy(model_resp), copy(cmd_str))
                    top_score = score

            print(f"@@@ Command Exec // Attempt {num_attempts}: {cmd_str}")
            print(f"$$$ Score: {score}")
            num_attempts += 1

            if num_attempts >= 2 and top_score is not None:
                break

        # Accept final best doc
        if best_pkg is not None:
            self.doc_lines, self.prev_doc_ret, model_resp, cmd_str = best_pkg

        if top_score is None:
            top_score = 0.0

        # Insert into best_report if it surpasses last known best
        if top_score > self.best_report[-1][1]:
            if len(self.best_report) >= max_papers:
                self.best_report.pop(-1)
            self.best_report.append((copy(self.doc_lines), copy(top_score), self.prev_doc_ret))
            self.best_report.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

    ###################################################################
    ## INITIAL REPORT CREATION
    ###################################################################
    def gen_initial_report(self):
        """
        Create an initial skeleton or scaffolding with 
        DocReplace commands.
        """
        # For demonstration, let's produce just a single 
        # doc scaffolding. In a more complex system, 
        # you might break it into multiple sections 
        # like 'Overview', 'Implementation', etc.

        # We'll do a single pass of "scaffolding" below.
        num_attempts = 0
        doc_text = ""
        compiled_ret = ""
        while True:
            prompt = (
                "Please create a scaffolding for the documentation, "
                "including placeholders for sections like [OVERVIEW], "
                "[IMPLEMENTATION], [RESULTS_AND_ANALYSIS], [FUTURE_EXTENSIONS].\n"
                "Respond with: ```REPLACE\n<LaTeX-like doc>\n```"
            )
            model_resp = query_model(
                model_str=self.model,
                system_prompt=self.system_prompt(section="scaffold"),
                prompt=prompt,
                temp=0.8,
                openai_api_key=self.openai_api_key
            )
            model_resp = self.clean_text(model_resp)

            cmd_str, doc_lines, doc_ret, score = self.process_command(model_resp, scoring=False)

            print(f"@@@ INIT ATTEMPT {num_attempts} => {cmd_str}")
            num_attempts += 1

            if doc_lines is not None and len(doc_lines) > 0:
                doc_text = "\n".join(doc_lines)
                compiled_ret = doc_ret
                break
            if num_attempts > 3:
                break

        # For an initial pass, let's give a placeholder score = 0.5
        return doc_text.split("\n"), compiled_ret, 0.5

    ###################################################################
    ## PROCESS COMMAND
    ###################################################################
    def process_command(self, model_resp, scoring=True):
        """
        Parses the command from model_resp and executes it 
        (DOC-EDIT or DOC-REPLACE). 
        """
        cmd_str = None
        score = None
        prev_ret = self.prev_doc_ret
        doc_lines = copy(self.doc_lines)

        # Possibly fix figure references for local paths
        if "\\includegraphics[width=\\textwidth]{Figure_1.png}" in model_resp or \
           "\\includegraphics[width=\\textwidth]{Figure_2.png}":
            cwd = os.getcwd()
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_1.png}",
                f"\\includegraphics[width=\\textwidth]{{{cwd}/Figure_1.png}}"
            )
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_2.png}",
                f"\\includegraphics[width=\\textwidth]{{{cwd}/Figure_2.png}}"
            )

        for cmd in self.commands:
            if cmd.matches_command(model_resp):

                # DOC-EDIT
                if cmd.cmd_type == "DOC-edit":
                    success, parsed = cmd.parse_command(model_resp, doc_lines)
                    if not success:
                        cmd_str = "Documentation edit parse failed."
                        return cmd_str, None, None, None

                    doc_result = cmd.execute_command((parsed[0], parsed[1], doc_lines, parsed[3], self.compile_pdf))
                    if not doc_result[0]:
                        cmd_str = f"Doc edit failed => {doc_result[2]}"
                        return cmd_str, None, None, None

                    # doc_result => (True, updated_lines, compile_ret)
                    doc_lines = copy(doc_result[1])
                    prev_ret = copy(doc_result[2])

                    # Possibly do a scoring check
                    if scoring:
                        # e.g. use your own get_score or a placeholder
                        # score, cmd_str, is_valid = get_score(...)
                        # We'll just mimic that we always succeed with 0.7
                        score, cmd_str, is_valid = 0.7, "Doc editing scored 0.7", True
                    else:
                        score, cmd_str, is_valid = 0.0, "No scoring performed", True

                    return cmd_str, doc_lines, prev_ret, score

                # DOC-REPLACE
                elif cmd.cmd_type == "DOC-replace":
                    success, parsed = cmd.parse_command(model_resp, self.compile_pdf)
                    if not success:
                        cmd_str = "Documentation replace parse failed."
                        return cmd_str, None, None, None

                    doc_lines = copy(parsed[0])
                    prev_ret = copy(parsed[1])

                    if scoring:
                        score, cmd_str, is_valid = 0.6, "Doc replaced => Score 0.6", True
                    else:
                        score, cmd_str, is_valid = 0.0, "No scoring performed", True

                    return cmd_str, doc_lines, prev_ret, score

        # If we get here => no recognized command
        return "No recognized doc command found.", None, None, None

    ###################################################################
    ## PROMPTS & HELPERS
    ###################################################################
    def system_prompt(self, commands=True, section=None):
        """
        Build the system prompt with context about 
        your project, code, results, and so forth.
        """
        sec_cmd = ""
        if section == "scaffold":
            sec_cmd = (
                "Your goal is to create scaffolding placeholders for the main documentation. "
                "No actual content in each section, just the placeholders or minimal lines. "
            )
        doc_len = len(" ".join(self.doc_lines))
        doc_progress = f"Current doc length is about {doc_len} characters."

        # Command usage text
        cmd_usage = self.command_descriptions() if commands else ""

        # Possibly references or extra reviews
        ref_text = "\n".join(self.reference_docs)

        base_prompt = (
            f"{ref_text}\n"
            f"{self.role_description()}\n"
            f"Plan: {self.plan}\n"
            f"Experiment Code: {self.exp_code}\n"
            f"Results Observed: {self.exp_results}\n"
            f"Insights: {self.insights}\n"
            f"Additional Info: {self.additional_review}\n\n"
            f"{doc_progress}\n"
            f"{sec_cmd}\n"
            f"{cmd_usage}\n"
            "Here is the current documentation:\n"
            f"{self.generate_doc_lines(self.doc_lines)}\n"
        )
        return base_prompt

    def command_descriptions(self):
        """
        Collect docstrings of all available commands, 
        with general usage instructions.
        """
        cmd_strings = "\n".join([cmd.docstring() for cmd in self.commands])
        usage_notes = (
            "You can only use a single command per iteration. "
            "Structure is:\n"
            "```COMMAND\n<contents>\n```\n"
        )
        return f"{usage_notes}\n{cmd_strings}"

    def role_description(self):
        """
        Explanation of your 'role' or context 
        in generating the project documentation.
        """
        return (
            "You are an AI system that auto-generates and refines "
            "project documentation for a software/ML project. "
            "Your objective is to produce a thoroughly organized, "
            "clear, and validated set of LaTeX-form documentation, "
            "suitable for distribution or publication."
        )

    def generate_doc_lines(self, doc_lines):
        """
        Create line-numbered output for user reference 
        (particularly for use with EDIT commands).
        """
        output = ""
        for i, line in enumerate(doc_lines):
            output += f"{i} |{line}\n"
        return output

    def clean_text(self, text):
        """
        Minor text cleanup to ensure triple backticks 
        are handled consistently, etc.
        """
        text = text.replace("```\n", "```")
        return text

# End of code: automated_project_documentation.py
