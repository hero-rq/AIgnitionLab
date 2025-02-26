# Filename: automated_code_generation_manager.py

import random
from copy import copy, deepcopy
import sys
import os
import logging
import warnings
from pathlib import Path
from abc import abstractmethod
from contextlib import contextmanager

# External references (assuming these come from your project)
# from common_imports import execute_code, query_model, extract_prompt, remove_figures
# from tools import remove_figures
# from inference import query_model, extract_prompt, execute_code

#########################################################
## Suppress stdout context
#########################################################
@contextmanager
def suppress_stdout():
    """
    Context manager to cleanly suppress standard output, 
    typically used when executing code we do not want 
    to clutter the console logs with.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#########################################################
## Global Settings
#########################################################
os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)

# Attempts to fix code if the code generation leads to errors
GLOBAL_REPAIR_ATTEMPTS = 2

#########################################################
## Command Abstraction
#########################################################
class Command:
    """
    Abstract representation of a 'command' used for 
    editing or replacing code. Derived classes handle 
    specific behavior such as partial line edits or 
    full code replacement.
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
## Replace Command
#########################################################
class Replace(Command):
    """
    The Replace command is used to entirely replace 
    the existing code with new code.
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "This tool allows you to entirely re-write/replace all current code.\n"
            "Invoke with:\n"
            "  REPLACE\n"
            "  <new code here>\n"
            "This will be tested before becoming the final code.\n"
            "Use sparingly when you need to drastically change the code."
        )

    def execute_command(self, *args) -> str:
        """
        Actually perform the replacement. 
        args[0] -> new code lines
        """
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the command string is recognized 
        as a REPLACE command.
        """
        if "
REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Parse the given command string to extract the 
        new replacement code, test it, and return results.
        """
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret:
            return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)

#########################################################
## Edit Command
#########################################################
class Edit(Command):
    """
    The Edit command is used to replace a contiguous block 
    of lines (indexed n through m) of the current code 
    with new lines.
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "Invoke with:\n"
            "  EDIT N M\n"
            "  <new lines to replace old lines>\n"
            "where N is the first line index to replace, M is the last line index.\n"
            "This removal is inclusive of N and M.\n"
            "The replaced code is tested before acceptance."
        )

    def execute_command(self, *args) -> str:
        """
        Perform the code editing on lines [N : M].
        args[0] -> N (int)
        args[1] -> M (int)
        args[2] -> old code lines
        args[3] -> new lines to replace
        args[4] -> entire dataset or environment code to execute
        """
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1] + 1)))

            # Remove lines N through M
            for _ln in lines_to_replace:
                current_code.pop(_ln)

            # Insert new lines
            for _line in lines_to_add:
                current_code.insert(args[0], _line)

            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            code_ret = execute_code(code_exec)
            if "CODE EXECUTION ERROR" in code_ret:
                return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the command string is recognized 
        as an EDIT command.
        """
        if "
EDIT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Parse the command string to find the line 
        indexes and the new code snippet.
        """
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0:
                return False, None

            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, None

            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0:
                return False, None

            return success, (
                lines_to_edit[0], 
                lines_to_edit[1], 
                codelines, 
                text[1:], 
                datasetcode
            )
        except Exception as e:
            return False, (None, None, None, None, None)

#########################################################
## Helper Functions for Scoring and Repair
#########################################################
def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    """
    Queries a 'reward model' to gauge how well the 
    current code and output fulfill the outlined plan.
    Returns a float between 0 and 1.
    """
    e = str()
    for _attempt in range(attempts):
        try:
            # In a real environment, you'd have the reward model 
            # that evaluates correctness, plan adherence, etc.
            sys_prompt = (
                "You are a professor agent who is serving as an "
                "expert reward model that can read a research plan, "
                "research code, and code output, then produce a float "
                "score from 0 to 1.\n\n"
                "Output strictly in the format:\n"
                "SCORE\n<score here>\n"
            )
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys_prompt,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Research Plan: {outlined_plan}\n\n"
                    f"Code: \n{code}\n\n"
                    f"Code Output: {code_return}\n\n"
                ),
                temp=0.6
            )
            performance = extract_prompt(scoring, "SCORE")
            performance = float(performance)
            return performance, f"The performance is: {performance}", True
        except Exception as e:
            return None, str(e), False
    return 0, e

def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    """
    Attempts to fix the code based on the error.
    Different fix strategies for 'replace' vs 'edit'.
    """
    if ctype == "replace":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal: fix the code so that the same "
            "error does not recur, while preserving logic.\n"
            "Wrap final code in:\n"
            "```python\n<code>\n```\n"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Error: {error}\n\nCode:\n{code}",
            temp=0.8
        )
        return extract_prompt(model_resp, "python")

    elif ctype == "edit":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal: fix the code so that the same "
            "error does not recur, while preserving logic.\n\n"
            "============= CODE EDITING TOOL =============\n"
            "You can fix the code with this pattern:\n"
            "  EDIT N M\n"
            "  <replacement lines>\n"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Error: {error}\n\nCode:\n{code}",
            temp=0.2
        )
        return model_resp

#########################################################
## CodeAutomationManager Class
#########################################################
class CodeAutomationManager:
    """
    Main class that orchestrates the entire process 
    of generating, editing, and replacing code 
    automatically based on feedback from a reward model.
    
    Workflow:
      1. Generate initial code and test it.
      2. Score the code with a 'reward model.'
      3. If the score is low or there's an error, attempt 
         to automatically edit or replace code with new 
         commands.
      4. Reflect and refine until a desired threshold 
         or attempt limit is reached.
    """
    def __init__(
        self,
        dataset_code, 
        openai_api_key=None, 
        notes=None, 
        max_steps=10, 
        insights=None, 
        plan=None, 
        llm_str=None
    ):
        # Basic initialization
        self.notes = notes if notes else []
        self.dataset_code = dataset_code
        self.plan = plan if plan else ""
        self.llm_str = llm_str
        self.verbose = False

        self.max_codes = 2
        self.st_hist_len = 2
        self.min_gen_trials = 2
        self.code_lines = str()
        self.st_history = []
        self.insights = insights
        self.code_reflect = ""
        self.max_steps = max_steps
        self.prev_code_ret = ""
        self.should_execute_code = True
        self.openai_api_key = openai_api_key

        # Commands available to the system
        self.commands = [Replace()]

        # For storing best code among multiple tries
        self.best_score = None
        self.best_codes = []

        # Track success code
        self.prev_working_code = None

    def initial_solve(self):
        """
        Kick off the entire automated code generation 
        and test the initial code creation.
        """
        self.best_score = None
        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [
            (copy(init_code), self.best_score, init_return)
        ]

        self.code_lines = init_code
        self.commands = [Edit(), Replace()]
        self.prev_working_code = copy(self.code_lines)

    def gen_initial_code(self):
        """
        Produce an initial code template from the LLM 
        using the REPLACE command.
        """
        num_attempts = 0
        error_hist = []

        while True:
            if num_attempts == 0:
                err = ""
                err_hist = ""
            else:
                err = (
                    f"Previous attempt had an error. "
                    f"Do not repeat it. Attempt {num_attempts}"
                )
                error_hist.append(err)
                if len(error_hist) == 5:
                    _ = error_hist.pop(0)
                err = "\n".join(error_hist)
                err_hist = (
                    "Previous errors were:\n" + 
                    err + 
                    "\nTry to solve them differently."
                )

            # Get LLM response with REPLACE command
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.llm_str,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\nUse REPLACE to create initial code:\n",
                temp=1.0
            )
            model_resp = self.clean_text(model_resp)
            (
                cmd_str, 
                code_lines, 
                prev_code_ret, 
                should_execute_code, 
                score
            ) = self.process_command(model_resp)

            print(f"@@@ INIT ATTEMPT: {num_attempts}, cmd: {cmd_str}")
            print(f"$$$ Score: {score}")

            if score is not None:
                # Successfully generated code with no errors
                break
            num_attempts += 1

        return code_lines, prev_code_ret, score

    def solve(self):
        """
        Main iterative loop to refine the code 
        until we reach a stable or high score solution.
        """
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_code_ret = None
        self.should_execute_code = False

        while True:
            if len(self.commands) == 2:
                cmd_app_str = (
                    "You must output either EDIT or REPLACE.\n"
                )
            else:
                cmd_app_str = ""

            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.llm_str,
                system_prompt=self.system_prompt(),
                prompt=(
                    f"History:\n{self.history_str()}\n\n"
                    f"{cmd_app_str}"
                    "Now please enter a command:\n"
                ),
                temp=1.0
            )
            model_resp = self.clean_text(model_resp)

            # Choose from best existing code as a starting point 
            # for the next iteration
            self.code_lines = copy(random.choice(self.best_codes)[0])

            cmd_str, code_lines, prev_code_ret, should_execute_code, score = \
                self.process_command(model_resp)

            self.st_history.append([
                model_resp, 
                prev_code_ret, 
                code_lines, 
                cmd_str
            ])
            if len(self.st_history) > self.st_hist_len:
                self.st_history.pop(0)

            if score is not None:
                if top_score is None:
                    best_pkg = (
                        copy(code_lines), 
                        copy(prev_code_ret), 
                        copy(should_execute_code), 
                        copy(model_resp), 
                        copy(cmd_str)
                    )
                    top_score = score
                elif score > top_score:
                    best_pkg = (
                        copy(code_lines), 
                        copy(prev_code_ret), 
                        copy(should_execute_code), 
                        copy(model_resp), 
                        copy(cmd_str)
                    )
                    top_score = score

            print(f"@@@ Command Exec (attempt {num_attempts}): {cmd_str}")
            print(f"$$$ Score: {score}")

            if num_attempts >= self.min_gen_trials and top_score is not None:
                break

            num_attempts += 1

        # Acquire final best code
        (
            self.code_lines, 
            self.prev_code_ret, 
            self.should_execute_code, 
            model_resp, 
            cmd_str
        ) = best_pkg

        # Insert the top scoring code among best codes
        if top_score > self.best_codes[-1][1]:
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                self.code_reflect = self.reflect_code()
            self.best_codes.append((
                copy(self.code_lines), 
                copy(top_score), 
                self.prev_code_ret
            ))
            self.best_codes.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

    def reflect_code(self):
        """
        Creates a reflection or summary from analyzing 
        top performed code to glean generalizable insights 
        that can be used in further improvements.
        """
        code_strs = ("$"*40 + "\n\n").join([
            self.generate_code_lines(item[0]) + 
            f"\nCode Return: {item[1]}" 
            for item in self.best_codes
        ])
        code_strs = (
            "Reflect on the following best code sets:\n" + 
            code_strs + 
            "\nProvide insights to improve performance further.\n"
        )
        syst = self.system_prompt(commands=False) + code_strs

        return query_model(
            prompt=(
                "Based on these best codes, please reflect on how "
                "to further improve the approach. Provide line-by-line "
                "suggestions or general strategies."
            ),
            system_prompt=syst,
            model_str=self.llm_str,
            openai_api_key=self.openai_api_key
        )

    def process_command(self, model_resp):
        """
        Interprets the LLM's command and executes the 
        corresponding code modifications.
        
        Returns:
          cmd_str, 
          code_lines, 
          prev_code_ret, 
          should_execute_code, 
          score
        """
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)

        remove_figures()  # Clears any existing figures if relevant

        with suppress_stdout():
            for cmd in self.commands:
                if cmd.matches_command(model_resp):
                    # Edit command
                    if cmd.cmd_type == "CODE-edit":
                        score = None
                        failed = True
                        code_err = ""

                        for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                            success, args = cmd.parse_command(
                                model_resp, 
                                copy(self.code_lines), 
                                self.dataset_code
                            )

                            if success:
                                cmd_return = cmd.execute_command(args)
                                code_err = (
                                    f"Execute code result: {cmd_return[2]}"
                                )
                                if cmd_return[0]:
                                    code_lines = copy(cmd_return[1])
                                    score, cmd_str, is_valid = get_score(
                                        self.plan,
                                        "\n".join(code_lines),
                                        cmd_return[2],
                                        openai_api_key=self.openai_api_key,
                                        REWARD_MODEL_LLM=self.llm_str
                                    )
                                    if is_valid:
                                        failed = False
                                        break
                                    code_err += f"\nScore error: {cmd_str}"

                            # Attempt to repair after failure
                            repaired_code = code_repair(
                                model_resp, 
                                code_err, 
                                REPAIR_LLM=self.llm_str, 
                                ctype="edit", 
                                openai_api_key=self.openai_api_key
                            )
                            model_resp = repaired_code
                            print(f"  -> Attempting repair: try {_tries}")

                        if failed:
                            cmd_str = (
                                "Code editing FAILED, error:\n" + code_err
                            )
                            print("$$$ CODE EDIT (failed)")
                        else:
                            cmd_str = "Code successfully edited."
                            prev_code_ret = copy(cmd_return[2])
                            print("$$$ CODE EDIT (success)")
                            should_execute_code = True

                        return cmd_str, code_lines, prev_code_ret, should_execute_code, score

                    # Replace command
                    elif cmd.cmd_type == "CODE-replace":
                        score = None
                        failed = True
                        code_err = ""

                        for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                            success, args = cmd.parse_command(
                                model_resp,
                                self.dataset_code
                            )
                            code_err = (
                                f"Execute code result: {args[1]}"
                                if args and len(args) > 1 else ""
                            )

                            if success:
                                code_lines = copy(args[0])
                                score, cmd_str, is_valid = get_score(
                                    self.plan,
                                    "\n".join(code_lines),
                                    args[1],
                                    openai_api_key=self.openai_api_key,
                                    REWARD_MODEL_LLM=self.llm_str
                                )
                                if is_valid:
                                    failed = False
                                    break
                                code_err += (
                                    f"\nScore error on test set: {cmd_str}"
                                )

                            # Attempt to repair after failure
                            repaired_code = code_repair(
                                extract_prompt(model_resp, "REPLACE"),
                                code_err,
                                ctype="replace",
                                openai_api_key=self.openai_api_key,
                                REPAIR_LLM=self.llm_str
                            )
                            repaired_code = f"
REPLACE\n{repaired_code}\n
"
                            model_resp = repaired_code
                            print(f"  -> Attempting repair: try {_tries}")

                        if failed:
                            cmd_str = (
                                "Code replacement FAILED, error:\n" + code_err
                            )
                            print("$$$ CODE REPLACE (failed)")
                        else:
                            cmd_str = "Code was successfully replaced."
                            code_lines = copy(args[0])
                            prev_code_ret = copy(args[1])
                            print("$$$ CODE REPLACE (success)")
                            should_execute_code = True

                        return cmd_str, code_lines, prev_code_ret, should_execute_code, score

            # If no known command was found
            print("$$$ INVALID COMMAND (failed)")
            return (
                "Command not supported; choose from EDIT or REPLACE.", 
                None, 
                None, 
                None, 
                None
            )

    def history_str(self):
        """
        Returns a text summary of short-term history 
        for the system prompt or debugging.
        """
        hist_str = ""
        for i, record in enumerate(self.st_history):
            hist_str += (
                f"-------- History ({len(self.st_history) - i} steps ago) -----\n"
            )
            hist_str += (
                f"LLM Response: {record[0]}\n"
                f"Command Output: {record[3]}\n"
                f"Code Lines:\n{'#'*20}\n{record[2]}\n{'#'*20}\n"
                f"Feedback: {record[1]}\n"
                f"-------- End of history -----\n"
            )
        return hist_str

    def system_prompt(self, commands=True):
        """
        Returns the system prompt that instructs the 
        LLM on how to handle code generation tasks, 
        including plan and dataset details.
        """
        system_pieces = [
            self.role_description(),
            f"Task instructions: {self.phase_prompt()}",
            f"Literature insights: {self.insights}",
            self.code_reflect,
            f"Notes: {self.notes}",
            f"Plan to follow: {self.plan}",
            f"Dataset info (auto-added at code start): {self.dataset_code}",
            "Your method must not yield 0% accuracy if it is a ML task.\n"
            "Aim to generate at least two figures if relevant.\n"
        ]

        if commands:
            system_pieces.append(self.command_descriptions())

        return "\n".join(system_pieces)

    def generate_code_lines(self, code):
        """
        Helper to produce code with line numbers for 
        easier edits.
        """
        codestr = ""
        for i, line in enumerate(code):
            codestr += f"{i} |{line}\n"
        return codestr

    def feedback(self, code_return):
        """
        Provide execution feedback. If there's an error, 
        reflect on possible improvements. If no error, 
        check for submission or further improvements.
        """
        if code_return is not None:
            code_str = self.generate_code_lines(self.code_lines)
            if "[CODE EXECUTION ERROR]" in code_return:
                reflect_prompt = (
                    f"This code returned the following error:\n{code_return}\n"
                    f"Code:\n{code_str}\n"
                    "Reflect on lines causing the error and how to fix them. "
                    "Do not provide entirely new code, just your reflection."
                )
            elif os.path.exists("submission.csv"):
                grade_return = get_score(
                    self.plan,
                    "\n".join(self.prev_working_code),
                    code_return,
                    openai_api_key=self.openai_api_key
                )[0]
                reflect_prompt = (
                    f"Submission CSV found. Score: {grade_return}\n"
                    f"Code:\n{code_str}\n"
                    "Reflect on how to improve performance or approach further."
                )
                for file in os.listdir("."):
                    if file.endswith(".csv"):
                        os.system(f"rm {file}")
            else:
                reflect_prompt = (
                    "No error, but no submission CSV found.\n"
                    "Reflect on how you can refine or finalize the solution.\n"
                    f"Code:\n{code_str}\n"
                )
        else:
            code_return = "No changes were made to the code."
            reflect_prompt = "Reflect on future improvements to code."

        reflection = self.reflection(reflect_prompt, code_return)
        return f"Code Return: {code_return}\nReflection:\n{reflection}"

    def reflection(self, reflect_prompt, code_return):
        """
        Provides a reflection of the code and output 
        using an LLM, with the system prompt as context.
        """
        refl = query_model(
            prompt=reflect_prompt,
            system_prompt=self.system_prompt(commands=False),
            model_str=self.llm_str,
            openai_api_key=self.openai_api_key
        )
        return (
            f"The code returned:\n{code_return}\n"
            f"Reflection from model:\n{refl}\n"
        )

    def generate_dataset_descr_prompt(self):
        """
        Additional dataset description prompt (not strictly needed).
        """
        return (
            f"\nAuto-included dataset code:\n"
            f"{self.dataset_code}"
        )

    def phase_prompt(self):
        """
        A short descriptive text on the overall phase or 
        purpose for the code generation.
        """
        return (
            "You are creating code for a research or automation project. "
            "Try to implement the plan thoroughly while keeping the code "
            "as simple and robust as possible."
        )

    def role_description(self):
        """
        Summarizes the role of this system in more direct terms.
        """
        return (
            "You are an advanced automated code generation system. "
            "Your goal is to produce code that satisfies a plan, "
            "handle errors, and refine code until you reach a "
            "satisfactory solution."
        )

    @staticmethod
    def _common_code_errors():
        """
        Warnings or suggestions about typical pitfalls 
        encountered in code generation.
        """
        return (
            "Make sure imports are correct.\n"
            "Ensure typed commands (EDIT or REPLACE) are used properly.\n"
            "Avoid illegal or unsupported libraries. Use Python standard or well-known libs.\n"
            "Test thoroughly after each code update.\n"
        )

    def command_descriptions(self):
        """
        Concatenate docstrings of each available command plus 
        general usage hints.
        """
        cmd_strings = "\n".join([cmd.docstring() for cmd in self.commands])
        usage_info = (
            "You can only use one command at a time. The format is:\n"
            "  <COMMAND>\n"
            "  <content>\n"
            "  \n"
            "where <COMMAND> is EDIT or REPLACE, etc."
        )
        return (
            f"{usage_info}\n"
            f"{self._common_code_errors()}\n"
            f"{cmd_strings}"
        )

    def clean_text(self, text):
        """
        Helper to adapt raw LLM outputs to match internal 
        expected command format.
        """
        text = text.replace("
\n", "
")
        text = text.replace("
python\n", "
REPLACE\n")
        return text

    def run_code(self):
        """
        Execute the code we've settled on, if no prior result 
        is saved.
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            return execute_code("\n".join(self.code_lines))
        return "No new code changes to execute."
