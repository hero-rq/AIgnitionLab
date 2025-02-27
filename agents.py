
import re
import json
from frameworks import *
from tools_latex import *
from llm_engine import *
from lab_coordinator import *
from code_engine import * 
from automated_project_documentation import * 
from automated_data_search import *

def extract_json_data(llm_output):
    """
    Attempt to extract JSON from a text block. 
    Tries to find JSON enclosed by '```json' and '```'. 
    Fallbacks to searching for '{ }' patterns. 
    Cleans up common issues before returning valid JSON or None.
    """
    pattern_json_block = r"```json(.*?)```"
    matches = re.findall(pattern_json_block, llm_output, re.DOTALL)

    if not matches:
        # Fallback: find any JSON-looking substring in the output
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)

    for candidate in matches:
        candidate_str = candidate.strip()
        try:
            parsed = json.loads(candidate_str)
            return parsed
        except json.JSONDecodeError:
            # Attempt to fix common issues (control characters, etc.)
            try:
                cleaned_str = re.sub(r"[\x00-\x1F\x7F]", "", candidate_str)
                parsed_again = json.loads(cleaned_str)
                return parsed_again
            except json.JSONDecodeError:
                continue

    return None  # No valid JSON found after attempts

def evaluate_performance(
    draft_plan, 
    compiled_text, 
    feedback_model, 
    assessor_style=None, 
    attempts=3, 
    api_token=None
):
    """
    Calls an LLM (the 'feedback_model') to review the given plan and compiled_text. 
    It returns a numeric 'rating' (in the 0-10 range) and a string that includes
    the LLM's assessment. If it fails to parse, returns None or an error message.
    """
    last_error = ""
    for _attempt in range(attempts):
        try:
            # Example meta instructions for the model
            template_instructions = """
            Respond in the following format:

            THOUGHT:
            <THOUGHT>

            REVIEW JSON:
            ```json
            <JSON>
            ```

            In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
            In <JSON>, provide the review in JSON format with the following fields:
            - "Summary"
            - "Strengths"
            - "Weaknesses"
            - "Originality" (1-4)
            - "Quality" (1-4)
            - "Clarity" (1-4)
            - "Significance" (1-4)
            - "Questions"
            - "Limitations"
            - "Ethical Concerns" (boolean)
            - "Soundness" (1-4)
            - "Presentation" (1-4)
            - "Contribution" (1-4)
            - "Overall" (1-10)
            - "Confidence" (1-5)
            - "Decision" (Accept or Reject only)
            """
            conference_form = (
                """
                ## Example of a Conference Review Form
                1. Summary: ...
                2. Strengths / Weaknesses: ...
                3. Originality, Quality, Clarity, Significance: ...
                4. Limitations: ...
                5. Ethical concerns: ...
                6. Ratings for Soundness / Presentation / Contribution: ...
                7. Overall rating (1-10)
                8. Confidence rating (1-5)
                9. Final Decision (Accept or Reject)
                """ 
                + template_instructions
            )
            if assessor_style is None:
                assessor_style = ""
            sys_prompt = (
                "You are an advanced AI researcher reviewing a submission. "
                f"Maintain a critical stance. {assessor_style}\n"
                + conference_form
            )
            # Query the LLM
            prompt_content = (
                f"Here is the research plan to be evaluated: {draft_plan}\n\n"
                f"And here is the compiled text/LaTeX content: \n{compiled_text}\n\n"
            )
            evaluation_text = ask_model(
                model_str=feedback_model,
                system_prompt=sys_prompt,
                prompt=prompt_content,
                temp=0.0,
                openai_api_key=api_token
            )
            assessment_json = extract_json_data(evaluation_text)

            # Parse the rating fields from JSON
            overall_score = int(assessment_json["Overall"]) / 10
            soundness_score = int(assessment_json["Soundness"]) / 4
            confidence_score = int(assessment_json["Confidence"]) / 5
            contribution_score = int(assessment_json["Contribution"]) / 4
            presentation_score = int(assessment_json["Presentation"]) / 4
            clarity_score = int(assessment_json["Clarity"]) / 4
            originality_score = int(assessment_json["Originality"]) / 4
            quality_score = int(assessment_json["Quality"]) / 4
            significance_score = int(assessment_json["Significance"]) / 4

            # Weighted factors
            clarity_factor = 0.1
            quality_factor = 0.1
            overall_factor = 1.0
            soundness_factor = 0.1
            confidence_factor = 0.1
            originality_factor = 0.1
            significance_factor = 0.1
            contribution_factor = 0.4
            presentation_factor = 0.2

            max_possible = (
                clarity_factor 
                + quality_factor 
                + overall_factor 
                + soundness_factor 
                + confidence_factor 
                + originality_factor 
                + significance_factor 
                + contribution_factor 
                + presentation_factor
            )

            rating = (
                (
                    soundness_factor * soundness_score 
                    + presentation_factor * presentation_score 
                    + confidence_factor * confidence_score 
                    + contribution_factor * contribution_score 
                    + overall_factor * overall_score 
                    + originality_factor * originality_score 
                    + significance_factor * significance_score 
                    + clarity_factor * clarity_score 
                    + quality_factor * quality_score
                ) 
                / max_possible
            ) * 10

            return rating, (
                f"Performance Rating: {rating}\n"
                + evaluation_text
            ), True

        except Exception as exc:
            last_error = str(exc)
            return None, str(exc), False
    return 0, last_error, False

class AssessorsGroup:
    """
    This class gathers multiple 'assessor' perspectives (similar to reviewers) 
    and aggregates or returns their feedback.
    """
    def __init__(self, model="some-compact-model", notes=None, api_token=None):
        self.notes = [] if notes is None else notes
        self.model = model
        self.api_token = api_token

    def gather_evaluations(self, plan_outline, doc_text):
        """
        Collect multiple evaluations from different style assessors.
        """
        style_1 = "You are a tough but balanced assessor expecting strong experimental results."
        evaluation_1 = evaluate_performance(
            draft_plan=plan_outline,
            compiled_text=doc_text,
            feedback_model=self.model,
            assessor_style=style_1,
            api_token=self.api_token
        )

        style_2 = "You are extremely critical but fair, seeking high field impact in the idea."
        evaluation_2 = evaluate_performance(
            draft_plan=plan_outline,
            compiled_text=doc_text,
            feedback_model=self.model,
            assessor_style=style_2,
            api_token=self.api_token
        )

        style_3 = "You are tough but open-minded, focusing on novelty above all."
        evaluation_3 = evaluate_performance(
            draft_plan=plan_outline,
            compiled_text=doc_text,
            feedback_model=self.model,
            assessor_style=style_3,
            api_token=self.api_token
        )

        return (
            f"Assessor #1:\n{evaluation_1},\n"
            f"Assessor #2:\n{evaluation_2},\n"
            f"Assessor #3:\n{evaluation_3}"
        )


class AbstractModule:
    """
    An abstract foundation for different roles (like SeniorMentor, 
    ExperiencedResearcher, etc.). 
    Each specialized subclass defines their communication style, context, and workflow.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        self.observations = [] if notes is None else notes
        self.limit = max_rounds
        self.model = model
        self.stages = []
        self.blueprint = ""
        self.summary = ""
        self.dialog_history = []
        self.previous_message = ""
        self.previous_summary = ""
        self.results_info = ""
        self.dataset_code = ""
        self.results_code = ""
        self.lit_synthesis = ""
        self.insights = ""
        self.prev_results = ""
        self.assessor_feedback = ""
        self.prev_res_code = ""
        self.prev_insights = ""
        self.api_token = api_token

        self.second_cycle = False
        self.max_history = 15

    def update_model_core(self, model):
        self.model = model

    @staticmethod
    def tidy_text(text):
        return text.replace("```\n", "```")

    def execute_stage(self, research_topic, segment, iteration, external_feedback="", temperature=None):
        """
        The main method for each specialized role, 
        controlling how they respond or produce new output for the current stage.
        """
        sys_prompt = (
            f"You are {self.describe_role()} "
            f"\nGuidelines: {self.stage_instructions(segment)}\n{self.available_commands(segment)}"
        )
        cur_context = self.provide_context(segment)
        combined_history = "\n".join([entry[1] for entry in self.dialog_history])
        relevant_notes = [n for n in self.observations if segment in n.get("phases", [])]
        note_block = f"Additional Observations: {relevant_notes}\n" if len(relevant_notes) > 0 else ""

        final_push = ""
        if iteration / (self.limit - 1) > 0.7:
            final_push = "Ensure you finalize your output soon!"

        full_prompt = (
            f"{cur_context}\n{'~' * 10}\n"
            f"Dialogue history: {combined_history}\n{'~' * 10}\n"
            f"Iteration #{iteration}, Stage: {segment}\n{final_push}\n"
            f"[Objective] The research topic is: {research_topic}\n"
            f"External Feedback: {external_feedback}\n{note_block}\n"
            f"Previous output was: {self.previous_message}. Please differentiate your new output.\n"
            f"Provide only a single command below:\n"
        )
        model_response = ask_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=full_prompt,
            temp=temperature,
            openai_api_key=self.api_token
        )
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", segment, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        model_response = self.tidy_text(model_response)
        self.previous_message = model_response

        # Possibly parse expiration instructions
        step_expiration = None
        if external_feedback is not None and "```EXPIRATION" in external_feedback:
            try:
                first_line = external_feedback.split("\n")[0]
                step_expiration = int(first_line.replace("```EXPIRATION ", ""))
                # (We won't do anything with feedback extraction here, just a parallel to the original code.)
            except:
                pass

        self.dialog_history.append((step_expiration, f"Iteration #{iteration}, Stage: {segment}, ExtFeedback: {external_feedback}, Response: {model_response}"))
        # Adjust expiration counters
        for idx in reversed(range(len(self.dialog_history))):
            if self.dialog_history[idx][0] is not None:
                updated_count = self.dialog_history[idx][0] - 1
                new_record = (updated_count, self.dialog_history[idx][1])
                self.dialog_history[idx] = new_record
                if updated_count < 0:
                    self.dialog_history.pop(idx)

        if len(self.dialog_history) >= self.max_history:
            self.dialog_history.pop(0)

        return model_response

    def reset_conversation(self):
        self.dialog_history.clear()
        self.previous_message = ""

    def provide_context(self, segment):
        """
        Override in subclass. Should provide relevant info from the object's state (e.g. last experiment results).
        """
        raise NotImplementedError

    def stage_instructions(self, segment):
        """
        Override in subclass. Should specify the main instructions for that stage.
        """
        raise NotImplementedError

    def describe_role(self):
        """
        Override in subclass. Describe the role's identity, e.g. "a professor at a top institution."
        """
        raise NotImplementedError

    def available_commands(self, segment):
        """
        Override in subclass. Return a string describing the commands the agent can use in this stage.
        """
        raise NotImplementedError

    def command_example(self, segment):
        """
        Override in subclass. Possibly give an example command usage for that stage.
        """
        raise NotImplementedError


class SeniorMentor(AbstractModule):
    """
    A role that mimics a senior faculty or mentor guiding the final drafting of a research outcome.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        super().__init__(model, notes, max_rounds, api_token)
        self.stages = ["final writeup"]

    def produce_readme(self):
        """
        Generates a README file using the combined knowledge from the summary,
        code, and notes, in Markdown format.
        """
        sys_prompt = (
            f"You are {self.describe_role()} \n"
            f"Here is the partial draft (written text):\n{self.summary}\n"
            f"Goal: Combine knowledge, code, and any relevant notes to create a readme.md for a repository."
        )
        combined_history = "\n".join([record[1] for record in self.dialog_history])
        usage_prompt = f"Dialogue History: {combined_history}\nPlease produce a README in markdown:\n"

        model_response = ask_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=usage_prompt,
            openai_api_key=self.api_token
        )
        return model_response.replace("```markdown", "")

    def provide_context(self, segment):
        # If we had a second cycle or any advanced states, we can thread them in here
        return ""

    def command_example(self, segment):
        """
        Minimal example command usage for the 'final writeup' stage
        """
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return (
            "Example:\n"
            "```DIALOGUE\nThis is some concluding text.\n```\n"
            "Or to submit final content in LaTeX:\n"
            "```LATEX\n\\section{Conclusion}\n...\n```\n"
        )

    def available_commands(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return (
            "You can produce normal dialogue with ```DIALOGUE\n...\n```, "
            "or finalize your LaTeX-based writeup with ```LATEX\n...\n```. "
            "Be thorough but also aim to finalize promptly."
        )

    def stage_instructions(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return (
            "You are finalizing a research paper or writeup in LaTeX, "
            "incorporating all previous experiment results, interpretation, and analysis. "
            "Include relevant data and do not omit important details."
        )

    def describe_role(self):
        return "a highly experienced academic mentor at a leading institution"


class ExperiencedResearcher(AbstractModule):
    """
    A role analogous to a postdoc or advanced researcher who helps shape 
    experiment plans and interpret results.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        super().__init__(model, notes, max_rounds, api_token)
        self.stages = ["experiment planning", "result interpretation"]

    def provide_context(self, segment):
        accum = ""
        if self.second_cycle:
            accum = (
                "Prior experiment data:\n"
                f"Earlier code: {self.prev_res_code}\n"
                f"Earlier results: {self.prev_results}\n"
                f"Earlier interpretation: {self.prev_insights}\n"
                f"Older draft: {self.previous_summary}\n"
                f"Assessor feedback: {self.assessor_feedback}\n\n"
            )
        if segment == "experiment planning":
            return (
                accum
                + f"Current literature synthesis: {self.lit_synthesis}\n"
            )
        elif segment == "result interpretation":
            return (
                accum
                + f"Lit. synthesis: {self.lit_synthesis}\n"
                + f"Planned blueprint: {self.blueprint}\n"
                + f"Dataset code snippet: {self.dataset_code}\n"
                + f"Result code snippet: {self.results_code}\n"
                + f"Raw results: {self.results_info}\n"
            )
        return ""

    def command_example(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        # Could show some example usage, but we'll keep it minimal here
        return ()

    def available_commands(self, segment):
        if segment == "experiment planning":
            return (
                "You can use ```DIALOGUE\n...\n``` to discuss. "
                "When you have a final plan, use ```PLAN\n...\n``` to submit it. "
                "One command per response."
            )
        elif segment == "result interpretation":
            return (
                "Use ```DIALOGUE\n...\n``` for discussion. "
                "When final, use ```INTERPRETATION\n...\n``` to submit. "
                "One command per response, keep it timely."
            )
        else:
            return ""

    def stage_instructions(self, segment):
        if segment == "experiment planning":
            return (
                "Focus on designing a new experimental approach. "
                "Integrate literature findings, plan the dataset usage, model architecture, and so forth."
            )
        elif segment == "result interpretation":
            return (
                "Examine the experimental outcomes, interpret them in the context of the plan, "
                "and provide insights for a research paper draft. Include metrics/figures of merit."
            )
        else:
            raise ValueError(f"Invalid stage: {segment}")

    def describe_role(self):
        return "an advanced researcher (similar to a postdoc) guiding experiment designs and interpretations."


class ModelSpecialist(AbstractModule):
    """
    A specialized role focusing on preparing data, running the actual code, 
    or hooking up ML frameworks as directed by the plan.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        super().__init__(model, notes, max_rounds, api_token)
        self.stages = ["data setup", "run training"]

    def provide_context(self, segment):
        accum = ""
        if self.second_cycle:
            accum = (
                "Prior cycle data:\n"
                f"Old code: {self.prev_res_code}\n"
                f"Previous results: {self.prev_results}\n"
                f"Prior interpretation: {self.prev_insights}\n"
                f"Older draft: {self.previous_summary}\n"
                f"Assessor feedback: {self.assessor_feedback}\n\n"
            )
        if segment == "data setup":
            return (
                accum
                + f"Current lit. review: {self.lit_synthesis}\n"
                + f"Plan excerpt: {self.blueprint}"
            )
        # Additional stage "run training" can be extended if needed
        return ""

    def command_example(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return ()

    def available_commands(self, segment):
        if segment == "data setup":
            return (
                "You may produce code with ```python\n# code\n``` to run in a Python environment. "
                "Use huggingface datasets. One command per turn. "
                "Or discuss with ```DIALOGUE\n...\n```."
            )
        elif segment == "run training":
            # We can define training code instructions similarly
            return "Commands for training code or discussion."
        return ""

    def stage_instructions(self, segment):
        if segment == "data setup":
            return "Focus on preparing the dataset, verifying columns, splitting data, etc."
        elif segment == "run training":
            return "Implement or run the training scripts, gather logs or outputs."
        else:
            raise ValueError(f"Invalid stage: {segment}")

    def describe_role(self):
        return "an ML engineer implementing code for data prep and model training tasks."


class SoftwareSpecialist(AbstractModule):
    """
    A role for a software engineer focusing on code structure, data handling, and 
    ensuring the final code is polished and efficient.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        super().__init__(model, notes, max_rounds, api_token)
        self.stages = ["data arrangement"]

    def provide_context(self, segment):
        accum = ""
        if self.second_cycle:
            accum = (
                "Old experiment info:\n"
                f"Previous code snippet: {self.prev_res_code}\n"
                f"Earlier results: {self.prev_results}\n"
                f"Prior interpretation: {self.prev_insights}\n"
                f"Older summary: {self.previous_summary}\n"
                f"Review feedback: {self.assessor_feedback}\n\n"
            )
        if segment == "data arrangement":
            return (
                accum
                + f"Lit. review summary: {self.lit_synthesis}\n"
                + f"Proposed blueprint: {self.blueprint}\n"
            )
        return ""

    def command_example(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return ()

    def available_commands(self, segment):
        if segment == "data arrangement":
            return (
                "Use ```DIALOGUE\n...\n``` to chat or ```SUBMIT_CODE\n...\n``` to finalize code. "
                "One command per turn, do not delay the final submission."
            )
        return ""

    def stage_instructions(self, segment):
        if segment == "data arrangement":
            return (
                "Help coordinate the code for setting up or preparing data from huggingface, "
                "ensuring it's tidy and can be used by the rest of the pipeline."
            )
        else:
            raise ValueError(f"Invalid stage: {segment}")

    def describe_role(self):
        return "a software engineer focusing on robust data pipeline construction."


class JuniorResearcher(AbstractModule):
    """
    A role for a junior researcher, akin to a PhD student, 
    who will undertake tasks like literature review, plan drafting, experiment running, etc.
    """

    def __init__(self, model="some-compact-model", notes=None, max_rounds=100, api_token=None):
        super().__init__(model, notes, max_rounds, api_token)
        self.stages = [
            "lit review",
            "plan drafting",
            "execute experiment",
            "interpret results",
            "write paper",
            "polish submission"
        ]
        self.lit_collection = []

    def provide_context(self, segment):
        accum = ""
        if self.second_cycle:
            accum = (
                f"Old experiment code: {self.prev_res_code}\n"
                f"Old results: {self.prev_results}\n"
                f"Previous insights: {self.prev_insights}\n"
                f"Old summary: {self.previous_summary}\n"
                f"Assessor remarks: {self.assessor_feedback}\n\n"
            )
        if segment == "plan drafting":
            return accum + f"Current lit. synthesis: {self.lit_synthesis}"
        elif segment == "execute experiment":
            return accum + f"Lit. synthesis: {self.lit_synthesis}\nPlan details: {self.blueprint}"
        elif segment == "interpret results":
            return (
                accum
                + f"Lit. synthesis: {self.lit_synthesis}\n"
                + f"Plan: {self.blueprint}\n"
                + f"Dataset code: {self.dataset_code}\n"
                + f"Experiment code: {self.results_code}\n"
                + f"Raw results: {self.results_info}"
            )
        elif segment == "polish submission":
            return (
                accum
                + f"Lit. synthesis: {self.lit_synthesis}\n"
                + f"Plan: {self.blueprint}\n"
                + f"Dataset code: {self.dataset_code}\n"
                + f"Experiment code: {self.results_code}\n"
                + f"Results: {self.results_info}\n"
                + f"Insights: {self.insights}"
            )
        elif segment == "lit review":
            return accum
        return ""

    def produce_requirements(self):
        """
        Create a 'requirements.txt'-style listing of packages needed for the code, 
        gleaned from the various code blocks and references so far.
        """
        sys_prompt = (
            f"You are {self.describe_role()} "
            f"\nGoal: compile a 'requirements.txt' from any known code usage and logs."
        )
        combined_history = "\n".join([entry[1] for entry in self.dialog_history])
        prompt_str = (
            f"Dialogue history: {combined_history}\n"
            f"Please produce the requirements file:\n"
        )
        model_response = ask_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt_str,
            openai_api_key=self.api_token
        )
        return model_response

    def command_example(self, segment):
        if segment not in self.stages:
            raise ValueError(f"Invalid stage: {segment}")
        return ()

    def available_commands(self, segment):
        if segment == "lit review":
            return (
                "Use ```SUMMARY\nsearch string\n``` to search papers by keyword. "
                "Use ```FULL_TEXT\narxiv_id\n``` to get the entire text of a chosen paper. "
                "Use ```ADD_PAPER\narxiv_id\npaper_summary\n``` to add it to your official references. "
                "One command per turn."
            )
        elif segment == "plan drafting":
            return (
                "Use ```DIALOGUE\n...\n``` for discussion. "
                "One command per turn."
            )
        elif segment == "execute experiment":
            return (
                "Use ```DIALOGUE\n...\n``` or ```SUBMIT_CODE\n...\n``` to finalize code. "
                "One command per turn."
            )
        elif segment == "interpret results":
            return (
                "Use ```DIALOGUE\n...\n``` for discussion. "
                "When final, just present your interpretation. One command per turn."
            )
        elif segment == "write paper":
            return (
                "Use ```DIALOGUE\n...\n``` for discussion. Possibly finalize with a latex block. "
                "One command per turn."
            )
        elif segment == "polish submission":
            return (
                "Time to finalize the paper submission details or fix any last issues. "
                "One command per turn."
            )
        return ""

    def stage_instructions(self, segment):
        if segment == "lit review":
            return (
                "You are performing a literature review. "
                "Search for relevant papers (SUMMARY), read them (FULL_TEXT), and add them (ADD_PAPER)."
            )
        elif segment == "plan drafting":
            return (
                "Propose a plan for the experiment (datasets, architecture, metrics, etc.). "
                "Discuss with an advanced researcher if needed."
            )
        elif segment == "execute experiment":
            return (
                "Write or finalize code to run the experiment, in collaboration with engineers if appropriate."
            )
        elif segment == "interpret results":
            return (
                "Review the experiment's output, reason about what happened, "
                "and produce an interpretation (numbers, charts, significance)."
            )
        elif segment == "write paper":
            return (
                "Draft a LaTeX-based paper describing the experiment, results, and analysis. "
                "You might refine it with a senior mentor."
            )
        elif segment == "polish submission":
            return "Refine the final paper for submission, possibly addressing last-minute feedback."
        else:
            raise ValueError(f"Invalid stage: {segment}")

    def describe_role(self):
        return "a junior researcher (similar to a PhD student) at a renowned institution."

    def add_literature(self, addition, retrieval_engine):
        """
        Similar to the original 'add_review' idea: parse the paper ID and summary, then store it.
        """
        try:
            arxiv_id, summary_text = addition.strip().split("\n", 1)
            full_text = retrieval_engine.retrieve_full_paper_text(arxiv_id)
            new_paper = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": summary_text
            }
            self.lit_collection.append(new_paper)
            return f"Added paper {arxiv_id} successfully.", full_text
        except Exception as e:
            return f"Error: {str(e)}", ""

    def consolidated_lit_review(self):
        """
        Return a combined text that enumerates all references collected so far.
        """
        if not self.lit_collection:
            return "No references added yet."
        output = "Collected Literature:\n"
        for entry in self.lit_collection:
            output += f"- ID: {entry['arxiv_id']}, Summary: {entry['summary']}\n"
        return output
