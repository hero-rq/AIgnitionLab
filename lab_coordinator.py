# lab_coordinator.py

from agents import *
from copy import copy
from imports import *
from code_engine import *
from llm_engine import *
from torch.backends.mkl import verbose

import argparse
import pickle

DEFAULT_LLM_BACKBONE = "o1-mini"


class LabCoordinator:
    def __init__(
        self,
        research_topic,
        openai_api_key,
        max_steps=100,
        num_papers_lit_review=5,
        agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}",
        notes=None,
        human_in_loop_flag=None,
        compile_pdf=True,
        mlesolver_max_steps=3,
        papersolver_max_steps=5
    ):
        """
        Orchestrates the entire research lab flow across multiple phases (lit review, planning, etc.),
        with multiple specialized roles (AssessorsGroup, JuniorResearcher, ExperiencedResearcher, etc.).
        """

        if notes is None:
            notes = []

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True
        self.review_ovrd_steps = 0
        self.arxiv_paper_exp_time = 3
        self.reference_papers = []

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS #######
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0
        self.arxiv_num_summaries = 5
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        # Mapping big phases to sub-stages:
        #   "lit review"
        #   "experiment planning"
        #   "data setup"
        #   "run training"
        #   "result interpretation"
        #   "final writeup"
        #   "submission polishing"

        self.phases = [
            ("lit review", ["lit review"]),
            ("experiment planning", ["experiment planning"]),
            ("experimentation", ["data setup", "run training"]),
            ("result interpretation", ["result interpretation", "final writeup", "submission polishing"])
        ]

        self.phase_status = {}
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = {}
        if isinstance(agent_model_backbone, str):
            # Use the same model for all sub-phases
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif isinstance(agent_model_backbone, dict):
            # Possibly different models for each sub-phase
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "lit review":           {"time": 0.0, "steps": 0.0},
            "experiment planning":  {"time": 0.0, "steps": 0.0},
            "data setup":           {"time": 0.0, "steps": 0.0},
            "run training":         {"time": 0.0, "steps": 0.0},
            "result interpretation": {"time": 0.0, "steps": 0.0},
            "final writeup":        {"time": 0.0, "steps": 0.0},
            "submission polishing": {"time": 0.0, "steps": 0.0},
        }

        self.save = True
        self.verbose = True

        # Roles (formerly "agents")
        self.assessors = AssessorsGroup(model=self.model_backbone, notes=self.notes, api_token=self.openai_api_key)
        self.junior = JuniorResearcher(model=self.model_backbone, notes=self.notes, max_rounds=self.max_steps, api_token=self.openai_api_key)
        self.expert = ExperiencedResearcher(model=self.model_backbone, notes=self.notes, max_rounds=self.max_steps, api_token=self.openai_api_key)
        self.mentor = SeniorMentor(model=self.model_backbone, notes=self.notes, max_rounds=self.max_steps, api_token=self.openai_api_key)
        self.ml_specialist = ModelSpecialist(model=self.model_backbone, notes=self.notes, max_rounds=self.max_steps, api_token=self.openai_api_key)
        self.sw_specialist = SoftwareSpecialist(model=self.model_backbone, notes=self.notes, max_rounds=self.max_steps, api_token=self.openai_api_key)

        # Remove old directories or figures if needed
        remove_figures()
        remove_directory("research_dir")

        # Create new directories for the workflow
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        os.mkdir(os.path.join(".", "research_dir"))
        os.mkdir(os.path.join("./research_dir", "src"))
        os.mkdir(os.path.join("./research_dir", "tex"))

    def set_model(self, model):
        self.set_all_agents_attr("model", model)
        self.assessors.model = model

    def set_all_agents_attr(self, attr_name, value):
        """
        Update a particular attribute (like model) in all roles/agents.
        """
        setattr(self.junior, attr_name, value)
        setattr(self.expert, attr_name, value)
        setattr(self.mentor, attr_name, value)
        setattr(self.ml_specialist, attr_name, value)
        setattr(self.sw_specialist, attr_name, value)

    def reset_all_agents(self):
        self.junior.reset_conversation()
        self.expert.reset_conversation()
        self.mentor.reset_conversation()
        self.ml_specialist.reset_conversation()
        self.sw_specialist.reset_conversation()

    def save_state(self, phase):
        """
        Save an entire snapshot of this LabCoordinator to disk.
        """
        safe_phase_name = phase.replace(" ", "_")
        with open(f"state_saves/{safe_phase_name}.pkl", "wb") as f:
            pickle.dump(self, f)

    def perform_research(self):
        """
        Execute the major phases in order, each subtask in each phase.
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()
            if self.verbose:
                print("*" * 50 + f"\nBeginning phase: {phase}\n" + "*" * 50)

            for subtask in subtasks:
                if self.verbose:
                    print("&" * 30 + f"\nBeginning subtask: {subtask}\n" + "&" * 30)

                if isinstance(self.phase_models, dict):
                    # Switch model if there's a custom assignment for the subtask
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else:
                        self.set_model(DEFAULT_LLM_BACKBONE)

                # Check if not already done
                if not self.phase_status.get(subtask, False):

                    if subtask == "lit review":
                        repeat = True
                        while repeat:
                            repeat = self.do_lit_review()
                        self.phase_status[subtask] = True

                    elif subtask == "experiment planning":
                        repeat = True
                        while repeat:
                            repeat = self.do_experiment_planning()
                        self.phase_status[subtask] = True

                    elif subtask == "data setup":
                        repeat = True
                        while repeat:
                            repeat = self.do_data_setup()
                        self.phase_status[subtask] = True

                    elif subtask == "run training":
                        repeat = True
                        while repeat:
                            repeat = self.do_run_training()
                        self.phase_status[subtask] = True

                    elif subtask == "result interpretation":
                        repeat = True
                        while repeat:
                            repeat = self.do_result_interpretation()
                        self.phase_status[subtask] = True

                    elif subtask == "final writeup":
                        repeat = True
                        while repeat:
                            repeat = self.do_final_writeup()
                        self.phase_status[subtask] = True

                    elif subtask == "submission polishing":
                        # Decide whether to finalize or circle back
                        circle_back = self.do_submission_polishing()
                        if not circle_back:
                            if self.save:
                                self.save_state(subtask)
                            return
                        else:
                            # If we decide to circle back, mark relevant steps undone and re-run
                            self.set_all_agents_attr("second_cycle", circle_back)
                            self.set_all_agents_attr("prev_report", copy(self.junior.summary))
                            self.set_all_agents_attr("prev_exp_results", copy(self.junior.results_info))
                            self.set_all_agents_attr("prev_results_code", copy(self.junior.results_code))
                            self.set_all_agents_attr("prev_insights", copy(self.junior.insights))

                            # Reset statuses so we re-run
                            self.phase_status["experiment planning"] = False
                            self.phase_status["data setup"] = False
                            self.phase_status["run training"] = False
                            self.phase_status["result interpretation"] = False
                            self.phase_status["final writeup"] = False
                            self.phase_status["submission polishing"] = False
                            self.perform_research()

                    if self.save:
                        self.save_state(subtask)

                    phase_end_time = time.time()
                    phase_duration = phase_end_time - phase_start_time
                    print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                    self.statistics_per_phase[subtask]["time"] = phase_duration

    def do_submission_polishing(self):
        """
        Formerly 'report_refinement': gather assessments, ask if user/agent is satisfied
        or wants to re-run. Returns True to circle back or False to finalize.
        """
        feedback_text = self.assessors.gather_evaluations(self.junior.blueprint, self.junior.summary)
        print("Assessors' feedback:\n", feedback_text)

        # Human or agent decides
        if self.human_in_loop_flag["submission polishing"]:
            print(f"Below are the final assessments:\n{feedback_text}")
            ans = input("Do we finalize (n), or circle back to improve (y)? (y/n): ")
            if ans.lower().strip() == "y":
                self.set_all_agents_attr(
                    "assessor_feedback",
                    f"Assessors say:\n{feedback_text}"
                )
                return True
            else:
                if self.verbose:
                    print("*" * 40, "\nREVIEW COMPLETE\n", "*" * 40)
                return False
        else:
            # If no human intervention, ask the JuniorResearcher
            query_prompt = (
                f"Assessors returned these reviews:\n{feedback_text}\n"
                "Type 'y' if we should go back and improve, or 'n' if we are done."
            )
            if self.review_override:
                # If artificially forcing some back-and-forth
                if self.review_total_steps == self.review_ovrd_steps:
                    # choose 'n'
                    response = "n"
                else:
                    # choose 'y'
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.junior.execute_stage(
                    self.research_topic, 
                    "polish submission", 
                    iteration=0,
                    external_feedback=query_prompt
                )
                if not response:
                    raise Exception("No response from the junior agent")

                response = response.lower().strip()[0]

            if response == "n":
                if self.verbose:
                    print("*" * 40, "\nREVIEW COMPLETE\n", "*" * 40)
                return False
            elif response == "y":
                self.set_all_agents_attr("assessor_feedback", f"Assessors say:\n{feedback_text}")
                return True
            else:
                raise Exception("Model gave unexpected response during submission polishing")

    def do_final_writeup(self):
        """
        Formerly 'report_writing': uses a 'paper solver' approach or simply collects a LaTeX report from the roles.
        """
        # We might have some extra logic or calls to a PaperEngine:
        from engines import PaperEngine as PaperEngine  # was: from papersolver import PaperSolver

        # Gather relevant notes for final writeup
        final_notes = [
            note_item["note"] for note_item in self.ml_specialist.observations
            if "final writeup" in note_item.get("phases", [])
        ]
        note_block = "\n".join(final_notes) if final_notes else ""

        solver = PaperEngine(
            notes=note_block,
            max_steps=self.papersolver_max_steps,
            plan=self.junior.blueprint,
            exp_code=self.junior.results_code,
            exp_results=self.junior.results_info,
            insights=self.junior.insights,
            lit_review=self.junior.lit_synthesis,
            ref_papers=self.reference_papers,
            topic=self.research_topic,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["final writeup"] if isinstance(self.model_backbone, dict) else str(self.model_backbone),
            compile_pdf=self.compile_pdf
        )

        solver.initial_solve()
        for _ in range(self.papersolver_max_steps):
            solver.solve()

        best_text, best_score = solver.best_report[0]
        final_report_str = "\n".join(best_text)
        if self.verbose:
            print(f"Final writeup completed, best score: {best_score}")

        # Possibly get human confirmation
        if self.human_in_loop_flag["final writeup"]:
            re_do = self.handle_human_decision("final writeup", final_report_str)
            if re_do:
                return re_do

        # Save final text
        self.set_all_agents_attr("summary", final_report_str)

        # Possibly produce a readme with the SeniorMentor
        readme_content = self.mentor.produce_readme()
        save_to_file("./research_dir", "readme.md", readme_content)
        save_to_file("./research_dir", "report.txt", final_report_str)
        self.reset_all_agents()
        return False

    def do_result_interpretation(self):
        """
        Formerly 'results_interpretation': toggles between the ExperiencedResearcher and the JuniorResearcher
        until an INTERPRETATION is submitted.
        """
        max_tries = self.max_steps
        conversation = ""

        for attempt in range(max_tries):
            # Let the advanced researcher speak first
            resp = self.expert.execute_stage(
                self.research_topic,
                "result interpretation",
                iteration=attempt,
                external_feedback=conversation
            )
            if self.verbose:
                print("ExperiencedResearcher says:\n", resp, "\n---")

            conversation = ""
            if "```DIALOGUE" in resp:
                snippet = extract_prompt(resp, "DIALOGUE")
                conversation = f"ExperiencedResearcher: {snippet}"

            if "```INTERPRETATION" in resp:
                interpretation_text = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["result interpretation"]:
                    re_do = self.handle_human_decision("result interpretation", interpretation_text)
                    if re_do:
                        return re_do
                self.set_all_agents_attr("insights", interpretation_text)
                self.reset_all_agents()
                self.statistics_per_phase["result interpretation"]["steps"] = attempt
                return False

            # Then the junior researcher responds
            resp = self.junior.execute_stage(
                self.research_topic,
                "interpret results",
                iteration=attempt,
                external_feedback=conversation
            )
            if self.verbose:
                print("JuniorResearcher says:\n", resp, "\n---")

            conversation = ""
            if "```DIALOGUE" in resp:
                snippet = extract_prompt(resp, "DIALOGUE")
                conversation = f"JuniorResearcher: {snippet}"

        raise Exception("Max attempts exhausted in result interpretation phase")

    def do_run_training(self):
        """
        Formerly 'running experiments': uses a MLEngine solver to produce experiment code, run, and get results.
        """
        from engines import MLEngine as MLEngine  # we already had this import, but repeated here for clarity

        # Gather notes for running experiments
        notes_for_experiments = [
            n["note"] for n in self.ml_specialist.observations
            if "run training" in n.get("phases", [])
        ]
        note_block = "\n".join(notes_for_experiments) if notes_for_experiments else ""

        solver = MLEngine(
            dataset_code=self.ml_specialist.dataset_code,
            notes=note_block,
            insights=self.ml_specialist.lit_synthesis,
            max_steps=self.mlesolver_max_steps,
            plan=self.ml_specialist.blueprint,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["run training"] if isinstance(self.model_backbone, dict) else str(self.model_backbone)
        )

        solver.initial_solve()
        for _ in range(self.mlesolver_max_steps - 1):
            solver.solve()

        top_code, top_score, top_results = solver.best_codes[0]
        final_code_str = "\n".join(top_code)
        if self.verbose:
            print(f"Running experiments done, best code score: {top_score}")

        # run the final code, gather any figure generation, etc.
        execute_code(final_code_str)
        if self.human_in_loop_flag["run training"]:
            re_do = self.handle_human_decision("run training", final_code_str)
            if re_do:
                return re_do

        save_to_file("./research_dir/src", "run_experiments.py", final_code_str)
        self.set_all_agents_attr("results_code", final_code_str)
        self.set_all_agents_attr("results_info", top_results)
        self.reset_all_agents()
        return False

    def do_data_setup(self):
        """
        Formerly 'data_preparation': interactive back-and-forth between the SoftwareSpecialist and ModelSpecialist
        to finalize a code snippet for data loading.
        """
        max_tries = self.max_steps
        feedback_from_swe = ""
        feedback_from_ml = ""
        swe_dialogue = ""
        ml_dialogue = ""
        hf_engine = DatasetArchive()  # was: HFDataSearch

        for attempt in range(max_tries):
            # SW Engineer's turn
            combined_feedback = f"{ml_dialogue}\n{feedback_from_swe}\n{feedback_from_ml}"
            resp_swe = self.sw_specialist.execute_stage(
                self.research_topic, 
                "data arrangement",
                iteration=attempt,
                external_feedback=combined_feedback
            )

            feedback_from_swe = ""
            swe_dialogue = ""
            if "```DIALOGUE" in resp_swe:
                snippet = extract_prompt(resp_swe, "DIALOGUE")
                swe_dialogue = f"SoftwareSpecialist: {snippet}"
                if self.verbose:
                    print("#" * 40 + "\nSoftwareSpecialist Dialogue:\n", snippet)

            if "```SUBMIT_CODE" in resp_swe:
                final_data_code = extract_prompt(resp_swe, "SUBMIT_CODE")
                code_exec_resp = execute_code(final_data_code, timeout=60)
                if self.verbose:
                    print("CODE RESPONSE:\n", code_exec_resp)

                if "[CODE EXECUTION ERROR]" in code_exec_resp:
                    feedback_from_swe = "\nERROR: submission code had a runtime error!\n"
                else:
                    if self.human_in_loop_flag["data setup"]:
                        re_do = self.handle_human_decision("data setup", final_data_code)
                        if re_do:
                            return re_do
                    save_to_file("./research_dir/src", "load_data.py", final_data_code)
                    self.set_all_agents_attr("dataset_code", final_data_code)
                    self.reset_all_agents()
                    self.statistics_per_phase["data setup"]["steps"] = attempt
                    return False

            # ML Engineer's turn
            combined_feedback_ml = f"{swe_dialogue}\n{feedback_from_ml}"
            resp_ml = self.ml_specialist.execute_stage(
                self.research_topic,
                "data setup",
                iteration=attempt,
                external_feedback=combined_feedback_ml
            )

            feedback_from_ml = ""
            ml_dialogue = ""
            if "```DIALOGUE" in resp_ml:
                snippet = extract_prompt(resp_ml, "DIALOGUE")
                ml_dialogue = f"ModelSpecialist: {snippet}"
                if self.verbose:
                    print("#" * 40 + "\nModelSpecialist Dialogue:\n", snippet)

            if "```python" in resp_ml:
                code_snippet = extract_prompt(resp_ml, "python")
                # Possibly combine with existing code
                code_snippet = self.ml_specialist.dataset_code + "\n" + code_snippet
                code_exec_resp = execute_code(code_snippet, timeout=120)
                feedback_from_ml = f"Code response: {code_exec_resp}"
                if self.verbose:
                    print("CODE EXECUTION OUTPUT:\n", code_exec_resp)

            if "```SEARCH_HF" in resp_ml:
                hf_query = extract_prompt(resp_ml, "SEARCH_HF")
                search_results = hf_engine.results_str(hf_engine.retrieve_ds(hf_query))
                feedback_from_ml = f"HF search results:\n{search_results}"

        raise Exception("Exceeded max tries in data setup phase")

    def do_experiment_planning(self):
        """
        Formerly 'plan_formulation': interactive back-and-forth between the ExperiencedResearcher and the JuniorResearcher
        until a plan is submitted.
        """
        max_tries = self.max_steps
        conversation = ""

        for attempt in range(max_tries):
            # Let the ExperiencedResearcher speak
            resp_exp = self.expert.execute_stage(
                self.research_topic, 
                "experiment planning",
                iteration=attempt,
                external_feedback=conversation
            )
            if self.verbose:
                print("ExperiencedResearcher:\n", resp_exp)

            conversation = ""
            if "```DIALOGUE" in resp_exp:
                snippet = extract_prompt(resp_exp, "DIALOGUE")
                conversation = f"ExperiencedResearcher: {snippet}"

            if "```PLAN" in resp_exp:
                plan_body = extract_prompt(resp_exp, "PLAN")
                if self.human_in_loop_flag["experiment planning"]:
                    re_do = self.handle_human_decision("experiment planning", plan_body)
                    if re_do:
                        return re_do
                self.set_all_agents_attr("blueprint", plan_body)
                self.reset_all_agents()
                self.statistics_per_phase["experiment planning"]["steps"] = attempt
                return False

            # Then the JuniorResearcher
            resp_jun = self.junior.execute_stage(
                self.research_topic,
                "plan drafting",
                iteration=attempt,
                external_feedback=conversation
            )
            if self.verbose:
                print("JuniorResearcher:\n", resp_jun)

            if "```DIALOGUE" in resp_jun:
                snippet = extract_prompt(resp_jun, "DIALOGUE")
                conversation = f"JuniorResearcher: {snippet}"

        raise Exception("Exceeded max tries in experiment planning phase")

    def do_lit_review(self):
        """
        Formerly 'literature_review': the JuniorResearcher queries a PaperArchive, reads details,
        and adds relevant papers. Exits after collecting enough references.
        """
        from archives import PaperArchive as PaperArchive  # was: ArxivSearch
        archive_engine = PaperArchive()

        max_tries = self.max_steps * 5

        # get initial reaction
        resp = self.junior.execute_stage(self.research_topic, "lit review", iteration=0, temperature=0.8)
        if self.verbose:
            print(resp, "\n~" * 15)

        for attempt in range(max_tries):
            feedback_block = ""

            if "```SUMMARY" in resp:
                search_str = extract_prompt(resp, "SUMMARY")
                results_list = archive_engine.find_papers_by_str(search_str, N=self.arxiv_num_summaries)
                feedback_block = (
                    f"You requested articles for: {search_str}\n"
                    f"Found papers:\n{results_list}"
                )

            elif "```FULL_TEXT" in resp:
                requested_id = extract_prompt(resp, "FULL_TEXT")
                full_text_block = archive_engine.retrieve_full_paper_text(requested_id)
                # Add expiration so it doesn't persist too long
                feedback_block = f"```EXPIRATION {self.arxiv_paper_exp_time}\n{full_text_block}```"

            elif "```ADD_PAPER" in resp:
                paper_add_data = extract_prompt(resp, "ADD_PAPER")
                feedback_block, text_content = self.junior.add_literature(paper_add_data, archive_engine)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text_content)

            # If enough references have been gathered, compile the summary and exit
            if len(self.junior.lit_collection) >= self.num_papers_lit_review:
                # produce combined text
                rev_sum = self.junior.consolidated_lit_review()
                if self.human_in_loop_flag["lit review"]:
                    re_do = self.handle_human_decision("lit review", rev_sum)
                    if re_do:
                        self.junior.lit_collection = []
                        return re_do

                self.set_all_agents_attr("lit_synthesis", rev_sum)
                self.reset_all_agents()
                self.statistics_per_phase["lit review"]["steps"] = attempt
                return False

            # Another turn
            resp = self.junior.execute_stage(
                self.research_topic,
                "lit review",
                iteration=attempt + 1,
                external_feedback=feedback_block,
                temperature=0.8
            )
            if self.verbose:
                print(resp, "\n~~~~~~~")

        raise Exception("Too many attempts in lit review stage")

    def handle_human_decision(self, phase, generated_content):
        """
        If user is in the loop, ask them if they're satisfied with the agent output (like an interactive revision loop).
        If they say 'no', append the feedback to self.notes and re-run this phase.
        Returns True if we should re-run, False otherwise.
        """
        print("\n\n---\nPHASE:", phase)
        print("Here is the current result:\n", generated_content)
        user_answer = None
        while user_answer not in ["y", "n"]:
            user_answer = input("Are you satisfied? (y/n): ").strip().lower()
            if user_answer == "y":
                return False
            elif user_answer == "n":
                feedback_text = input("Please provide some notes for improvement:\n")
                self.reset_all_agents()
                self.notes.append({"phases": [phase], "note": feedback_text})
                return True
            else:
                print("Please type 'y' or 'n' only.")
