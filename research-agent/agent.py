import os
import json
from typing import Dict, Any, List, Callable
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schemas import TOOLS_SCHEMA, ResearchReport
from tools import web_search

load_dotenv()

class ResearchAgent:
    def __init__(self, model: str = "openai/gpt-oss-120b", max_iterations: int = 3):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.model = model
        self.max_iterations = max_iterations
        self.tool_registry: Dict[str, Callable] = {
            "web_search": web_search
        }

    def _execute_tool(self, tool_name: str, arguments_json: str) -> str:
        if tool_name not in self.tool_registry:
            return json.dumps({"error": f"Unknown tool: '{tool_name}'"})
        
        try:
            kwargs = json.loads(arguments_json)
        except json.JSONDecodeError as err:
            return json.dumps({"error": f"Invalid JSON arguments: {str(err)}"})
        
        func = self.tool_registry[tool_name]
        try:
            return func(**kwargs)
        except Exception as exc:
            return json.dumps({"error": f"Tool execution error: {str(exc)}"})

    def _synthesize_structured_report(self, query: str, research_notes: List[str]) -> ResearchReport:
        """Phase 2: Pure synthesis without tool interference."""
        print("\n[Agent] Synthesizing final structured report...")
        
        schema_definition = json.dumps(ResearchReport.model_json_schema(), indent=2)
        combined_notes = "\n\n--- Source Finding ---\n\n".join(research_notes)
        
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical research analyst. Synthesize the provided research notes into "
                    "a clean, comprehensive report. You must output ONLY a valid JSON object matching "
                    f"this JSON Schema:\n{schema_definition}\n\n"
                    "Do not call tools. Do not output markdown fences (```json). Output pure JSON only."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\n\n"
                    f"Research Findings Collected:\n{combined_notes}"
                )
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=synthesis_messages,
            response_format={"type": "json_object"}
        )

        raw_json_str = response.choices[0].message.content
        return ResearchReport.model_validate_json(raw_json_str)

    def run(self, user_query: str) -> ResearchReport:
        """Phase 1: The ReAct Loop."""
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a research agent. Use the `web_search` tool to gather factual information. "
                    "Once you have sufficient information or after 2-3 searches, answer the user."
                )
            },
            {"role": "user", "content": user_query}
        ]

        collected_observations: List[str] = []

        print(f"\n[Agent] Starting research loop for: '{user_query}'")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # If no tools requested, model thinks it has enough context
            if not msg.tool_calls:
                print("[Agent] Research complete. Model decided no more searches are needed.")
                if msg.content:
                    collected_observations.append(msg.content)
                break

            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                call_id = tool_call.id

                print(f"[Tool Request] {tool_name}({tool_args})")
                observation = self._execute_tool(tool_name, tool_args)
                print(f"[Observation] Retrieved {len(observation)} characters of search data.")

                collected_observations.append(observation)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": observation
                })

        # Phase 2: Synthesize structured Pydantic report from collected observations
        return self._synthesize_structured_report(user_query, collected_observations)


if __name__ == "__main__":
    agent = ResearchAgent()
    query = "What are the latest breakthrough updates in quantum computing this month?"
    report: ResearchReport = agent.run(query)
    
    print("\n================ FINAL PYDANTIC REPORT ================\n")
    print(f"TITLE: {report.title}\n")
    print(f"EXECUTIVE SUMMARY:\n{report.executive_summary}\n")
    print("KEY FINDINGS:")
    for finding in report.key_findings:
        print(f"  • {finding}")
    print(f"\nDETAILED ANALYSIS:\n{report.detailed_analysis}\n")
    print("SOURCES:")
    for source in report.sources:
        print(f"  - [{source.title}]({source.url})")

# [Agent] Starting research loop for: 'What are the latest breakthrough updates in quantum computing this month?'

# --- Iteration 1/3 ---
# [Tool Request] web_search({"max_results":10,"query":"quantum computing breakthrough June 2024 news"})
# [Observation] Retrieved 5340 characters of search data.

# --- Iteration 2/3 ---
# [Tool Request] web_search({"max_results":10,"query":"September 2024 quantum computing breakthrough"})
# [Observation] Retrieved 5451 characters of search data.

# --- Iteration 3/3 ---
# [Tool Request] web_search({"max_results":10,"query":"September 2024 quantum computing news"})
# [Observation] Retrieved 5105 characters of search data.

# [Agent] Synthesizing final structured report...

# ================ FINAL PYDANTIC REPORT ================

# TITLE: September 2024 Quantum Computing Breakthroughs

# EXECUTIVE SUMMARY:
# September 2024 saw several notable milestones in quantum computing, including BosonQ Psi's hybrid quantum‑classical CFD simulation with only 30 logical qubits, IonQ's four‑nines two‑qubit gate fidelity record, and continued hardware advances from Quantinuum and Infleqtion's defense partnership. These developments signal rapid progress toward practical, high‑performance quantum applications across industry and research.

# KEY FINDINGS:
#   • BosonQ Psi demonstrated a hybrid quantum‑classical solver for computational fluid dynamics using just 30 logical qubits, marking a first real‑world application at this scale.
#   • IonQ set a new world record for two‑qubit gate fidelity, exceeding 99.99% and crossing the "four‑nines" benchmark for error rates.
#   • Quantinuum's 56‑qubit trapped‑ion H2 system achieved a 100× improvement over prior industry benchmarks, underscoring the scalability of trapped‑ion architectures.
#   • Infleqtion expanded its quantum technology collaboration with the Australian Army, illustrating growing defense interest in quantum sensing and communications.

# DETAILED ANALYSIS:
# The September 2024 landscape of quantum computing is defined by both application breakthroughs and hardware performance gains. BosonQ Psi's achievement, reported in a September 2024 article on The Quantum Insider, leveraged a hybrid quantum‑classical algorithm on its BQPhy platform to simulate jet‑engine fluid dynamics with a mere 30 logical qubits, a task that would be infeasible for classical supercomputers at comparable cost. This showcases the transition from abstract algorithmic demonstrations to domain‑specific problem solving. Meanwhile, IonQ announced that its proprietary Electronic Qubit Control (EQC) technology pushed two‑qubit gate fidelity beyond 99.99%, a threshold that dramatically reduces error correction overhead and brings fault‑tolerant thresholds within reach. On the hardware front, Quantinuum's recent launch of a 56‑qubit trapped‑ion system (H2) delivered a 100‑fold speedup on benchmark Random Circuit Sampling, reinforcing trapped‑ion platforms as competitive contenders against superconducting approaches. Finally, strategic partnerships are expanding, as highlighted by Infleqtion's new work with the Australian Army, indicating that national defense agencies are actively integrating quantum sensing and secure communication capabilities. Together, these advances illustrate a multi‑pronged acceleration: application‑driven quantum algorithms, record‑setting gate performance, scalable hardware, and increasing commercial‑military collaborations.

# SOURCES:
#   - [Quantum Computing Companies in 2026 (76 Major Players)](https://thequantuminsider.com/2025/09/23/top-quantum-computing-companies)
#   - [Quantum Wednesday September 25 2024 notable and interesting news articles and papers](https://sutorgroupintelligenceandadvisory.com/2024/09/25/quantum-wednesday-september-25-2024-notable-and-interesting-news-articles-and-papers)
#   - [September Quantum News Update 📰 - Qureca](https://www.qureca.com/september-quantum-news-update)
#   - [IonQ Achieves Landmark Result, Setting New World Record in Quantum Computing Performance](https://www.ionq.com/news/ionq-achieves-landmark-result-setting-new-world-record-in-quantum-computing)
#   - [Quantinuum Launches Industry-First, Trapped-Ion 56-Qubit Quantum Computer, Breaking Key Benchmark Record](https://www.quantinuum.com/press-releases/quantinuum-launches-industry-first-trapped-ion-56-qubit-quantum-computer-that-challenges-the-worlds-best-supercomputers)