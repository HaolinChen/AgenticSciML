"""
Proposal generation pipeline.

Combines KB knowledge and AB analysis insights with selector reasoning
to generate improvement proposals through (Propose, Critic) x N loop.
"""

import os
import argparse
import time
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import AIMessage

from constants import (
    PROPOSAL_POOL_DIR,
    MAX_PROPOSE_CRITIC_ROUNDS,
    USER_INPUT_DIR,
    TESTING_DIR,
    AGENT_MODELS,
    get_speaker_name
)
from agents import proposer_agent, critic_agent
from retrieve_KB import retrieve_KB, load_problem_description
from retrieve_AB import retrieve_AB


# ============================================================================
# Speaker Names (Dynamic based on model assignments)
# ============================================================================

PROPOSER_SPEAKER = get_speaker_name("proposer")
CRITIC_SPEAKER = get_speaker_name("critic")


# ============================================================================
# State Definition
# ============================================================================

class ProposalState(TypedDict):
    """State for proposal generation workflow"""
    parent_id: str
    parent_code: str
    parent_analysis: str
    kb_entry: str | None
    kb_entry_name: str | None
    ab_reports: str  # Formatted parent + sibling + uncle analyses
    selector_reasoning: str  # Why this solution was selected for mutation
    testing_contract: str
    problem: str
    requirements: str
    training_set_info: str  # Training dataset information (from dataset_config.json)
    data_analysis_report: str  # Data analysis report from EDA phase (if exists)

    round_num: int
    messages: list  # Conversation history with speaker names

    proposal: str | None
    discussion_file: str


# ============================================================================
# Helper Functions
# ============================================================================

def initialize_discussion_file(filepath: str, child_id: str, parent_id: str,
                               kb_entry_name: str | None, num_relatives: int,
                               selector_reasoning: str) -> None:
    """
    Create discussion file with header.

    Args:
        filepath: Path to discussion file
        child_id: Child solution ID
        parent_id: Parent solution ID being mutated
        kb_entry_name: Name of KB entry (or None)
        num_relatives: Number of AB reports (parent + siblings + uncles)
        selector_reasoning: Why this solution was selected
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = f"""# Proposal Discussion: {child_id}

**Parent**: {parent_id}
**KB Entry**: {kb_entry_name or "None"}
**AB Reports**: {num_relatives} relative(s) (parent, siblings, uncles)
**Selection Reasoning**: {selector_reasoning}
**Models**: Proposer={AGENT_MODELS["proposer"]}, Critic={AGENT_MODELS["critic"]}
**Timestamp**: {timestamp}

---

"""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(header)


def append_to_discussion(filepath: str, round_num: int, speaker: str, content: str) -> None:
    """
    Append message to discussion file IMMEDIATELY.

    Args:
        filepath: Path to discussion file
        round_num: Current round number
        speaker: Speaker identifier (model-agnostic, e.g., proposer_gemini, critic_gpt)
        content: Message content
    """
    # Determine section header (only add for proposer, indicating new round)
    if speaker.startswith("proposer"):
        if round_num < MAX_PROPOSE_CRITIC_ROUNDS - 1:
            section_title = f"Reasoning (Round {round_num})"
        elif round_num == MAX_PROPOSE_CRITIC_ROUNDS - 1:
            section_title = "Synthesis"
        else:  # round_num == MAX_PROPOSE_CRITIC_ROUNDS
            section_title = "Finalization"

        section_header = f"## Round {round_num}: {section_title}\n\n"
    else:
        section_header = ""

    entry = f"{section_header}**[{speaker}]**:\n{content}\n\n---\n\n"

    with open(filepath, 'a') as f:
        f.write(entry)


# ============================================================================
# LangGraph Nodes
# ============================================================================

def propose_node(state: ProposalState) -> ProposalState:
    """Generate or refine proposal"""
    round_num = state["round_num"]

    print(f"[Round {round_num}] Proposer generating...")

    result = proposer_agent(
        champion_code=state["parent_code"],
        champion_analysis=state["parent_analysis"],
        testing_contract=state["testing_contract"],
        problem=state["problem"],
        requirements=state["requirements"],
        kb_entry=state["kb_entry"] or "No KB entry selected",
        ab_reports=state["ab_reports"],
        selector_reasoning=state["selector_reasoning"],
        round_num=round_num,
        conversation_history=state["messages"],
        training_set_info=state["training_set_info"],
        data_analysis_report=state["data_analysis_report"]
    )

    proposal_text = result.proposal_markdown

    # Add to conversation history
    message = AIMessage(content=proposal_text, name=PROPOSER_SPEAKER)
    state["messages"].append(message)
    state["proposal"] = proposal_text

    # IMMEDIATE logging to file
    append_to_discussion(
        state["discussion_file"],
        round_num,
        PROPOSER_SPEAKER,
        proposal_text
    )

    print(f"[Round {round_num}] Proposal logged ({len(proposal_text)} chars)")

    return state


def critique_node(state: ProposalState) -> ProposalState:
    """Critique the proposal"""
    print(f"[Round {state['round_num']}] Critic evaluating...")

    result = critic_agent(
        proposal=state["proposal"],
        champion_code=state["parent_code"],
        champion_analysis=state["parent_analysis"],
        problem=state["problem"],
        requirements=state["requirements"],
        testing_contract=state["testing_contract"],
        selector_reasoning=state["selector_reasoning"],
        round_num=state["round_num"],
        conversation_history=state["messages"],
        data_analysis_report=state["data_analysis_report"]
    )

    critique_text = result.critique_markdown

    # Add to conversation history
    message = AIMessage(content=critique_text, name=CRITIC_SPEAKER)
    state["messages"].append(message)

    # IMMEDIATE logging to file
    append_to_discussion(
        state["discussion_file"],
        state["round_num"],
        CRITIC_SPEAKER,
        critique_text
    )

    print(f"[Round {state['round_num']}] Critique logged ({len(critique_text)} chars)")

    # Increment round for next iteration
    state["round_num"] += 1

    return state


def should_critique(state: ProposalState) -> str:
    """Decide whether to critique or finalize after proposal"""
    if state["round_num"] == MAX_PROPOSE_CRITIC_ROUNDS:
        return "finalize"
    return "critique"


def should_continue(state: ProposalState) -> str:
    """Decide whether to continue loop or finalize"""
    if state["round_num"] > MAX_PROPOSE_CRITIC_ROUNDS:
        return "finalize"
    return "continue"


def finalize_node(state: ProposalState) -> ProposalState:
    """Finalize and save proposal"""
    print("Proposal generation complete")
    # Final proposal is already in state["proposal"]
    return state


# ============================================================================
# Workflow Creation
# ============================================================================

def create_proposal_workflow():
    """Create LangGraph workflow for proposal generation"""

    workflow = StateGraph(ProposalState)

    # Nodes
    workflow.add_node("propose", propose_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("finalize", finalize_node)

    # Edges
    workflow.set_entry_point("propose")

    # Conditional: critique or finalize after proposal
    workflow.add_conditional_edges(
        "propose",
        should_critique,
        {
            "critique": "critique",
            "finalize": "finalize"
        }
    )

    # Conditional: continue loop or finalize after critique
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "continue": "propose",
            "finalize": "finalize"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# Main Function
# ============================================================================

def generate_proposal(parent_id: str, child_id: str, selector_reasoning: str = None) -> tuple[str, str]:
    """
    Generate proposal for a child solution.

    Args:
        parent_id: Parent solution ID (e.g., "solution_0")
        child_id: Child solution ID (e.g., "solution_00")
        selector_reasoning: Why this parent was selected for mutation (None for best-scoring solution)

    Returns:
        Tuple of (proposal_markdown, discussion_filepath)
    """

    print(f"\n{'='*80}")
    print(f"PROPOSAL GENERATION: {child_id} (parent: {parent_id})")
    print(f"{'='*80}\n")

    # Default selector reasoning if not provided
    if selector_reasoning is None:
        selector_reasoning = "This is currently the best-scoring solution ever exists, and the current best is always selected for further improvement."

    # 1. Load parent code and analysis
    print("Loading parent solution...")
    from constants import SOLUTION_AND_OUTPUTS_DIR, AB_DIR

    parent_code_path = os.path.join(SOLUTION_AND_OUTPUTS_DIR, parent_id, "solution.py")
    with open(parent_code_path, 'r') as f:
        parent_code = f.read()

    parent_analysis_path = os.path.join(AB_DIR, f"{parent_id}_analysis.md")
    with open(parent_analysis_path, 'r') as f:
        parent_analysis = f.read()

    print(f"✓ Parent: {parent_id}")

    # 2. Retrieve KB entry (0-1)
    print("\nRetrieving knowledge base entry...")
    kb_data = retrieve_KB(parent_id=parent_id)
    kb_retrieved = kb_data['entry_name'] is not None
    print(f"✓ KB Entry: {kb_data['entry_name'] or 'None'}")

    # 3. Retrieve AB reports (parent + siblings + uncles)
    print("\nRetrieving analysis reports...")
    ab_data = retrieve_AB(parent_id)

    # Format AB reports
    ab_reports_text = ""
    if ab_data["parent_analysis"]:
        ab_reports_text += f"## Parent Analysis ({ab_data['parent_id']})\n\n{ab_data['parent_analysis']}\n\n"

    for sibling in ab_data["sibling_analyses"]:
        ab_reports_text += f"## Sibling Analysis ({sibling['sibling_id']})\n\n{sibling['analysis']}\n\n"

    for uncle in ab_data["uncle_analyses"]:
        ab_reports_text += f"## Uncle Analysis ({uncle['uncle_id']})\n\n{uncle['analysis']}\n\n"

    if not ab_reports_text:
        ab_reports_text = "No parent, sibling, or uncle analyses available (root solution)"

    num_relatives = (
        (1 if ab_data["parent_analysis"] else 0) +
        len(ab_data["sibling_analyses"]) +
        len(ab_data["uncle_analyses"])
    )

    print(f"✓ AB Reports: {num_relatives} relative(s)")

    # 4. Load testing contract
    print("\nLoading testing contract...")
    guidelines_path = f"{TESTING_DIR}/guidelines.md"
    with open(guidelines_path, 'r') as f:
        testing_contract = f.read()
    print(f"✓ Testing contract loaded ({len(testing_contract)} chars)")

    # 5. Load problem description
    print("\nLoading problem description...")
    problem_description = load_problem_description()
    print(f"✓ Problem loaded ({len(problem_description)} chars)")

    # 6. Load data analysis report if exists
    print("\nLoading data analysis report...")
    from constants import DATA_ANALYSIS_DIR
    data_analysis_report = ""
    report_path = os.path.join(DATA_ANALYSIS_DIR, "data_analysis_report.md")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data_analysis_report = f.read()
        print(f"✓ Data analysis report loaded ({len(data_analysis_report)} chars)")
    else:
        print("✓ No data analysis report found (user may not have provided training dataset)")

    # 6.5. Load training set info if exists
    print("\nLoading training dataset info...")
    from constants import DATASET_CONFIG_PATH
    import json
    training_set_info = ""
    if os.path.exists(DATASET_CONFIG_PATH):
        try:
            with open(DATASET_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            if "training_set" in config:
                training_set_info = f"""**File:** `{config['training_set']['filename']}`

**Description:** {config['training_set']['description']}

**Loading Instructions:** {config['training_set']['loading_instructions']}"""
                print(f"✓ Training dataset info loaded ({len(training_set_info)} chars)")
        except Exception as e:
            print(f"Warning: Could not load training dataset info: {e}")
    else:
        print("✓ No training dataset config found")

    # 7. Initialize discussion file
    os.makedirs(PROPOSAL_POOL_DIR, exist_ok=True)
    discussion_file = f"{PROPOSAL_POOL_DIR}/discussion_{child_id.replace('solution_', '')}.md"
    initialize_discussion_file(
        discussion_file,
        child_id,
        parent_id,
        kb_data["entry_name"],
        num_relatives,
        selector_reasoning
    )
    print(f"✓ Discussion log initialized: {discussion_file}")

    # 8. Initialize state
    initial_state = ProposalState(
        parent_id=parent_id,
        parent_code=parent_code,
        parent_analysis=parent_analysis,
        kb_entry=kb_data["kb_entry"],
        kb_entry_name=kb_data["entry_name"],
        ab_reports=ab_reports_text,
        selector_reasoning=selector_reasoning,
        testing_contract=testing_contract,
        problem=problem_description,
        requirements=problem_description,  # Included in problem_description
        training_set_info=training_set_info,
        data_analysis_report=data_analysis_report,
        round_num=1,
        messages=[],
        proposal=None,
        discussion_file=discussion_file
    )

    # 8. Run workflow
    print(f"\nStarting {MAX_PROPOSE_CRITIC_ROUNDS}-round debate...\n")
    workflow = create_proposal_workflow()
    final_state = workflow.invoke(initial_state)

    # 9. Save final proposal
    proposal_file = f"{PROPOSAL_POOL_DIR}/proposal_{child_id.replace('solution_', '')}.md"
    with open(proposal_file, 'w') as f:
        f.write(final_state["proposal"])

    # Track timing and word counts

    print(f"\n{'='*80}")
    print(f"✓ Proposal saved: {proposal_file}")
    print(f"✓ Discussion saved: {discussion_file}")
    print(f"✓ Total length: {len(final_state['proposal'])} chars")
    print(f"{'='*80}\n")

    return (final_state["proposal"], discussion_file)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Generate proposal")
    parser.add_argument("--parent_id", type=str, required=True,
                       help="Parent solution ID (e.g., solution_0)")
    parser.add_argument("--child_id", type=str, required=True,
                       help="Child solution ID (e.g., solution_00)")
    parser.add_argument("--selector_reasoning", type=str, default=None,
                       help="Selector reasoning (optional, defaults to 'best-scoring')")
    args = parser.parse_args()

    try:
        proposal, discussion = generate_proposal(
            args.parent_id, args.child_id, args.selector_reasoning
        )
        print(f"SUCCESS! Proposal generated for {args.child_id}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
