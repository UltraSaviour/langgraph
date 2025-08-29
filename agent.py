"""
LangGraph Customer Support Agent Implementation
Author: Prateek Sharma
Description: Multi-stage customer support workflow using LangGraph with MCP client integration.
"""

import json
import logging
from typing import Dict, List, Any, TypedDict
from enum import Enum
from datetime import datetime
import asyncio
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==================================
# Logging Configuration
# ==================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangieAgent")


# ==================================
# Implicit Agent Configuration
# ==================================
AGENT_CONFIG = {
    "input_schema": {
        "description": "Schema for the initial customer support ticket.",
        "properties": {
            "customer_name": {"type": "string", "example": "Priya Sharma"},
            "email": {"type": "string", "example": "priya.sharma@example.com"},
            "query": {"type": "string", "example": "My UPI payment failed but the amount was deducted."},
            "priority": {"type": "string", "example": "High"},
            "ticket_id": {"type": "string", "example": "TKT-67890"}
        },
        "required": ["customer_name", "email", "query", "priority", "ticket_id"]
    },
    "stages": [
        {
            "name": "INTAKE",
            "mode": "PAYLOAD_ONLY",
            "description": "Accepts the initial request payload.",
            "abilities": [
                {"name": "accept_payload", "mcp_server": "STATE"}
            ]
        },
        {
            "name": "UNDERSTAND",
            "mode": "DETERMINISTIC",
            "description": "Convert unstructured request to structured data and extract key entities.",
            "abilities": [
                {"name": "parse_request_text", "mcp_server": "COMMON"},
                {"name": "extract_entities", "mcp_server": "ATLAS"}
            ]
        },
        {
            "name": "PREPARE",
            "mode": "DETERMINISTIC",
            "description": "Standardize fields, enrich with external data, and compute flags.",
            "abilities": [
                {"name": "normalize_fields", "mcp_server": "COMMON"},
                {"name": "enrich_records", "mcp_server": "ATLAS"},
                {"name": "add_flags_calculations", "mcp_server": "COMMON"}
            ]
        },
        {
            "name": "ASK",
            "mode": "HUMAN",
            "description": "Requests missing information from the user if necessary.",
            "abilities": [
                {"name": "clarify_question", "mcp_server": "ATLAS"}
            ]
        },
        {
            "name": "WAIT",
            "mode": "DETERMINISTIC",
            "description": "Waits for and captures the user's response to a clarification.",
            "abilities": [
                {"name": "extract_answer", "mcp_server": "ATLAS"},
                {"name": "store_answer", "mcp_server": "STATE"}
            ]
        },
        {
            "name": "RETRIEVE",
            "mode": "DETERMINISTIC",
            "description": "Searches the knowledge base for relevant articles.",
            "abilities": [
                {"name": "knowledge_base_search", "mcp_server": "ATLAS"},
                {"name": "store_data", "mcp_server": "STATE"}
            ]
        },
        {
            "name": "DECIDE",
            "mode": "NON_DETERMINISTIC",
            "description": "Scores solutions and decides if escalation is required.",
            "abilities": [
                {"name": "solution_evaluation", "mcp_server": "COMMON"},
                {"name": "escalation_decision", "mcp_server": "ATLAS"},
                {"name": "update_payload", "mcp_server": "STATE"}
            ]
        },
        {
            "name": "UPDATE",
            "mode": "DETERMINISTIC",
            "description": "Updates the ticket status in the system (e.g., escalated or closed).",
            "abilities": [
                {"name": "update_ticket", "mcp_server": "ATLAS"},
                {"name": "close_ticket", "mcp_server": "ATLAS"}
            ]
        },
        {
            "name": "CREATE",
            "mode": "DETERMINISTIC",
            "description": "Generates the final response to send to the customer.",
            "abilities": [
                {"name": "response_generation", "mcp_server": "COMMON"}
            ]
        },
        {
            "name": "DO",
            "mode": "DETERMINISTIC",
            "description": "Executes final actions like API calls and sending notifications.",
            "abilities": [
                {"name": "execute_api_calls", "mcp_server": "ATLAS"},
                {"name": "trigger_notifications", "mcp_server": "ATLAS"}
            ]
        },
        {
            "name": "COMPLETE",
            "mode": "PAYLOAD_ONLY",
            "description": "Outputs the final structured payload.",
            "abilities": [
                {"name": "output_payload", "mcp_server": "STATE"}
            ]
        }
    ]
}


# ==================================
# Type Definitions and Enums
# ==================================
class MCPServer(Enum):
    """MCP Server types."""
    ATLAS = "atlas"
    COMMON = "common"
    STATE = "state"


class AgentState(TypedDict):
    """State that persists across all stages."""
    # Input payload
    customer_name: str
    email: str
    query: str
    priority: str
    ticket_id: str
    
    # Processing state
    current_stage: str
    execution_logs: List[Dict[str, Any]]
    
    # Enriched data
    parsed_data: Dict[str, Any]
    entities: Dict[str, Any]
    normalized_data: Dict[str, Any]
    sla_info: Dict[str, Any]
    flags: Dict[str, Any]
    
    # Retrieved information
    kb_results: List[Dict[str, Any]]
    clarification_needed: bool
    user_clarification: str
    
    # Decision outcomes
    solution_score: int
    escalation_required: bool
    
    # Response and actions
    ticket_status: str
    generated_response: str
    api_calls_executed: List[str]
    notifications_sent: List[str]
    
    # Final output
    final_payload: Dict[str, Any]


# ==================================
# MCP Client (Simulation)
# ==================================
class MCPClient:
    """A simulated client to execute abilities on ATLAS or COMMON servers."""
    
    def __init__(self, server_type: MCPServer):
        self.server_type = server_type
        logger.info(f"Initialized SIMULATED MCPClient for server: {self.server_type.value}")

    async def execute_ability(self, ability_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Logs and simulates the execution of an ability."""
        logger.info(f"Executing ability '{ability_name}' on {self.server_type.value} server with params: {params}")
        # Simulate a small network delay
        await asyncio.sleep(0.1)
        return self._simulate_ability_execution(ability_name, params)

    def _simulate_ability_execution(self, ability_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates the execution of an ability and returns a mock response."""
        responses = {
            "parse_request_text": {"intent": "failed_transaction", "confidence": 0.98},
            "extract_entities": {"entities": {"transaction_id": "T123456789"}},
            "normalize_fields": {"normalized": {"date": "2025-08-29"}},
            "enrich_records": {"sla": "gold", "history": 5},
            "add_flags_calculations": {"flags": {"high_value_customer": True, "high_priority": True}},
            "clarify_question": {"question": "Could you please provide the UPI transaction ID?"},
            "extract_answer": {"answer": "my_transaction_id_is_T123456789"},
            "knowledge_base_search": [{"id": "KB456", "title": "Handling Failed UPI Transactions", "score": 0.97}],
            "solution_evaluation": {"score": 98},
            "escalation_decision": {"escalated_to": "Tier 2 Finance Support", "reason": "Low confidence score"},
            "update_ticket": {"status": "updated", "ticket_id": params.get("ticket_id")},
            "close_ticket": {"status": "closed", "ticket_id": params.get("ticket_id")},
            "response_generation": {"response": "We have checked your transaction. The amount will be refunded to your account within 3-5 business days."},
            "execute_api_calls": {"calls": ["refund_processing_api"]},
            "trigger_notifications": {"notifications": ["sms_sent_to_customer", "email_sent_to_customer"]}
        }
        return responses.get(ability_name, {})


# ==================================
# Customer Support Agent
# ==================================
class CustomerSupportAgent:
    """Orchestrates the customer support workflow stages."""

    def __init__(self):
        self.atlas_client = MCPClient(MCPServer.ATLAS)
        self.common_client = MCPClient(MCPServer.COMMON)

    def log_stage_execution(self, state: AgentState, stage_name: str, abilities: List[str]) -> AgentState:
        """Logs the execution of a stage and the abilities called."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "abilities_executed": abilities
        }
        state["execution_logs"].append(log_entry)
        state["current_stage"] = stage_name
        return state

    async def stage_1_intake(self, state: AgentState) -> AgentState:
        state["execution_logs"] = []
        return self.log_stage_execution(state, "INTAKE", ["accept_payload"])

    async def stage_2_understand(self, state: AgentState) -> AgentState:
        parsed = await self.common_client.execute_ability("parse_request_text", {"text": state["query"]})
        entities = await self.atlas_client.execute_ability("extract_entities", {"text": state["query"], "parsed_data": parsed})
        state["parsed_data"] = parsed
        state["entities"] = entities.get("entities", {})
        return self.log_stage_execution(state, "UNDERSTAND", ["parse_request_text", "extract_entities"])

    async def stage_3_prepare(self, state: AgentState) -> AgentState:
        normalized = await self.common_client.execute_ability("normalize_fields", {"entities": state["entities"]})
        enriched = await self.atlas_client.execute_ability("enrich_records", {"customer_id": state["email"]})
        flags = await self.common_client.execute_ability("add_flags_calculations", {"normalized": normalized, "enriched": enriched})
        state["normalized_data"] = normalized
        state["sla_info"] = enriched
        state["flags"] = flags.get("flags", {})
        return self.log_stage_execution(state, "PREPARE", ["normalize_fields", "enrich_records", "add_flags_calculations"])

    async def stage_4_ask(self, state: AgentState) -> AgentState:
        state["clarification_needed"] = not state.get("entities", {}).get("transaction_id")
        return self.log_stage_execution(state, "ASK", ["clarify_question"])

    async def stage_5_wait(self, state: AgentState) -> AgentState:
        if state["clarification_needed"]:
            answer = await self.atlas_client.execute_ability("extract_answer", {"context": "User provided clarification"})
            state["user_clarification"] = answer.get("answer", "")
        return self.log_stage_execution(state, "WAIT", ["extract_answer", "store_answer"])

    async def stage_6_retrieve(self, state: AgentState) -> AgentState:
        results = await self.atlas_client.execute_ability("knowledge_base_search", {"query": state["query"], "entities": state["entities"]})
        state["kb_results"] = results
        return self.log_stage_execution(state, "RETRIEVE", ["knowledge_base_search", "store_data"])

    async def stage_7_decide(self, state: AgentState) -> AgentState:
        evaluation = await self.common_client.execute_ability("solution_evaluation", {"kb_results": state["kb_results"]})
        state["solution_score"] = evaluation.get("score", 0)
        abilities_executed = ["solution_evaluation", "update_payload"]
        if state["solution_score"] < 90:
            await self.atlas_client.execute_ability("escalation_decision", {"score": state["solution_score"]})
            state["escalation_required"] = True
            abilities_executed.append("escalation_decision")
        else:
            state["escalation_required"] = False
        return self.log_stage_execution(state, "DECIDE", abilities_executed)

    async def stage_8_update(self, state: AgentState) -> AgentState:
        abilities_executed = []
        if state["escalation_required"]:
            update_result = await self.atlas_client.execute_ability("update_ticket", {"ticket_id": state["ticket_id"], "status": "escalated"})
            state["ticket_status"] = "escalated"
            abilities_executed.append("update_ticket")
        else:
            close_result = await self.atlas_client.execute_ability("close_ticket", {"ticket_id": state["ticket_id"], "status": "resolved"})
            state["ticket_status"] = "resolved"
            abilities_executed.append("close_ticket")
        return self.log_stage_execution(state, "UPDATE", abilities_executed)

    async def stage_9_create(self, state: AgentState) -> AgentState:
        response = await self.common_client.execute_ability("response_generation", {"context": state["kb_results"]})
        state["generated_response"] = response.get("response", "")
        return self.log_stage_execution(state, "CREATE", ["response_generation"])

    async def stage_10_do(self, state: AgentState) -> AgentState:
        api_calls = await self.atlas_client.execute_ability("execute_api_calls", {"ticket_status": state["ticket_status"]})
        notifications = await self.atlas_client.execute_ability("trigger_notifications", {"customer_email": state["email"]})
        state["api_calls_executed"] = api_calls.get("calls", [])
        state["notifications_sent"] = notifications.get("notifications", [])
        return self.log_stage_execution(state, "DO", ["execute_api_calls", "trigger_notifications"])

    async def stage_11_complete(self, state: AgentState) -> AgentState:
        state["final_payload"] = {
            "ticket_id": state["ticket_id"],
            "ticket_status": state["ticket_status"],
            "customer_email": state["email"],
            "escalation_required": state["escalation_required"],
            "solution_score": state["solution_score"],
            "final_response": state["generated_response"],
            "summary": "Workflow completed."
        }
        return self.log_stage_execution(state, "COMPLETE", ["output_payload"])


# ==================================
# Graph Definition
# ==================================
def build_customer_support_graph() -> StateGraph:
    """Builds the LangGraph StateGraph for the customer support workflow."""
    agent = CustomerSupportAgent()
    workflow = StateGraph(AgentState)

    # Add nodes for each stage
    workflow.add_node("intake", agent.stage_1_intake)
    workflow.add_node("understand", agent.stage_2_understand)
    workflow.add_node("prepare", agent.stage_3_prepare)
    workflow.add_node("ask", agent.stage_4_ask)
    workflow.add_node("wait", agent.stage_5_wait)
    workflow.add_node("retrieve", agent.stage_6_retrieve)
    workflow.add_node("decide", agent.stage_7_decide)
    workflow.add_node("update", agent.stage_8_update)
    workflow.add_node("create", agent.stage_9_create)
    workflow.add_node("do", agent.stage_10_do)
    workflow.add_node("complete", agent.stage_11_complete)

    # Define the execution flow (edges)
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "understand")
    workflow.add_edge("understand", "prepare")
    workflow.add_edge("prepare", "ask")
    workflow.add_edge("ask", "wait")
    workflow.add_edge("wait", "retrieve")
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("decide", "update")
    workflow.add_edge("update", "create")
    workflow.add_edge("create", "do")
    workflow.add_edge("do", "complete")
    workflow.add_edge("complete", END)

    # Compile the graph with memory to persist state
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==================================
# Main Execution Block
# ==================================
async def run_support_with_user_input(input_payload: Dict[str, Any], graph: StateGraph) -> AgentState:
    """Runs the graph with a specific input payload."""
    config = {"configurable": {"thread_id": input_payload["ticket_id"]}}
    final_state = await graph.ainvoke(input_payload, config)
    return final_state

def get_user_input() -> Dict[str, Any]:
    """Collects ticket details from the user via the command line."""
    print("--- 🤖 Langie Customer Support Agent ---")
    customer_name = input("Enter Customer Name [Priya Sharma]: ") or "Priya Sharma"
    email = input("Enter Email [priya.sharma@example.com]: ") or "priya.sharma@example.com"
    query = input("Enter Query [My UPI payment failed but amount was deducted]: ") or "My UPI payment failed but amount was deducted"
    priority = input("Enter Priority [High]: ") or "High"
    ticket_id = input("Enter Ticket ID [TKT-67890]: ") or "TKT-67890"
    return {
        "customer_name": customer_name,
        "email": email,
        "query": query,
        "priority": priority,
        "ticket_id": ticket_id
    }

def display_results(final_state: AgentState):
    """Displays the final results of the workflow in a readable format."""
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE - FINAL RESULTS")
    print("="*60)
    
    if final_state.get("final_payload"):
        print(json.dumps(final_state["final_payload"], indent=2))
    
    if final_state.get("generated_response"):
        print(f"\n💬 RESPONSE TO CUSTOMER:")
        print("-" * 40)
        print(f"{final_state['generated_response']}")
        print("-" * 40)
    
    if final_state.get("execution_logs"):
        print(f"\n🔄 WORKFLOW EXECUTION SUMMARY:")
        for i, log in enumerate(final_state["execution_logs"], 1):
            stage = log.get('stage', 'Unknown')
            abilities = log.get('abilities_executed', [])
            print(f"  {i:2d}. {stage:12s} → {', '.join(abilities)}")

if __name__ == "__main__":
    try:
        input_payload = get_user_input()
        print(f"\n📥 Processing your request for ticket: {input_payload['ticket_id']}")
        
        graph = build_customer_support_graph()
        
        final_state = asyncio.run(run_support_with_user_input(input_payload, graph))
        
        display_results(final_state)
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)