"""
LangGraph Customer Support Agent Implementation
Author: Prateek Sharma
Description: Multi-stage customer support workflow using LangGraph with MCP client integration
"""


import json
import logging
from typing import Dict, List, Any, Optional, Literal, TypedDict
from enum import Enum
from datetime import datetime
import asyncio
from dataclasses import dataclass, field, asdict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangieAgent")


# ===========================
# Type Definitions and Enums
# ===========================


class StageMode(Enum):
    """Execution mode for each stage"""
    DETERMINISTIC = "deterministic"
    NON_DETERMINISTIC = "non-deterministic"
    HUMAN = "human"
    PAYLOAD_ONLY = "payload_only"


class MCPServer(Enum):
    """MCP Server types"""
    ATLAS = "atlas"  # External system interaction
    COMMON = "common"  # Internal processing
    STATE = "state"  # State management only


# ===========================
# State Definition
# ===========================


class AgentState(TypedDict):
    """State that persists across all stages"""
    # Input payload
    customer_name: str
    email: str
    query: str
    priority: str
    ticket_id: str
    
    # Processing state
    current_stage: str
    stage_history: List[str]
    
    # Extracted/enriched data
    entities: Dict[str, Any]
    normalized_data: Dict[str, Any]
    sla_info: Dict[str, Any]
    clarification_needed: bool
    clarification_response: Optional[str]
    
    # Retrieved information
    kb_results: List[Dict[str, Any]]
    
    # Decision outcomes
    solution_score: float
    escalation_required: bool
    selected_solution: Optional[Dict[str, Any]]
    
    # Response and actions
    generated_response: str
    api_calls_made: List[Dict[str, Any]]
    notifications_sent: List[Dict[str, Any]]
    
    # Ticket status
    ticket_status: str
    ticket_updated: bool
    
    # Final output
    final_payload: Dict[str, Any]
    
    # Execution logs
    execution_logs: List[Dict[str, Any]]


# ===========================
# MCP Client Interfaces
# ===========================


class MCPClient:
    """Base MCP Client interface"""
    
    def __init__(self, server_type: MCPServer):
        self.server_type = server_type
        self.logger = logging.getLogger(f"MCPClient.{server_type.value}")
    
    async def execute_ability(self, ability_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an ability on the MCP server"""
        self.logger.info(f"Executing {ability_name} on {self.server_type.value} server")
        # Simulate MCP server call
        return await self._simulate_ability_execution(ability_name, params)
    
    async def _simulate_ability_execution(self, ability_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ability execution for demo purposes"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Return simulated responses based on ability
        responses = {
            "parse_request_text": {
                "structured_data": {
                    "intent": "password_reset",
                    "urgency": "high",
                    "mentioned_product": "mobile_app"
                }
            },
            "extract_entities": {
                "entities": {
                    "product": "Mobile Banking App",
                    "account": "ACC123456",
                    "date_mentioned": "2024-01-15"
                }
            },
            "normalize_fields": {
                "normalized": {
                    "date": "2024-01-15T00:00:00Z",
                    "product_code": "MBA-001",
                    "priority_level": 1
                }
            },
            "enrich_records": {
                "sla_info": {
                    "response_time": "2 hours",
                    "resolution_time": "24 hours",
                    "historical_tickets": 3
                }
            },
            "knowledge_base_search": {
                "results": [
                    {
                        "title": "Password Reset Guide",
                        "content": "Steps to reset password...",
                        "relevance_score": 0.95
                    }
                ]
            },
            "solution_evaluation": {
                "score": 92.5,
                "confidence": "high",
                "recommended_action": "auto_resolve"
            },
            "response_generation": {
                "response": f"Dear {params.get('customer_name', 'Customer')}, I can help you with your request. Based on your query, here are the steps to resolve your issue..."
            },
            "clarify_question": {
                "clarification_needed": True,
                "question": "Could you please specify which product you're having trouble with?"
            },
            "extract_answer": {
                "response": params.get("clarification_response", "Mobile Banking App")
            },
            "escalation_decision": {
                "escalate": True,
                "reason": "Complex technical issue requiring specialist attention"
            },
            "update_ticket": {
                "status": "updated",
                "ticket_id": params.get("ticket_id")
            },
            "close_ticket": {
                "status": "closed",
                "ticket_id": params.get("ticket_id")
            },
            "execute_api_calls": {
                "status": "completed",
                "actions_performed": ["password_reset_initiated"]
            },
            "trigger_notifications": {
                "status": "sent",
                "notification_id": "NOTIF-001"
            }
        }
        
        return responses.get(ability_name, {"status": "completed", "data": params})


# ===========================
# Stage Implementations
# ===========================


class CustomerSupportAgent:
    """Main agent orchestrating the customer support workflow"""
    
    def __init__(self):
        self.atlas_client = MCPClient(MCPServer.ATLAS)
        self.common_client = MCPClient(MCPServer.COMMON)
        self.logger = logger
        
    def log_stage_execution(self, state: AgentState, stage: str, abilities: List[str], results: Dict[str, Any]):
        """Log stage execution details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "abilities_executed": abilities,
            "results_summary": results,
            "mcp_servers_used": []
        }
        
        if state.get("execution_logs") is None:
            state["execution_logs"] = []
        state["execution_logs"].append(log_entry)
        
        self.logger.info(f"Stage {stage} completed: {abilities}")
    
    async def stage_1_intake(self, state: AgentState) -> AgentState:
        """Stage 1: INTAKE - Accept initial payload"""
        self.logger.info("🔵 Stage 1: INTAKE - Accepting payload")
        
        state["current_stage"] = "INTAKE"
        state["stage_history"] = ["INTAKE"]
        state["ticket_status"] = "new"
        state["execution_logs"] = []
        
        self.log_stage_execution(state, "INTAKE", ["accept_payload"], 
                                {"status": "payload_accepted"})
        
        return state
    
    async def stage_2_understand(self, state: AgentState) -> AgentState:
        """Stage 2: UNDERSTAND - Parse and extract entities (Deterministic)"""
        self.logger.info("🔵 Stage 2: UNDERSTAND - Parsing request")
        
        state["current_stage"] = "UNDERSTAND"
        state["stage_history"].append("UNDERSTAND")
        
        # Execute abilities in sequence (deterministic)
        parsed = await self.common_client.execute_ability(
            "parse_request_text", 
            {"text": state["query"]}
        )
        
        entities = await self.atlas_client.execute_ability(
            "extract_entities",
            {"text": state["query"], "parsed_data": parsed}
        )
        
        state["entities"] = entities.get("entities", {})
        
        self.log_stage_execution(state, "UNDERSTAND", 
                                ["parse_request_text", "extract_entities"],
                                {"entities_found": len(state["entities"])})
        
        return state
    
    async def stage_3_prepare(self, state: AgentState) -> AgentState:
        """Stage 3: PREPARE - Normalize and enrich data (Deterministic)"""
        self.logger.info("🔵 Stage 3: PREPARE - Normalizing and enriching")
        
        state["current_stage"] = "PREPARE"
        state["stage_history"].append("PREPARE")
        
        # Execute abilities in sequence
        normalized = await self.common_client.execute_ability(
            "normalize_fields",
            {"entities": state["entities"], "priority": state["priority"]}
        )
        
        enriched = await self.atlas_client.execute_ability(
            "enrich_records",
            {"ticket_id": state["ticket_id"], "customer_email": state["email"]}
        )
        
        flags = await self.common_client.execute_ability(
            "add_flags_calculations",
            {"priority": state["priority"], "sla_info": enriched.get("sla_info", {})}
        )
        
        state["normalized_data"] = normalized.get("normalized", {})
        state["sla_info"] = enriched.get("sla_info", {})
        
        self.log_stage_execution(state, "PREPARE",
                                ["normalize_fields", "enrich_records", "add_flags_calculations"],
                                {"data_enriched": True})
        
        return state
    
    async def stage_4_ask(self, state: AgentState) -> AgentState:
        """Stage 4: ASK - Clarify with human if needed"""
        self.logger.info("🔵 Stage 4: ASK - Checking for clarifications")
        
        state["current_stage"] = "ASK"
        state["stage_history"].append("ASK")
        
        # Check if clarification is needed based on entity extraction
        if not state.get("entities") or len(state["entities"]) < 2:
            clarification = await self.atlas_client.execute_ability(
                "clarify_question",
                {"missing_info": "product_details", "context": state["query"]}
            )
            state["clarification_needed"] = True
        else:
            state["clarification_needed"] = False
        
        self.log_stage_execution(state, "ASK", ["clarify_question"],
                                {"clarification_needed": state["clarification_needed"]})
        
        return state
    
    async def stage_5_wait(self, state: AgentState) -> AgentState:
        """Stage 5: WAIT - Extract and store answer if clarification was needed"""
        self.logger.info("🔵 Stage 5: WAIT - Processing clarification response")
        
        state["current_stage"] = "WAIT"
        state["stage_history"].append("WAIT")
        
        if state.get("clarification_needed", False):
            # Simulate waiting for response
            answer = await self.atlas_client.execute_ability(
                "extract_answer",
                {"wait_time": 5, "ticket_id": state["ticket_id"]}
            )
            state["clarification_response"] = answer.get("response", "Mobile Banking App")
            
            # Update entities with clarification
            if state["clarification_response"]:
                state["entities"]["product"] = state["clarification_response"]
        
        self.log_stage_execution(state, "WAIT", ["extract_answer", "store_answer"],
                                {"response_received": bool(state.get("clarification_response"))})
        
        return state
    
    async def stage_6_retrieve(self, state: AgentState) -> AgentState:
        """Stage 6: RETRIEVE - Search knowledge base (Deterministic)"""
        self.logger.info("🔵 Stage 6: RETRIEVE - Searching knowledge base")
        
        state["current_stage"] = "RETRIEVE"
        state["stage_history"].append("RETRIEVE")
        
        kb_search = await self.atlas_client.execute_ability(
            "knowledge_base_search",
            {
                "query": state["query"],
                "entities": state["entities"],
                "product": state["entities"].get("product", "")
            }
        )
        
        state["kb_results"] = kb_search.get("results", [])
        
        self.log_stage_execution(state, "RETRIEVE", ["knowledge_base_search", "store_data"],
                                {"kb_results_found": len(state["kb_results"])})
        
        return state
    
    async def stage_7_decide(self, state: AgentState) -> AgentState:
        """Stage 7: DECIDE - Evaluate solutions and decide on escalation (Non-deterministic)"""
        self.logger.info("🔵 Stage 7: DECIDE - Evaluating solutions (Non-deterministic)")
        
        state["current_stage"] = "DECIDE"
        state["stage_history"].append("DECIDE")
        
        # Non-deterministic: Choose which abilities to execute based on context
        abilities_to_execute = []
        
        # Always evaluate solution
        evaluation = await self.common_client.execute_ability(
            "solution_evaluation",
            {
                "kb_results": state["kb_results"],
                "query": state["query"],
                "priority": state["priority"]
            }
        )
        abilities_to_execute.append("solution_evaluation")
        
        state["solution_score"] = evaluation.get("score", 0)
        
        # Conditionally decide on escalation based on score
        if state["solution_score"] < 90:
            escalation = await self.atlas_client.execute_ability(
                "escalation_decision",
                {
                    "score": state["solution_score"],
                    "priority": state["priority"],
                    "customer_email": state["email"]
                }
            )
            state["escalation_required"] = True
            abilities_to_execute.append("escalation_decision")
        else:
            state["escalation_required"] = False
            state["selected_solution"] = state["kb_results"][0] if state["kb_results"] else None
        
        self.log_stage_execution(state, "DECIDE", abilities_to_execute,
                                {
                                    "solution_score": state["solution_score"],
                                    "escalation_required": state["escalation_required"]
                                })
        
        return state
    
    async def stage_8_update(self, state: AgentState) -> AgentState:
        """Stage 8: UPDATE - Update or close ticket (Deterministic)"""
        self.logger.info("🔵 Stage 8: UPDATE - Updating ticket status")
        
        state["current_stage"] = "UPDATE"
        state["stage_history"].append("UPDATE")
        
        if state["escalation_required"]:
            # Update ticket for escalation
            update_result = await self.atlas_client.execute_ability(
                "update_ticket",
                {
                    "ticket_id": state["ticket_id"],
                    "status": "escalated",
                    "assigned_to": "tier2_support",
                    "priority": "high"
                }
            )
            state["ticket_status"] = "escalated"
        else:
            # Close ticket as resolved
            close_result = await self.atlas_client.execute_ability(
                "close_ticket",
                {
                    "ticket_id": state["ticket_id"],
                    "resolution": "auto_resolved",
                    "solution_applied": state.get("selected_solution", {})
                }
            )
            state["ticket_status"] = "closed"
        
        state["ticket_updated"] = True
        
        self.log_stage_execution(state, "UPDATE", ["update_ticket", "close_ticket"],
                                {"ticket_status": state["ticket_status"]})
        
        return state
    
    async def stage_9_create(self, state: AgentState) -> AgentState:
        """Stage 9: CREATE - Generate response (Deterministic)"""
        self.logger.info("🔵 Stage 9: CREATE - Generating customer response")
        
        state["current_stage"] = "CREATE"
        state["stage_history"].append("CREATE")
        
        response_gen = await self.common_client.execute_ability(
            "response_generation",
            {
                "customer_name": state["customer_name"],
                "query": state["query"],
                "solution": state.get("selected_solution", {}),
                "escalation": state["escalation_required"],
                "kb_results": state["kb_results"]
            }
        )
        
        state["generated_response"] = response_gen.get("response", "")
        
        self.log_stage_execution(state, "CREATE", ["response_generation"],
                                {"response_generated": bool(state["generated_response"])})
        
        return state
    
    async def stage_10_do(self, state: AgentState) -> AgentState:
        """Stage 10: DO - Execute API calls and notifications"""
        self.logger.info("🔵 Stage 10: DO - Executing actions")
        
        state["current_stage"] = "DO"
        state["stage_history"].append("DO")
        
        # Execute API calls if needed
        if not state["escalation_required"] and state.get("selected_solution"):
            api_result = await self.atlas_client.execute_ability(
                "execute_api_calls",
                {
                    "action": "apply_solution",
                    "solution": state["selected_solution"],
                    "customer_email": state["email"]
                }
            )
            state["api_calls_made"] = [api_result]
        else:
            state["api_calls_made"] = []
        
        # Send notifications
        notification_result = await self.atlas_client.execute_ability(
            "trigger_notifications",
            {
                "type": "email",
                "recipient": state["email"],
                "message": state["generated_response"],
                "ticket_id": state["ticket_id"]
            }
        )
        state["notifications_sent"] = [notification_result]
        
        self.log_stage_execution(state, "DO", ["execute_api_calls", "trigger_notifications"],
                                {
                                    "api_calls": len(state["api_calls_made"]),
                                    "notifications": len(state["notifications_sent"])
                                })
        
        return state
    
    async def stage_11_complete(self, state: AgentState) -> AgentState:
        """Stage 11: COMPLETE - Output final payload"""
        self.logger.info("🔵 Stage 11: COMPLETE - Generating final output")
        
        state["current_stage"] = "COMPLETE"
        state["stage_history"].append("COMPLETE")
        
        # Compile final payload
        state["final_payload"] = {
            "ticket_id": state["ticket_id"],
            "customer_name": state["customer_name"],
            "email": state["email"],
            "original_query": state["query"],
            "ticket_status": state["ticket_status"],
            "escalation_required": state["escalation_required"],
            "solution_score": state["solution_score"],
            "resolution": {
                "response_sent": state["generated_response"],
                "kb_articles_used": len(state["kb_results"]),
                "automated_actions": len(state["api_calls_made"]),
                "notifications_sent": len(state["notifications_sent"])
            },
            "execution_summary": {
                "stages_completed": state["stage_history"],
                "total_stages": len(state["stage_history"]),
                "processing_logs": len(state["execution_logs"])
            }
        }
        
        self.log_stage_execution(state, "COMPLETE", ["output_payload"],
                                {"final_payload_ready": True})
        
        # Print final output
        self.logger.info("=" * 60)
        self.logger.info("FINAL PAYLOAD OUTPUT:")
        self.logger.info(json.dumps(state["final_payload"], indent=2))
        self.logger.info("=" * 60)
        
        return state


# ===========================
# LangGraph Workflow Builder (FIXED)
# ===========================


def build_customer_support_graph():
    """Build the LangGraph workflow for customer support with async stages"""
    
    # Initialize the agent
    agent = CustomerSupportAgent()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add all nodes (stages) using async handlers - NO asyncio.run()
    async def intake_node(s):
        return await agent.stage_1_intake(s)
    
    async def understand_node(s):
        return await agent.stage_2_understand(s)
    
    async def prepare_node(s):
        return await agent.stage_3_prepare(s)
    
    async def ask_node(s):
        return await agent.stage_4_ask(s)
    
    async def wait_node(s):
        return await agent.stage_5_wait(s)
    
    async def retrieve_node(s):
        return await agent.stage_6_retrieve(s)
    
    async def decide_node(s):
        return await agent.stage_7_decide(s)
    
    async def update_node(s):
        return await agent.stage_8_update(s)
    
    async def create_node(s):
        return await agent.stage_9_create(s)
    
    async def do_node(s):
        return await agent.stage_10_do(s)
    
    async def complete_node(s):
        return await agent.stage_11_complete(s)
    
    workflow.add_node("intake", intake_node)
    workflow.add_node("understand", understand_node)
    workflow.add_node("prepare", prepare_node)
    workflow.add_node("ask", ask_node)
    workflow.add_node("wait", wait_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("update", update_node)
    workflow.add_node("create", create_node)
    workflow.add_node("do", do_node)
    workflow.add_node("complete", complete_node)
    
    # Define the flow (edges)
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
    
    # Compile the graph with memory for state persistence
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    
    return compiled_graph


# ===========================
# User Input Integration (FIXED)
# ===========================


async def run_support_with_user_input(input_payload: dict, graph):
    """Asynchronously execute the customer support workflow with user-provided input"""
    config = {"configurable": {"thread_id": f"support_thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    # Execute the workflow via async API since nodes are async
    result = await graph.ainvoke(input_payload, config)
    return result


def get_user_input():
    """Interactive function to collect user input for customer support ticket"""
    print("🎯 LangGraph Customer Support Agent")
    print("=" * 60)
    print("Please enter customer support ticket details:")
    print()
    
    # Collect user input interactively
    customer_name = input("👤 Customer Name: ").strip()
    while not customer_name:
        customer_name = input("❗ Customer Name is required: ").strip()
    
    email = input("📧 Email: ").strip()
    while not email or "@" not in email:
        email = input("❗ Valid email is required: ").strip()
    
    query = input("❓ Query/Issue Description: ").strip()
    while not query:
        query = input("❗ Query description is required: ").strip()
    
    priority = input("⚡ Priority (low/medium/high) [default: medium]: ").strip().lower()
    if priority not in ["low", "medium", "high"]:
        priority = "medium"
    
    ticket_id = input("🎫 Ticket ID [auto-generated if empty]: ").strip()
    if not ticket_id:
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{datetime.now().strftime('%H%M%S')}"
    
    return {
        "customer_name": customer_name,
        "email": email,
        "query": query,
        "priority": priority,
        "ticket_id": ticket_id
    }


def display_results(final_state: dict):
    """Display the results in a user-friendly format"""
    print("\n✅ Workflow completed successfully!")
    print("=" * 60)
    
    # Display key results
    if final_state.get("final_payload"):
        payload = final_state["final_payload"]
        
        print("📊 **PROCESSING RESULTS:**")
        print(f"🎫 Ticket ID: {payload.get('ticket_id')}")
        print(f"👤 Customer: {payload.get('customer_name')}")
        print(f"📧 Email: {payload.get('email')}")
        print(f"🎯 Status: {payload.get('ticket_status', 'unknown').upper()}")
        
        if payload.get('escalation_required'):
            print("⚠️  **ESCALATED** - Requires human intervention")
        else:
            print("✅ **AUTO-RESOLVED** - Solution provided")
        
        print(f"\n📈 Solution Confidence Score: {payload.get('solution_score', 0)}/100")
        
        resolution = payload.get('resolution', {})
        print(f"📚 Knowledge Base Articles Used: {resolution.get('kb_articles_used', 0)}")
        print(f"🔧 Automated Actions Performed: {resolution.get('automated_actions', 0)}")
        print(f"📬 Notifications Sent: {resolution.get('notifications_sent', 0)}")
        
    # Display customer response
    if final_state.get("generated_response"):
        print(f"\n💬 **RESPONSE TO CUSTOMER:**")
        print("-" * 40)
        print(f"{final_state['generated_response']}")
        print("-" * 40)
    
    # Display execution summary
    if final_state.get("execution_logs"):
        print(f"\n🔄 **WORKFLOW EXECUTION SUMMARY:**")
        for i, log in enumerate(final_state["execution_logs"], 1):
            stage = log.get('stage', 'Unknown')
            abilities = log.get('abilities_executed', [])
            print(f"  {i:2d}. {stage:12s} → {', '.join(abilities)}")


if __name__ == "__main__":
    import sys
    
    try:
        # Get user input
        input_payload = get_user_input()
        
        print(f"\n📥 Processing your request for ticket: {input_payload['ticket_id']}")
        print("=" * 60)
        
        # Build and execute the workflow
        graph = build_customer_support_graph()
        
        # Execute the workflow with user input using asyncio
        final_state = asyncio.run(run_support_with_user_input(input_payload, graph))
        
        # Display results
        display_results(final_state)
        
        print(f"\n🎯 Customer support workflow completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
