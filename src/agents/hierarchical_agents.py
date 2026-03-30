"""
Hierarchical Multi-Agent System for Clinical Document Summarization
Google Cloud Agent Garden - Production Ready
"""

from typing import List, Literal, Dict, Any, Optional
from typing_extensions import TypedDict
from pathlib import Path

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from deepeval.tracing import observe, update_current_span
from config import get_config, CLINICAL_FEATURES

config = get_config()


class State(MessagesState):
    """Enhanced state with next routing and metadata"""
    next: str
    metadata: Dict[str, Any]


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web for clinical information and evidence"""
    tavily_tool = TavilySearch(max_results=max_results)
    return tavily_tool.invoke(query)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Scrape web pages for detailed clinical information"""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join([
        f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
        for doc in docs
    ])


@tool
def extract_clinical_features(text: str) -> Dict[str, str]:
    """Extract structured clinical features from text"""
    return {
        "instruction": f"Extract the following features from the text: {', '.join(CLINICAL_FEATURES)}",
        "text": text
    }


@tool
def validate_summary(summary: str, source_text: str) -> Dict[str, Any]:
    """Validate summary for fabrications and unsupported claims"""
    return {
        "summary": summary,
        "source": source_text,
        "instruction": "Check if all claims in summary are supported by source text"
    }


@tool
def deidentify_text(text: str) -> str:
    """Remove PHI from clinical text"""
    import re
    deidentified = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', deidentified)
    deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', deidentified)
    return deidentified


def make_supervisor_node(llm: ChatOpenAI, members: List[str]) -> callable:
    """Create a supervisor node for routing between workers"""
    options = ["FINISH"] + members
    system_prompt = (
        f"You are a supervisor managing a conversation between: {members}. "
        "Given the user request, respond with the worker to act next. "
        "Each worker will perform a task and respond with results. "
        "When finished, respond with FINISH."
    )
    
    class Router(TypedDict):
        """Worker to route to next"""
        next: Literal[*options]
    
    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """LLM-based router"""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})
    
    return supervisor_node


class ResearchTeam:
    """Research team with search and web scraping agents"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.model.primary_model,
            temperature=config.model.temperature
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build research team graph"""
        search_agent = create_react_agent(self.llm, tools=[tavily_search])
        web_scraper_agent = create_react_agent(self.llm, tools=[scrape_webpages])
        
        @observe(type="agent")
        def search_node(state: State) -> Command[Literal["supervisor"]]:
            result = search_agent.invoke(state)
            update_current_span(
                input=state["messages"][-1].content,
                output=result["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="search")
                    ]
                },
                goto="supervisor"
            )
        
        @observe(type="agent")
        def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
            result = web_scraper_agent.invoke(state)
            update_current_span(
                input=state["messages"][-1].content,
                output=result["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="web_scraper")
                    ]
                },
                goto="supervisor"
            )
        
        supervisor_node = make_supervisor_node(self.llm, ["search", "web_scraper"])
        
        builder = StateGraph(State)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("search", search_node)
        builder.add_node("web_scraper", web_scraper_node)
        builder.add_edge(START, "supervisor")
        
        return builder.compile()
    
    def invoke(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Invoke research team"""
        return self.graph.invoke({"messages": messages})


class SummarizationTeam:
    """Summarization team with feature extraction and validation"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.model.primary_model,
            temperature=config.model.temperature
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build summarization team graph"""
        feature_extractor_agent = create_react_agent(
            self.llm,
            tools=[extract_clinical_features],
            prompt="Extract structured clinical features from source documents. Be precise and cite evidence."
        )
        
        validator_agent = create_react_agent(
            self.llm,
            tools=[validate_summary],
            prompt="Validate summaries for fabrications. Flag any unsupported claims."
        )
        
        deidentifier_agent = create_react_agent(
            self.llm,
            tools=[deidentify_text],
            prompt="Remove all PHI from clinical text while preserving clinical information."
        )
        
        @observe(type="agent")
        def feature_extractor_node(state: State) -> Command[Literal["supervisor"]]:
            result = feature_extractor_agent.invoke(state)
            update_current_span(
                input=state["messages"][-1].content,
                output=result["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="feature_extractor")
                    ]
                },
                goto="supervisor"
            )
        
        @observe(type="agent")
        def validator_node(state: State) -> Command[Literal["supervisor"]]:
            result = validator_agent.invoke(state)
            update_current_span(
                input=state["messages"][-1].content,
                output=result["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="validator")
                    ]
                },
                goto="supervisor"
            )
        
        @observe(type="agent")
        def deidentifier_node(state: State) -> Command[Literal["supervisor"]]:
            result = deidentifier_agent.invoke(state)
            update_current_span(
                input=state["messages"][-1].content,
                output=result["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="deidentifier")
                    ]
                },
                goto="supervisor"
            )
        
        supervisor_node = make_supervisor_node(
            self.llm,
            ["feature_extractor", "validator", "deidentifier"]
        )
        
        builder = StateGraph(State)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("feature_extractor", feature_extractor_node)
        builder.add_node("validator", validator_node)
        builder.add_node("deidentifier", deidentifier_node)
        builder.add_edge(START, "supervisor")
        
        return builder.compile()
    
    def invoke(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Invoke summarization team"""
        return self.graph.invoke({"messages": messages})


class HierarchicalAgentSystem:
    """Top-level hierarchical agent system"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.model.primary_model,
            temperature=config.model.temperature
        )
        self.research_team = ResearchTeam(self.llm)
        self.summarization_team = SummarizationTeam(self.llm)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build top-level hierarchical graph"""
        
        @observe(type="team")
        def call_research_team(state: State) -> Command[Literal["supervisor"]]:
            response = self.research_team.invoke(state["messages"])
            update_current_span(
                input=state["messages"][-1].content,
                output=response["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response["messages"][-1].content,
                            name="research_team"
                        )
                    ]
                },
                goto="supervisor"
            )
        
        @observe(type="team")
        def call_summarization_team(state: State) -> Command[Literal["supervisor"]]:
            response = self.summarization_team.invoke(state["messages"])
            update_current_span(
                input=state["messages"][-1].content,
                output=response["messages"][-1].content
            )
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response["messages"][-1].content,
                            name="summarization_team"
                        )
                    ]
                },
                goto="supervisor"
            )
        
        supervisor_node = make_supervisor_node(
            self.llm,
            ["research_team", "summarization_team"]
        )
        
        builder = StateGraph(State)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("research_team", call_research_team)
        builder.add_node("summarization_team", call_summarization_team)
        builder.add_edge(START, "supervisor")
        
        return builder.compile()
    
    @observe(type="workflow")
    def invoke(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke hierarchical agent system"""
        result = self.graph.invoke({
            "messages": [("user", query)],
            "metadata": metadata or {}
        }, {"recursion_limit": config.agent.recursion_limit})
        
        update_current_span(input=query, output=result["messages"][-1].content)
        return result
    
    def stream(self, query: str, metadata: Optional[Dict[str, Any]] = None):
        """Stream hierarchical agent system responses"""
        for chunk in self.graph.stream(
            {
                "messages": [("user", query)],
                "metadata": metadata or {}
            },
            {"recursion_limit": config.agent.recursion_limit}
        ):
            yield chunk
    
    def visualize(self, output_path: Optional[Path] = None) -> bytes:
        """Visualize agent graph"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        if output_path:
            output_path.write_bytes(png_data)
        
        return png_data


if __name__ == "__main__":
    import os
    os.environ.setdefault("OPENAI_API_KEY", "your-key-here")
    
    system = HierarchicalAgentSystem()
    
    print("Hierarchical Agent System initialized")
    print(f"Research team: search, web_scraper")
    print(f"Summarization team: feature_extractor, validator, deidentifier")
    
    viz_path = Path("agent_graph.png")
    system.visualize(viz_path)
    print(f"Graph visualization saved to {viz_path}")
    
    test_query = "Extract clinical features from breast cancer pathology report"
    print(f"\nTest query: {test_query}")
    result = system.invoke(test_query)
    print(f"Result: {result['messages'][-1].content[:200]}...")
