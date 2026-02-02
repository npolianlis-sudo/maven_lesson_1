from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx


class WorkoutRequest(BaseModel):
    goal: str
    duration: str
    fitness_level: Optional[str] = None
    equipment: Optional[str] = None
    days_per_week: Optional[int] = None
    constraints: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class WorkoutResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test workout plan"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    class _NoKeyLLM:
        """Fallback LLM when no API key is configured.

        Allows the server to start (e.g., on Render) and returns a clear
        message instead of crashing at import time.
        """

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            class _Msg:
                content = (
                    "Missing LLM credentials. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
                    "in the environment and redeploy."
                )
                tool_calls: List[Dict[str, Any]] = []

            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        return _NoKeyLLM()


llm = _init_llm()


# Feature flag for optional RAG demo (opt-in for learning)
ENABLE_RAG = os.getenv("ENABLE_RAG", "0").lower() not in {"0", "false", "no"}


# RAG helper: Load curated exercises as LangChain documents
def _load_exercise_documents(path: Path) -> List[Document]:
    """Load exercises JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for row in raw:
        name = row.get("name")
        description = row.get("description", "")
        if not name:
            continue
        equipment = row.get("equipment", "")
        muscle_group = row.get("muscle_group", "")
        difficulty = row.get("difficulty", "")
        goal_tags = row.get("goal_tags", []) or []
        metadata = {
            "name": name,
            "equipment": equipment,
            "muscle_group": muscle_group,
            "difficulty": difficulty,
            "goal_tags": goal_tags,
            "source": row.get("source"),
        }
        goal_text = ", ".join(goal_tags) if goal_tags else "general fitness"
        content = f"Name: {name}\nEquipment: {equipment}\nMuscle: {muscle_group}\nDifficulty: {difficulty}\nGoals: {goal_text}\nDescription: {description}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class ExerciseRetriever:
    """Retrieves curated exercises using vector similarity search.
    Same production RAG patterns: vector search when enabled, keyword fallback,
    graceful degradation with ENABLE_RAG flag.
    """

    def __init__(self, data_path: Path):
        self._docs = _load_exercise_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        
        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, goal: str, equipment: Optional[str], difficulty: Optional[str], *, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant exercises for goal, equipment, and difficulty."""
        if not ENABLE_RAG or self.is_empty:
            return []

        if not self._vectorstore:
            return self._keyword_fallback(goal, equipment, difficulty, k=k)

        query = goal or ""
        if equipment:
            query = f"{query} {equipment}".strip()
        if difficulty:
            query = f"{query} {difficulty}".strip()
        if not query.strip():
            query = "general fitness"

        try:
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(goal, equipment, difficulty, k=k)

        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(goal, equipment, difficulty, k=k)
        return results

    def _keyword_fallback(self, goal: str, equipment: Optional[str], difficulty: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        goal_lower = (goal or "").lower()
        equip_lower = (equipment or "").lower()
        diff_lower = (difficulty or "").lower()
        goal_terms = [t.strip().lower() for t in (goal_lower or "fitness").split() if t.strip()]

        def _score(doc: Document) -> int:
            score = 0
            meta = doc.metadata
            goals = " ".join(meta.get("goal_tags") or []).lower()
            equip = (meta.get("equipment") or "").lower()
            diff = (meta.get("difficulty") or "").lower()
            if goal_terms:
                for term in goal_terms:
                    if term in goals or term in doc.page_content.lower():
                        score += 1
            if equip_lower and equip_lower in equip:
                score += 2
            if diff_lower and diff_lower in diff:
                score += 1
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
EXERCISE_RETRIEVER = ExerciseRetriever(_DATA_DIR / "exercises.json")


# Search API configuration and helpers
SEARCH_TIMEOUT = 10.0  # seconds


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise.
    
    This demonstrates graceful degradation: tools work with or without API keys.
    Students can enable real search by adding TAVILY_API_KEY or SERPAPI_API_KEY.
    """
    query = query.strip()
    if not query:
        return None

    # Try Tavily first (recommended for AI apps)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully, try next option

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    return None  # No search APIs configured


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available.
    
    This ensures tools always return useful information, even without API keys.
    """
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools with real API calls + LLM fallback (graceful degradation pattern)
@tool
def fitness_level_brief(level: str) -> str:
    """Return a brief summary of what to expect and focus on for this fitness level."""
    query = f"fitness level {level} workout expectations focus areas"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{level} level", summary)
    instruction = f"Summarize what to expect and focus on for a {level} fitness level in 200 characters or less."
    return _llm_fallback(instruction)


@tool
def constraints_brief(constraints: str) -> str:
    """Return brief guidance on how to adapt workouts given the user's constraints."""
    query = f"workout adaptations {constraints} exercise modifications"
    summary = _search_api(query)
    if summary:
        return _with_prefix("constraints", summary)
    instruction = f"Give brief guidance on adapting workouts for these constraints: {constraints}. Keep under 200 characters."
    return _llm_fallback(instruction)


@tool
def recovery_tips(goal: str, days_per_week: int) -> str:
    """Return brief recovery and rest day guidance for the goal and training frequency."""
    query = f"recovery rest days {goal} {days_per_week} days per week"
    summary = _search_api(query)
    if summary:
        return _with_prefix("recovery", summary)
    instruction = f"Give brief recovery and rest day tips for someone training {days_per_week} days per week with goal: {goal}. Under 200 characters."
    return _llm_fallback(instruction)


@tool
def equipment_options(equipment: str) -> str:
    """Return brief options and exercise ideas for the given equipment available."""
    query = f"workout exercises {equipment} home gym"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{equipment}", summary)
    instruction = f"List brief exercise options and ideas when only {equipment} is available. Under 200 characters."
    return _llm_fallback(instruction)


@tool
def session_length_guide(days_per_week: int, goal: str) -> str:
    """Return suggested session length and structure for the given frequency and goal."""
    query = f"workout session length {days_per_week} days per week {goal}"
    summary = _search_api(query)
    if summary:
        return _with_prefix("session length", summary)
    instruction = f"Suggest session length and structure for {days_per_week} days per week training with goal: {goal}. Under 200 characters."
    return _llm_fallback(instruction)


@tool
def exercise_suggestions(goal: str, equipment: str, difficulty: str) -> str:
    """Suggest specific exercises or movement patterns for the goal, equipment, and difficulty."""
    query = f"{goal} exercises {equipment} {difficulty}"
    summary = _search_api(query)
    if summary:
        return _with_prefix("exercises", summary)
    instruction = f"Suggest 2-3 specific exercises for goal={goal}, equipment={equipment}, difficulty={difficulty}. Under 200 characters."
    return _llm_fallback(instruction)


@tool
def progression_tips(goal: str) -> str:
    """Return brief progression and overload tips for the given goal."""
    query = f"workout progression {goal} progressive overload"
    summary = _search_api(query)
    if summary:
        return _with_prefix("progression", summary)
    instruction = f"Give brief progression tips for someone working toward: {goal}. Under 200 characters."
    return _llm_fallback(instruction)


class WorkoutState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    workout_request: Dict[str, Any]
    profile: Optional[str]
    equipment_schedule: Optional[str]
    exercises: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def fitness_profile_agent(state: WorkoutState) -> WorkoutState:
    req = state["workout_request"]
    goal = req.get("goal", "general fitness")
    fitness_level = req.get("fitness_level", "beginner")
    constraints = req.get("constraints", "") or ""
    days_per_week = req.get("days_per_week") or 3
    prompt_t = (
        "You are a fitness profile assistant.\n"
        "Gather level, constraints, and recovery needs for goal: {goal}, level: {fitness_level}, constraints: {constraints}, days per week: {days_per_week}.\n"
        "Use tools (fitness_level_brief, constraints_brief, recovery_tips), then summarize."
    )
    vars_ = {"goal": goal, "fitness_level": fitness_level, "constraints": constraints, "days_per_week": days_per_week}
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [fitness_level_brief, constraints_brief, recovery_tips]
    agent = llm.bind_tools(tools)
    calls: List[Dict[str, Any]] = []
    with using_attributes(tags=["fitness_profile", "info_gathering"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "fitness_profile")
                current_span.set_attribute("metadata.agent_node", "fitness_profile_agent")
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "fitness_profile", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        messages.append(res)
        messages.extend(tr["messages"])
        synthesis_prompt = "Based on the above, provide a concise fitness profile summary (level, constraints, recovery)."
        messages.append(SystemMessage(content=synthesis_prompt))
        synthesis_vars = {"goal": goal, "fitness_level": fitness_level}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    return {"messages": [SystemMessage(content=out)], "profile": out, "tool_calls": calls}


def equipment_schedule_agent(state: WorkoutState) -> WorkoutState:
    req = state["workout_request"]
    goal = req.get("goal", "general fitness")
    equipment = req.get("equipment", "bodyweight")
    days_per_week = req.get("days_per_week") or 3
    prompt_t = (
        "You are an equipment and schedule analyst.\n"
        "Analyze equipment options and session length for goal: {goal}, equipment: {equipment}, days per week: {days_per_week}.\n"
        "Use tools (equipment_options, session_length_guide), then provide a brief summary."
    )
    vars_ = {"goal": goal, "equipment": equipment, "days_per_week": days_per_week}
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [equipment_options, session_length_guide]
    agent = llm.bind_tools(tools)
    calls: List[Dict[str, Any]] = []
    with using_attributes(tags=["equipment", "schedule"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "equipment_schedule")
                current_span.set_attribute("metadata.agent_node", "equipment_schedule_agent")
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "equipment_schedule", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        messages.append(res)
        messages.extend(tr["messages"])
        synthesis_prompt = "Summarize equipment options and suggested session length/structure."
        messages.append(SystemMessage(content=synthesis_prompt))
        synthesis_vars = {"goal": goal, "equipment": equipment, "days_per_week": days_per_week}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    return {"messages": [SystemMessage(content=out)], "equipment_schedule": out, "tool_calls": calls}


def exercise_library_agent(state: WorkoutState) -> WorkoutState:
    req = state["workout_request"]
    goal = req.get("goal", "general fitness")
    equipment = req.get("equipment", "bodyweight")
    fitness_level = req.get("fitness_level", "beginner")
    context_lines = []
    if ENABLE_RAG:
        retrieved = EXERCISE_RETRIEVER.retrieve(goal, equipment, fitness_level, k=3)
        if retrieved:
            context_lines.append("=== Curated Exercises (from database) ===")
            for idx, item in enumerate(retrieved, 1):
                content = item["content"]
                source = item["metadata"].get("source", "Unknown")
                context_lines.append(f"{idx}. {content}")
                context_lines.append(f"   Source: {source}")
            context_lines.append("=== End of Curated Exercises ===\n")
    context_text = "\n".join(context_lines) if context_lines else ""
    prompt_t = (
        "You are an exercise library assistant.\n"
        "Suggest movements and progressions for goal: {goal}, equipment: {equipment}, level: {fitness_level}.\n"
        "Use tools (exercise_suggestions, progression_tips) and any curated exercises below.\n"
    )
    if context_text:
        prompt_t += "\nRelevant curated exercises from our database:\n{context}\n"
    vars_ = {
        "goal": goal,
        "equipment": equipment,
        "fitness_level": fitness_level,
        "context": context_text if context_text else "No curated context available.",
    }
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [exercise_suggestions, progression_tips]
    agent = llm.bind_tools(tools)
    calls: List[Dict[str, Any]] = []
    with using_attributes(tags=["exercise_library", "movements"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "exercise_library")
                current_span.set_attribute("metadata.agent_node", "exercise_library_agent")
                if ENABLE_RAG and context_text:
                    current_span.set_attribute("metadata.rag_enabled", "true")
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "exercise_library", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        messages.append(res)
        messages.extend(tr["messages"])
        synthesis_prompt = "Summarize suggested exercises and progression tips for the user's goal and level."
        messages.append(SystemMessage(content=synthesis_prompt))
        synthesis_vars = {"goal": goal, "equipment": equipment, "fitness_level": fitness_level}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    return {"messages": [SystemMessage(content=out)], "exercises": out, "tool_calls": calls}


def plan_synthesis_agent(state: WorkoutState) -> WorkoutState:
    req = state["workout_request"]
    goal = req.get("goal", "general fitness")
    duration = req.get("duration", "4 weeks")
    days_per_week = req.get("days_per_week") or 3
    constraints = (req.get("constraints") or "").strip()
    prompt_parts = [
        "Create a day-by-day workout plan.",
        "Goal: {goal}. Duration: {duration}. Days per week: {days_per_week}.",
        "",
        "Inputs:",
        "Fitness profile: {profile}",
        "Equipment & schedule: {equipment_schedule}",
        "Exercise suggestions: {exercises}",
    ]
    if constraints:
        prompt_parts.append("Constraints: {constraints}")
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "goal": goal,
        "duration": duration,
        "days_per_week": days_per_week,
        "constraints": constraints,
        "profile": (state.get("profile") or "")[:400],
        "equipment_schedule": (state.get("equipment_schedule") or "")[:400],
        "exercises": (state.get("exercises") or "")[:400],
    }
    with using_attributes(tags=["plan_synthesis", "final_agent"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "plan_synthesis")
                current_span.set_attribute("metadata.agent_node", "plan_synthesis_agent")
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(WorkoutState)
    g.add_node("profile_node", fitness_profile_agent)
    g.add_node("equipment_node", equipment_schedule_agent)
    g.add_node("library_node", exercise_library_agent)
    g.add_node("synthesis_node", plan_synthesis_agent)
    g.add_edge(START, "profile_node")
    g.add_edge(START, "equipment_node")
    g.add_edge(START, "library_node")
    g.add_edge("profile_node", "synthesis_node")
    g.add_edge("equipment_node", "synthesis_node")
    g.add_edge("library_node", "synthesis_node")
    g.add_edge("synthesis_node", END)
    return g.compile()


app = FastAPI(title="Workout Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "workout-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="workout-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-workout", response_model=WorkoutResponse)
def plan_workout(req: WorkoutRequest):
    graph = build_graph()
    state = {
        "messages": [],
        "workout_request": req.model_dump(),
        "tool_calls": [],
    }
    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index
    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id
    if turn_idx is not None and _TRACING:
        with using_attributes(**attrs_kwargs):
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("turn_index", turn_idx)
            out = graph.invoke(state)
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)
    return WorkoutResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
