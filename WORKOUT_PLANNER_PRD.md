# Workout Planner — Product Requirements Document (One-Page)

**Version:** 1.0  
**Author:** Senior Product Manager  
**Reference architecture:** This repo (AI Trip Planner) — FastAPI, LangGraph, optional RAG, Arize observability.

---

## 1. Overview & Goals

**Product:** An AI-powered **Workout Planner** that generates personalized weekly or multi-week workout plans from a single user request (goals, duration, fitness level, equipment, preferences).

**Goals:** Reuse the existing multi-agent, LangGraph-based architecture; replace “trip” domain with “workout” while keeping the same patterns: one FastAPI endpoint, parallel specialist agents, optional RAG over curated data, optional real-time APIs with LLM fallback, and Arize tracing.

---

## 2. User Stories

- As a **user**, I can describe my goal (e.g., “build strength,” “run a 5K,” “lose weight”), duration (e.g., “4 weeks”), fitness level, and equipment so that I get a day-by-day workout plan.
- As a **user**, I can specify constraints (e.g., “home only,” “3 days/week,” “no running”) so that the plan respects my situation.
- As a **developer**, I can call a single API (`POST /plan-workout`) and receive a structured plan plus optional tool-call metadata for debugging and evaluation.

---

## 3. Architecture (Aligned to This Repo)

| Trip Planner (current) | Workout Planner (target) |
|------------------------|--------------------------|
| `TripRequest` / `TripResponse` | `WorkoutRequest` / `WorkoutResponse` |
| `TripState` (research, budget, local, final, tool_calls) | `WorkoutState` (profile, exercises, recovery, final, tool_calls) |
| Research Agent | **Fitness Profile Agent** — level, constraints, injury/availability |
| Budget Agent | **Equipment & Schedule Agent** — equipment available, days per week, session length |
| Local Agent (+ RAG over `local_guides.json`) | **Exercise Library Agent** (+ RAG over `exercises.json`) — movements, progressions, cues |
| Itinerary Agent | **Plan Synthesis Agent** — day-by-day plan combining profile, equipment, and exercises |
| `POST /plan-trip` | `POST /plan-workout` |
| `backend/data/local_guides.json` | `backend/data/exercises.json` (curated movements with tags: goal, equipment, muscle group, difficulty) |

**Execution flow:** Same as repo — START → **parallel**: Fitness Profile, Equipment & Schedule, Exercise Library (with optional RAG) → **converge** → Plan Synthesis → END. All agents and tools traced via Arize (optional).

**Stack (unchanged):** FastAPI, LangGraph, LangChain, OpenAI/OpenRouter, optional RAG (vector search over `exercises.json`), optional web search (e.g., Tavily) for latest exercise/nutrition info with LLM fallback, OpenInference/Arize.

---

## 4. Request / Response

- **Request:** `WorkoutRequest`: `goal` (str), `duration` (e.g. `"4 weeks"`), `fitness_level` (e.g. beginner/intermediate/advanced), `equipment` (optional, e.g. `"none"` | `"dumbbells"` | `"full gym"`), `days_per_week` (optional), `constraints` (optional, e.g. “no running,” “knee-friendly”), plus optional `session_id`, `user_id` for observability.
- **Response:** `WorkoutResponse`: `result` (markdown plan), `tool_calls` (list of tool invocations for debugging/eval).

---

## 5. Data & RAG

- **Curated dataset:** `backend/data/exercises.json` — entries with fields such as name, description, equipment, muscle_group, difficulty, goal_tags (e.g. strength, cardio, mobility), and source. RAG (when `ENABLE_RAG=1`) retrieves relevant exercises for the Plan Synthesis Agent; otherwise LLM-only fallback, mirroring `local_guides.json` and `LocalGuideRetriever` pattern.

---

## 6. Non-Functional & Success

- **Performance:** Same as trip planner — parallel agents to keep latency low; aim for sub-10s p95 when LLM and optional APIs are healthy.
- **Observability:** Optional Arize project (e.g. `workout-planner`); trace all agent nodes and tool calls.
- **Deployment:** Same as repo — e.g. `render.yaml` style: install deps, run `uvicorn main:app`; env: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, optional `ENABLE_RAG`, `TAVILY_API_KEY`, Arize keys.

**Success:** User receives a coherent, personalized workout plan for the given goal, duration, level, and constraints; API and traces are available for iteration and evaluation.

---

## 7. Out of Scope (V1)

- User accounts, persistence, or history.
- Form-based UI (can add later; minimal UI at `/` analogous to trip planner).
- Real-time form corrections or multi-turn refinement (single request/response only for V1).
