# Deploy Workout Planner to Render

Your repo already has a `render.yaml` that defines the **workout-planner** web service. Follow these steps to deploy.

## Option A: Blueprint (uses render.yaml)

1. Go to [render.com](https://render.com) and sign in.
2. Click **New** → **Blueprint**.
3. Connect GitHub and select **npolianlis-sudo/maven_lesson_1**.
4. **Branch:** Choose **main** (where `render.yaml` lives). If Render only shows `master`, push your code to master first:
   ```bash
   git push origin main:master
   ```
   then select **master**.
5. Click **Apply**. Render will create the **workout-planner** service from `render.yaml`.
6. In the service **Environment** tab, add:
   - **OPENAI_API_KEY** = your OpenAI API key (required for the app to work)
7. Save. The first deploy will start automatically.
8. When it’s live, your app will be at `https://workout-planner-xxxx.onrender.com` (frontend at `/`, API docs at `/docs`).

## Option B: Web Service (no render.yaml)

If Blueprint can’t find `render.yaml`:

1. **New** → **Web Service**.
2. Connect **npolianlis-sudo/maven_lesson_1**, branch **main**.
3. Settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add **OPENAI_API_KEY** in Environment.
5. **Create Web Service**.

## After deploy

- Set **OPENAI_API_KEY** (or **OPENROUTER_API_KEY**) in the Render dashboard so the Workout Planner can call the LLM.
- Optional: **ENABLE_RAG**=1, **TAVILY_API_KEY**, **ARIZE_SPACE_ID**, **ARIZE_API_KEY**.
