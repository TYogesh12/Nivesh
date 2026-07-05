import asyncio
import os
import sys
import time

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the nivesh directory is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

# Import ADK modules and the agent/runner definitions
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from app.agent import root_agent, runner, session_service

# Create a persistent session at startup
# session_id is required by runner.run_async()
APP_NAME = "app"
USER_ID = "gradio_user"
_session = session_service.create_session_sync(
    app_name=APP_NAME, user_id=USER_ID, session_id="gradio_session"
)
SESSION_ID = _session.id

# Premium Custom CSS
css = """
body {
    background-color: #0b0f19 !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding: 20px;
    background-color: #0b0f19 !important;
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
}
.header-container {
    text-align: center;
    margin-bottom: 25px;
    padding: 24px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.03em;
}
.header-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 500;
}
.thinking-box {
    background: linear-gradient(180deg, #111827 0%, #1f2937 100%) !important;
    border: 1px solid #374151 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    height: 520px;
    overflow-y: auto;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
}
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid #374151 !important;
    background-color: #111827 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
"""


def render_thinking_md(tool_logs):
    if not tool_logs:
        return "### Agent Thinking\n\n*No tools called for this turn.*"

    md = "### Agent Thinking\n\n"
    for i, log in enumerate(tool_logs, 1):
        name = log["name"]
        args = log["args"]
        duration = log["duration"]
        status = log["status"]

        if status == "running":
            status_badge = "⏳ **Running...**"
        else:
            status_badge = f"✅ **Completed** ({duration:.2f}s)"

        args_str = (
            ", ".join(f"*{k}*=`{v}`" for k, v in args.items()) if args else "None"
        )

        md += '<div style="animation: fadeIn 0.3s ease-in">\n\n'
        md += f"**{i}. `{name}`**\n"
        md += f"- **Parameters**: {args_str}\n"
        md += f"- **Status**: {status_badge}\n\n"
        md += "---\n\n"
        md += "</div>\n\n"
    return md


async def respond(message, history):
    if not message.strip():
        yield history, "### Agent Thinking\n\n*No tools called for this turn.*"
        return

    # List to trace tool logs dynamically in this turn
    tool_logs = []

    # Callback handlers defined locally to capture tool_logs in closure
    async def before_tool_callback(tool, args, tool_context):
        tool_name = tool.name
        tool_logs.append(
            {"name": tool_name, "args": args, "duration": None, "status": "running"}
        )
        tool_context.state[f"start_{tool_name}"] = time.time()
        return None

    async def after_tool_callback(tool, args, tool_context, tool_response):
        tool_name = tool.name
        start_time = tool_context.state.get(f"start_{tool_name}")
        duration = time.time() - start_time if start_time else 0.0

        for log in tool_logs:
            if log["name"] == tool_name and log["status"] == "running":
                log["duration"] = duration
                log["status"] = "completed"
                break
        return None

    # Register callbacks dynamically on root_agent
    root_agent.before_tool_callback = before_tool_callback
    root_agent.after_tool_callback = after_tool_callback

    # Prepare streaming response
    bot_message = ""
    history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    yield history, render_thinking_md(tool_logs)

    new_msg = types.Content(role="user", parts=[types.Part.from_text(text=message)])

    try:
        async for event in runner.run_async(
            new_message=new_msg,
            user_id=USER_ID,
            session_id=SESSION_ID,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            # Always accumulate text if present
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        bot_message += part.text
                        history[-1]["content"] = bot_message

            # STREAMING FIX: yield on EVERY event so Agent Thinking panel
            # updates live on tool_call_start / tool_call_end events,
            # not just when text chunks arrive.
            yield history, gr.update()
        # Incremental replay of tool calls for visual effect
        displayed_logs = []
        for log in tool_logs:
            displayed_logs.append(log)
            yield history, render_thinking_md(displayed_logs)
            await asyncio.sleep(0.25)

        # Final yield with complete state
        yield history, gr.update()

        # After ALL events: fallback if model produced no text
        if not bot_message:
            bot_message = (
                "I couldn't generate a response. Please try rephrasing your question."
            )
            history[-1]["content"] = bot_message
            yield history, render_thinking_md(tool_logs)

    except Exception as e:
        import traceback

        traceback.print_exc()
        err_str = str(e)
        if (
            "429" in err_str
            or "RESOURCE_EXHAUSTED" in err_str
            or "quota" in err_str.lower()
        ):
            bot_message = "⚠️ **Gemini API rate limit reached.** You've exhausted your free-tier daily quota (20 req/day). Please wait until midnight UTC or use a billing-enabled API key."
        else:
            bot_message = f"❌ Error: {err_str[:200]}"
        history[-1]["content"] = bot_message


# Build Gradio UI
with gr.Blocks() as demo:
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">Nivesh — Your Indian Equity Concierge</h1>
            <p class="header-subtitle">Powered by Google ADK + Gemini</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_classes=["chatbot-container"], height=520)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about Indian stocks (e.g., RELIANCE, TCS, INFY)...",
                    show_label=False,
                    scale=4,
                    container=False,
                    lines=1,
                    max_lines=3,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=1):
            thinking_panel = gr.Markdown(
                value="### Agent Thinking\n\n*No tools called yet.*",
                elem_classes=["thinking-box"],
                sanitize_html=False,
            )

    # Event handlers
    submit_event = msg.submit(
        respond, inputs=[msg, chatbot], outputs=[chatbot, thinking_panel]
    )
    submit_event.then(lambda: "", outputs=[msg])

    btn_event = submit_btn.click(
        respond, inputs=[msg, chatbot], outputs=[chatbot, thinking_panel]
    )
    btn_event.then(lambda: "", outputs=[msg])

    clear_btn.click(
        lambda: ([], "### Agent Thinking\n\n*No tools called yet.*"),
        outputs=[chatbot, thinking_panel],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=7860, css=css, theme=gr.themes.Default()
    )
