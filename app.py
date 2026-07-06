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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0D0D0D;
    --bg-surface: #161616;
    --bg-elevated: #1E1E1E;
    --accent-saffron: #FF6B00;
    --accent-green: #00C896;
    --accent-red: #FF4444;
    --text-primary: #F0F0F0;
    --text-muted: #6B6B6B;
    --border: #2A2A2A;
}

body, html {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    margin: 0;
    padding: 0;
}

gradio-app, .gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    border-top: 3px solid var(--accent-saffron);
    height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Pulsing ticker-tape inspired top border animation */
@keyframes borderPulse {
    0% { border-top-color: var(--accent-saffron); box-shadow: 0 0 0 rgba(255, 107, 0, 0); }
    50% { border-top-color: #ff9d5c; box-shadow: 0 1px 10px rgba(255, 107, 0, 0.4); }
    100% { border-top-color: var(--accent-saffron); box-shadow: 0 0 0 rgba(255, 107, 0, 0); }
}

/* Pulse on generation/pending */
body:has(.pending, .loading, .generating) .gradio-container {
    animation: borderPulse 2s infinite ease-in-out;
}

.header-row {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    margin-bottom: 24px;
    padding: 16px 16px 0 16px !important;
    width: 100%;
}

.title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important;
    color: var(--accent-saffron) !important;
    margin: 0 !important;
    line-height: 1 !important;
    font-weight: normal !important;
    display: inline-block;
}

.subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    margin-left: 12px !important;
    border-left: 1px solid var(--border);
    padding-left: 12px;
    line-height: 1.2 !important;
    display: inline-block;
}

.main-row {
    padding: 0 16px !important;
    flex: 1 !important;
    overflow: hidden !important;
    min-height: 0 !important;
}

/* Chat container and items */
.chatbot-container {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.chatbot-container .message {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: var(--text-primary) !important;
}

/* Custom thin scrollbar in --border color */
.chatbot-container *::-webkit-scrollbar,
.thinking-box::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}
.chatbot-container *::-webkit-scrollbar-thumb,
.thinking-box::-webkit-scrollbar-thumb {
    background-color: var(--border);
    border-radius: 2px;
}
.chatbot-container *::-webkit-scrollbar-track,
.thinking-box::-webkit-scrollbar-track {
    background: transparent;
}

/* Thinking Box */
.thinking-box {
    background-color: var(--bg-surface) !important;
    border: none !important;
    border-left: 2px solid var(--accent-saffron) !important;
    border-radius: 0px !important;
    padding: 20px !important;
    height: 380px;
    overflow-y: auto;
}

.thinking-box h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.4rem !important;
    color: var(--text-primary) !important;
    margin-top: 0 !important;
    font-weight: normal !important;
}

/* Styled tool cards in Agent Thinking */
.thinking-box div[style*="animation"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    animation: fadeIn 0.3s ease-in !important;
}

.thinking-box div[style*="animation"] hr {
    display: none !important;
}

.thinking-box div[style*="animation"] code {
    font-family: monospace !important;
    font-size: 0.85rem !important;
    color: var(--accent-saffron) !important;
    background-color: transparent !important;
    padding: 0 !important;
}

/* Input area */
.input-row {
    margin-top: -8px !important;
    gap: 8px !important;
    flex-shrink: 0 !important;
    padding: 8px 16px !important;
}

.chat-input {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 0px !important;
}

.chat-input textarea {
    background-color: var(--bg-elevated) !important;
    border: none !important;
    color: var(--text-primary) !important;
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;
    resize: none !important;
    padding: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

.chat-input textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Buttons */
.send-btn {
    background-color: var(--accent-saffron) !important;
    color: #0D0D0D !important;
    font-weight: bold !important;
    border-radius: 0px !important;
    border: none !important;
    height: 44px !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.send-btn:hover {
    background-color: #e56000 !important;
}

.clear-btn {
    background-color: transparent !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0px !important;
    height: 44px !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer;
    transition: border-color 0.2s ease;
}

.clear-btn:hover {
    border-color: var(--text-muted) !important;
}

/* Remove Gradio Footer */
footer {
    display: none !important;
}

.disclaimer-row {
    margin-top: 4px !important;
    text-align: center !important;
    justify-content: center !important;
    width: 100% !important;
}

.disclaimer-text {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
    text-align: center !important;
    width: 100% !important;
}
"""


# PURPOSE: Formats the current tool call history logs into standard Markdown for display in the thinking panel.
# DESIGN: Uses custom wrapper HTML tags with fade-in styling variables to hook into CSS animations rather than rendering plain strings.
# TRADEOFF: Relies on sanitize_html=False on the client markdown container to properly support styles.
# BEHAVIOR: Returns a formatted Markdown string containing status tags and timings.
def render_thinking_md(tool_logs):
    if not tool_logs:
        return "### Agent Thinking\n\nWaiting for your question."

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


# PURPOSE: Handles incoming Gradio chat requests, streams the assistant text events, and replays tool executions.
# DESIGN: Implemented as an asynchronous generator yielding chatbot updates and thinking logs step-by-step to power live Gradio streaming.
# TRADEOFF: Spacing out logs post-loop introduces a minor 250ms animation delay per tool card.
# BEHAVIOR: Streams history and UI updates back to the browser; triggers callbacks during execution.
async def respond(message, history):
    if not message.strip():
        yield history, "### Agent Thinking\n\nWaiting for your question."
        return

    # List to trace tool logs dynamically in this turn
    tool_logs = []

    # Callback handlers defined locally to capture tool_logs in closure
    # PURPOSE: Intercepts the agent's tool execution start event to initialize a progress log and timestamp.
    # DESIGN: Registered dynamically inside the request scope as a closure to capture the local tool_logs list. If defined globally, concurrent requests would overwrite each other's callbacks.
    # TRADEOFF: Re-registered on every request, adding minor runtime overhead.
    # BEHAVIOR: Modifies the request-scoped tool_logs mutable list and returns None.
    async def before_tool_callback(tool, args, tool_context):
        tool_name = tool.name
        tool_logs.append(
            {"name": tool_name, "args": args, "duration": None, "status": "running"}
        )
        tool_context.state[f"start_{tool_name}"] = time.time()
        return None

    # PURPOSE: Intercepts the agent's tool execution completion event to update the progress log and track duration.
    # DESIGN: Registered dynamically inside the request scope as a closure to capture the local tool_logs list. If defined globally, concurrent requests would overwrite each other's callbacks.
    # TRADEOFF: Re-registered on every request, adding minor runtime overhead.
    # BEHAVIOR: Computes duration, updates the status badge in tool_logs, and returns None.
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

    yield history, gr.update()

    new_msg = types.Content(role="user", parts=[types.Part.from_text(text=message)])

    try:
        async for event in runner.run_async(
            new_message=new_msg,
            user_id=USER_ID,
            session_id=SESSION_ID,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            # Always accumulate text if present
            if (
                event.content
                and event.content.parts
                and getattr(event, "partial", True)
            ):
                for part in event.content.parts:
                    if part.text:
                        bot_message += part.text
                        updated_history = history[:-1] + [
                            {"role": "assistant", "content": bot_message}
                        ]
                        yield updated_history, gr.update()
        history = history[:-1] + [{"role": "assistant", "content": bot_message}]
        # Incremental replay of tool calls for visual effect
        displayed_logs = []
        for log in tool_logs:
            displayed_logs.append(log)
            yield history, render_thinking_md(displayed_logs)
            await asyncio.sleep(0.25)

        # After ALL events: fallback if model produced no text
        if not bot_message:
            bot_message = (
                "I couldn't generate a response. Please try rephrasing your question."
            )
            history = history[:-1] + [{"role": "assistant", "content": bot_message}]
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
        history = history[:-1] + [{"role": "assistant", "content": bot_message}]


# Build Gradio UI
with gr.Blocks(css=css, fill_height=True) as demo:
    with gr.Row(elem_classes=["header-row"]):
        gr.HTML(
            '<h1 class="title">● Nivesh</h1><p class="subtitle">Your Indian Equity Concierge</p>'
        )
    with gr.Row(equal_height=True, elem_classes=["main-row"]):
        with gr.Column(scale=65, min_width=500):
            chatbot = gr.Chatbot(
                height=380, elem_classes=["chatbot-container"], show_label=False
            )
        with gr.Column(scale=35, min_width=300):
            thinking_panel = gr.Markdown(
                value="### Agent Thinking\n\nWaiting for your question.",
                elem_classes=["thinking-box"],
                sanitize_html=False,
            )
    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Ask about your stocks or watchlist...",
            lines=1,
            max_lines=3,
            scale=8,
            elem_classes=["chat-input"],
            show_label=False,
        )
        send_btn = gr.Button("Ask →", scale=1, elem_classes=["send-btn"])
        clear_btn = gr.Button("Clear", scale=1, elem_classes=["clear-btn"])

    with gr.Row(elem_classes=["disclaimer-row"]):
        gr.HTML(
            '<p class="disclaimer-text">⚠️ Nivesh is a demonstration project and NOT a registered financial advisor. This agent does not provide financial or investment advice.</p>'
        )

    # Event handlers
    temp_msg = gr.State()

    def store_and_clear(message):
        return message, ""

    submit_event = msg.submit(
        store_and_clear, inputs=[msg], outputs=[temp_msg, msg]
    ).then(respond, inputs=[temp_msg, chatbot], outputs=[chatbot, thinking_panel])

    btn_event = send_btn.click(
        store_and_clear, inputs=[msg], outputs=[temp_msg, msg]
    ).then(respond, inputs=[temp_msg, chatbot], outputs=[chatbot, thinking_panel])

    clear_btn.click(
        lambda: ([], "### Agent Thinking\n\nWaiting for your question."),
        outputs=[chatbot, thinking_panel],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=7860, css=css, theme=gr.themes.Default()
    )
