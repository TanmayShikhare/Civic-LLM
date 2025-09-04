# gradio_app.py
import gradio as gr
from agent import interpret_then_plan, retrieve, answer

TITLE = "# Civic LLM "

def _render_chat_md(history):
    """
    history: list[(user, assistant)] -> a single markdown transcript.
    Each assistant turn already contains the Sources block (when applicable),
    so we just stitch everything together with separators.
    """
    if not history:
        return "_(no messages yet)_"
    lines = []
    for u, a in history:
        if u:
            lines.append(f"**You:** {u}")
        if a:
            lines.append(f"\n\n**Assistant:**\n\n{a}")
        lines.append("---")
    return "\n".join(lines[:-1])

def _normalize_history(obj):
    """
    Coerce whatever Gradio hands us into a clean list[(user, assistant)].
    Accepts: list of tuples/lists/strings; strings become (string, "").
    """
    hist = []
    if isinstance(obj, str):
        hist.append((obj, ""))
        return hist
    if not isinstance(obj, list):
        return []
    for t in obj:
        if isinstance(t, (list, tuple)):
            if len(t) >= 2:
                u, a = t[0], t[1]
            elif len(t) == 1:
                u, a = t[0], ""
            else:
                u, a = "", ""
        else:
            u, a = str(t), ""
        hist.append((str(u or ""), str(a or "")))
    return hist

def _format_sources_block(rows):
    """
    rows: evidence_table (list of dicts with 'url')
    Returns a markdown 'Sources' block (numbered), or empty string if none.
    """
    seen, uniq = set(), []
    for r in rows or []:
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)
    if not uniq:
        return ""
    listed = "\n".join(f"{i+1}. {u}" for i, u in enumerate(uniq[:15]))
    return f"\n\n---\n**Sources**\n\n{listed}"

def run_pipeline(user_msg: str, state_history):
    # Ensure history is always a list of (user, assistant)
    history = _normalize_history(state_history)

    # Interpret + plan
    pipe = interpret_then_plan(user_msg, history=history)
    interp, plan = pipe["interpret"], pipe["plan"]

    # If non-political, let the model produce a friendly, natural response (no sources)
    if interp.get("scope") == "non-political":
        ans = answer(interp, [])
        assistant_reply = ans  # no sources block for out-of-scope
        history.append((user_msg, assistant_reply))
        chat_md = _render_chat_md(history)
        return chat_md, history, ""  # clear textbox

    # Retrieve → Answer
    ev = retrieve(interp, plan)
    rows = ev.get("evidence_table", [])
    ans = answer(interp, rows)

    # Attach sources under this specific answer (if any)
    sources_block = _format_sources_block(rows)
    assistant_reply = ans + sources_block

    # Append the visible turn and render
    history.append((user_msg, assistant_reply))
    chat_md = _render_chat_md(history)
    return chat_md, history, ""

def clear_fn():
    # Reset transcript and state
    return "_(cleared)_", [], ""

with gr.Blocks() as demo:
    gr.Markdown(TITLE)

    chat_md = gr.Markdown("_(awaiting first message)_")

    # Keep state as a Python LIST of (user, assistant) tuples
    state = gr.State([])

    txt = gr.Textbox(
        label="Ask about political events",
        placeholder="e.g., What happened with the debt ceiling in 2023?",
        lines=2,
    )

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    send.click(
        fn=run_pipeline,
        inputs=[txt, state],
        outputs=[chat_md, state, txt],  # txt cleared by returning ""
    )

    txt.submit(
        fn=run_pipeline,
        inputs=[txt, state],
        outputs=[chat_md, state, txt],
    )

    clear.click(
        fn=clear_fn,
        inputs=None,
        outputs=[chat_md, state, txt],
    )

demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_api=False,  # no schema panel
)












