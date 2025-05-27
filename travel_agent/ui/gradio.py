import re
from functools import partial
from typing import Generator

import gradio as gr
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def _clean_model_output(model_output: str) -> str:
    """
    Clean up model output by removing trailing tags and extra backticks.

    Args:
        model_output (`str`): Raw model output.

    Returns:
        `str`: Cleaned model output.
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
    return model_output.strip()


def _format_code_content(content: str) -> str:
    """
    Format code content as Python code block if it's not already formatted.

    Args:
        content (`str`): Code content to format.

    Returns:
        `str`: Code content formatted as a Python code block.
    """
    content = content.strip()
    # Remove existing code blocks and end_code tags
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # Add Python code block formatting if not already present
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process an [`ActionStep`] and yield appropriate Gradio ChatMessage objects.

    Args:
        step_log ([`ActionStep`]): ActionStep to process.
        skip_model_outputs (`bool`): Whether to skip model outputs.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the action step.
    """
    # Output the step number
    step_number = f"Шаг {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**", metadata={"status": "done"})

    # First yield the thought/reasoning from the LLM
    if not skip_model_outputs and getattr(step_log, "model_output", "") and not getattr(step_log, "tool_calls", []):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(role="assistant", content=model_output, metadata={"status": "done"})

    # For tool calls, create a parent message
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # Process arguments based on type
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # Format code content if needed
        if used_code:
            content = _format_code_content(content)

        # Create the tool call message
        parent_message_tool = gr.ChatMessage(
            role="assistant",
            content=content,
            metadata={
                "title": f"🛠️ Использована утилита {first_tool_call.name}",
                "status": "done",
            },
        )
        yield parent_message_tool

    # Display execution logs if they exist
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = step_log.observations.strip()
        if log_content:
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            yield gr.ChatMessage(
                role="assistant",
                content=f"```bash\n{log_content}\n",
                metadata={"title": "📝 Лог выполнения", "status": "done"},
            )

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role="assistant",
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                metadata={"title": "🖼️ Изображение", "status": "done"},
            )

    # Handle errors
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role="assistant", content=str(step_log.error), metadata={"title": "💥 Ошибка", "status": "done"}
        )

    # Add step footnote and separator
    yield gr.ChatMessage(
        role="assistant", content=get_step_footnote_content(step_log, step_number), metadata={"status": "done"}
    )
    yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process a [`PlanningStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`PlanningStep`]): PlanningStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the planning step.
    """
    if not skip_model_outputs:
        yield gr.ChatMessage(
            role="assistant",
            content=f"{_clean_model_output(step_log.plan)}",
            metadata={"title": "🧠 **Планирование действий**", "status": "done"},
        )
    yield gr.ChatMessage(
        role="assistant", content=get_step_footnote_content(step_log, "Шаг планирования"), metadata={"status": "done"}
    )
    yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    """
    Process a [`FinalAnswerStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`FinalAnswerStep`]): FinalAnswerStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the final answer.
    """
    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Ответ пользователю:**\n{final_answer.to_string()}\n",
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"status": "done"},
        )
    else:
        yield gr.ChatMessage(
            role="assistant", content=f"**Ответ пользователю:** {str(final_answer)}", metadata={"status": "done"}
        )


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    """Extract Gradio ChatMessage objects from agent steps with proper nesting.

    Args:
        step_log: The step log to display as gr.ChatMessage objects.
        skip_model_outputs: If True, skip the model outputs when creating the gr.ChatMessage objects:
            This is used for instance when streaming model outputs have already been displayed.
    """

    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    intermediate_text = ""

    for event in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
            intermediate_text = ""
            for message in pull_messages_from_step(
                event,
                # If we're streaming model outputs, no need to display them twice
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
        elif isinstance(event, ChatMessageStreamDelta):
            intermediate_text += event.content or ""
            yield intermediate_text


class TravelGradioUI:
    def __init__(self, agent: MultiStepAgent):
        self.agent = agent

    def interact_with_agent(self, prompt, messages, session_state):
        if not prompt.strip():
            return

        # Get the agent type from the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt, metadata={"status": "done"}))
            yield messages

            thinking = "⏳ Думаю над ответом..."
            loading_message = gr.ChatMessage(
                role="assistant", content=thinking, metadata={"status": "pending", "is_loading": True}
            )
            messages.append(loading_message)
            yield messages

            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                if isinstance(msg, gr.ChatMessage):
                    if messages[-1].content == thinking:
                        messages[-1].content = ""

                    messages[-1].metadata["status"] = "done"
                    messages.append(msg)
                elif isinstance(msg, str):  # Then it's only a completion delta
                    msg = msg.replace("<", r"\<").replace(">", r"\>")  # HTML tags seem to break Gradio Chatbot
                    if messages[-1].metadata["status"] == "pending":
                        messages[-1].content = msg
                    else:
                        messages.append(gr.ChatMessage(role="assistant", content=msg, metadata={"status": "pending"}))
                yield messages

            yield messages
        except Exception as e:
            yield messages
            raise gr.Error(f"Error in interaction: {str(e)}")

    def log_user_message(self, text_input):
        return text_input, "", gr.Button(interactive=False)

    def launch(self, server_name: str, share: bool = True, **kwargs):
        self.create_app().launch(debug=True, server_name=server_name, share=share, **kwargs)

    def create_app(self):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            session_state = gr.State({})
            stored_messages = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    "# 🧭 Туристический помощник по России\n"
                    "👋 Привет! Я помогу тебе спланировать незабываемое путешествие по России — от уютных кафе Петербурга до музеев Суздаля и пляжей Сочи.\n"
                    "Хочешь культурный маршрут, гастротур или просто отдохнуть на выходных? Просто задай вопрос!"
                )

                with gr.Group():
                    gr.Markdown("**Запрос**")
                    text_input = gr.Textbox(
                        lines=3,
                        label="",
                        container=False,
                        placeholder="Введите запрос и нажмите Shift+Enter или выберите пример ниже",
                    )
                    submit_btn = gr.Button("Отправить", variant="primary")

                gr.HTML(
                    "<br><br><h4><center><a target='_blank' href='https://github.com/miem-refugees/travel-agent'><b>Github</b></a></center></h4>"
                )

            chatbot = gr.Chatbot(
                label="Помощник 🕵🏻‍♂️",
                type="messages",
                avatar_images=(
                    None,
                    "https://storage.yandexcloud.net/travel-rag/static/kizaru-s.jpeg",
                ),
                resizeable=True,
                scale=1,
                show_copy_button=True,
                watermark="by ksusonic and seara",
                placeholder="# 💬 Попробуй спросить:",
            )

            # ⛳️ Examples section (non-togglable, will be hidden on submit)
            examples_box = gr.Column(visible=True)
            with examples_box:
                gr.Markdown("💡 Примеры запросов:")
                example_queries = [
                    "Где в Москве лучший бизнес-ланч?",
                    "Составь маршрут по необычным кафе и барам в Новосибирске.",
                    "Составь детский маршрут по Москве: музеи, парки, интересные кафе.",
                    "Можешь спланировать 7-дневное путешествие по Золотому кольцу России?",
                    "Сделай гастрономический тур по югу России: Ростов-на-Дону, Краснодар, Сочи.",
                ]

                def handle_example_click(example_text):
                    return example_text, example_text, gr.update(visible=False)

                for example in example_queries:
                    gr.Button(
                        example,
                        variant="secondary",
                        size="md",
                    ).click(
                        partial(handle_example_click, example), outputs=[text_input, stored_messages, examples_box]
                    ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot])

            text_input.submit(
                self.log_user_message,
                [text_input],
                [stored_messages, text_input, submit_btn],
            ).then(
                lambda: gr.update(visible=False),  # Hide examples
                None,
                [examples_box],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Введите запрос и нажмите Shift+Enter или выберите пример ниже",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input],
                [stored_messages, text_input, submit_btn],
            ).then(
                lambda: gr.update(visible=False),  # Hide examples
                None,
                [examples_box],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Введите запрос и нажмите Shift+Enter или выберите пример ниже",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        return demo
