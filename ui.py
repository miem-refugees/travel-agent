import re

import gradio as gr
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.models import ChatMessageStreamDelta


def get_step_footnote_content(step_log: MemoryStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and hasattr(
        step_log, "output_token_count"
    ):
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = (
            f" | Duration: {round(float(step_log.duration), 2)}"
            if step_log.duration
            else None
        )
        step_footnote += step_duration
    step_footnote_content = (
        f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    )
    return step_footnote_content


def pull_messages_from_step(step_log: MemoryStep, skip_model_outputs: bool = False):
    """Extract ChatMessage objects from agent steps with proper nesting.

    Args:
        step_log: The step log to display as gr.ChatMessage objects.
        skip_model_outputs: If True, skip the model outputs when creating the gr.ChatMessage objects:
            This is used for instance when streaming model outputs have already been displayed.
    """
    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = (
            f"Step {step_log.step_number}"
            if step_log.step_number is not None
            else "Step"
        )

        # First yield the thought/reasoning from the LLM
        if not skip_model_outputs:
            yield gr.ChatMessage(
                role="assistant",
                content=f"**{step_number}**",
                metadata={"status": "done"},
            )
        elif (
            skip_model_outputs
            and hasattr(step_log, "model_output")
            and step_log.model_output is not None
        ):
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(
                r"```\s*<end_code>", "```", model_output
            )  # handles ```<end_code>
            model_output = re.sub(
                r"<end_code>\s*```", "```", model_output
            )  # handles <end_code>```
            model_output = re.sub(
                r"```\s*\n\s*<end_code>", "```", model_output
            )  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(
                role="assistant", content=model_output, metadata={"status": "done"}
            )

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(
                    r"```.*?\n", "", content
                )  # Remove existing code blocks
                content = re.sub(
                    r"\s*<end_code>\s*", "", content
                )  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

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
        if hasattr(step_log, "observations") and (
            step_log.observations is not None and step_log.observations.strip()
        ):  # Only yield execution logs if there's actual content
            log_content = step_log.observations.strip()
            if log_content:
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                yield gr.ChatMessage(
                    role="assistant",
                    content=f"```bash\n{log_content}\n",
                    metadata={"title": "📝 Лог выполнения", "status": "done"},
                )

        # Display any errors
        if hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Ошибка", "status": "done"},
            )

        # Update parent message metadata to done status without yielding a new message
        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                path_image = AgentImage(image).to_string()
                yield gr.ChatMessage(
                    role="assistant",
                    content={
                        "path": path_image,
                        "mime_type": f"image/{path_image.split('.')[-1]}",
                    },
                    metadata={"title": "🖼️ Изображение", "status": "done"},
                )

        # Handle standalone errors but not from tool calls
        if hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Ошибка", "status": "done"},
            )

        yield gr.ChatMessage(
            role="assistant",
            content=get_step_footnote_content(step_log, step_number),
            metadata={"status": "done"},
        )
        yield gr.ChatMessage(
            role="assistant", content="-----", metadata={"status": "done"}
        )

    elif isinstance(step_log, PlanningStep):
        yield gr.ChatMessage(
            role="assistant",
            content="**Шаг планирования**",
            metadata={"status": "done"},
        )
        yield gr.ChatMessage(
            role="assistant", content=step_log.plan, metadata={"status": "done"}
        )
        yield gr.ChatMessage(
            role="assistant",
            content=get_step_footnote_content(step_log, "Шаг планирования"),
            metadata={"status": "done"},
        )
        yield gr.ChatMessage(
            role="assistant", content="-----", metadata={"status": "done"}
        )

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        if isinstance(final_answer, AgentText):
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Финальный ответ:**\n{final_answer.to_string()}\n",
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
                role="assistant",
                content=f"**Финальный ответ:** {str(final_answer)}",
                metadata={"status": "done"},
            )

    else:
        raise ValueError(f"Неподдерживаемый тип шага: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    intermediate_text = ""

    for step_log in agent.run(
        task,
        images=task_images,
        stream=True,
        reset=reset_agent_memory,
        additional_args=additional_args,
    ):
        # Track tokens if model provides them
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, (ActionStep, PlanningStep)):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        if isinstance(step_log, MemoryStep):
            intermediate_text = ""
            for message in pull_messages_from_step(
                step_log,
                # If we're streaming model outputs, no need to display them twice
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
        elif isinstance(step_log, ChatMessageStreamDelta):
            intermediate_text += step_log.content or ""
            yield intermediate_text


class TravelGradioUI:
    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        self.agent = agent

    def interact_with_agent(self, prompt, messages, session_state):
        if not prompt.strip():
            return

        # Get the agent type from the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(
                gr.ChatMessage(role="user", content=prompt, metadata={"status": "done"})
            )
            yield messages

            loading_message = gr.ChatMessage(
                role="assistant",
                content="⏳ Думаю над ответом...",
                metadata={"status": "pending", "is_loading": True},
            )
            messages.append(loading_message)
            yield messages

            response_started = False

            for msg in stream_to_gradio(
                session_state["agent"], task=prompt, reset_agent_memory=False
            ):
                if isinstance(msg, gr.ChatMessage):
                    if not response_started:
                        messages[-1] = msg
                        response_started = True
                    else:
                        messages.append(msg)
                elif isinstance(msg, str):  # Then it's only a completion delta
                    try:
                        if not response_started:
                            messages[-1] = gr.ChatMessage(
                                role="assistant",
                                content=msg,
                                metadata={"status": "pending"},
                            )
                            response_started = True
                        elif messages[-1].metadata["status"] == "pending":
                            messages[-1].content = msg
                        else:
                            messages.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content=msg,
                                    metadata={"status": "pending"},
                                )
                            )
                    except Exception as e:
                        raise e
                yield messages

            if not response_started:
                messages.pop()
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content="Не удалось получить ответ. Пожалуйста, попробуйте еще раз.",
                        metadata={"status": "done"},
                    )
                )
                yield messages

            yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")

            if messages and messages[-1].metadata.get("is_loading"):
                messages.pop()
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=f"💥 Ошибка: {str(e)}",
                    metadata={"status": "done"},
                )
            )
            yield messages

    def log_user_message(self, text_input):
        return text_input, "", gr.Button(interactive=False)

    def launch(self, share: bool = True, **kwargs):
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    "# Туристический помощник 🕵🏻‍♂️\n"
                    "> Я помогу тебе подобрать лучший вариант для путешествия в России и расскажу об интересных местах, кафе, и музеях!"
                )

                with gr.Group():
                    gr.Markdown("**Запрос**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Введите запрос и нажмите Shift+Enter или кнопку 'Отправить'",
                    )
                    submit_btn = gr.Button("Отправить", variant="primary")

                gr.HTML(
                    "<br><br><h4><center><a target='_blank' href='https://github.com/miem-refugees/travel-agent'><b>Github travel-agent</b></a></center></h4>"
                )

            # Main chat interface with custom styling for loading indicators
            chatbot = gr.Chatbot(
                label="Помощник 🕵🏻‍♂️",
                type="messages",
                avatar_images=(
                    None,
                    "https://storage.yandexcloud.net/travel-rag/static/kizaru-s.jpeg",
                ),
                resizeable=True,
                scale=1,
            )

            # Set up event handlers
            text_input.submit(
                self.log_user_message,
                [text_input],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Введите запрос и нажмите Shift+Enter или кнопку 'Отправить'",
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
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Введите запрос и нажмите Shift+Enter или кнопку 'Отправить'",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        return demo
