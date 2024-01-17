import json
import os
from json import JSONDecodeError
from typing import Iterable, List, Optional

from injector import inject

from taskweaver.config.module_config import ModuleConfig
from taskweaver.llm import LLMApi
from taskweaver.llm.util import ChatMessageType, format_chat_message
from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Conversation, Memory, Post, Round, RoundCompressor
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.plugin import PluginRegistry
from taskweaver.misc.example import load_examples
from taskweaver.module.event_emitter import SessionEventEmitter
from taskweaver.role import PostTranslator, Role
from taskweaver.utils import read_yaml


class PlannerConfig(ModuleConfig):
    def _configure(self) -> None:
        self._set_name("planner")
        app_dir = self.src.app_base_path
        self.use_example = self._get_bool("use_example", True)
        self.prompt_file_path = self._get_path(
            "prompt_file_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "planner_prompt.yaml",
            ),
        )
        self.example_base_path = self._get_path(
            "example_base_path",
            os.path.join(
                app_dir,
                "planner_examples",
            ),
        )
        self.prompt_compression = self._get_bool("prompt_compression", False)
        self.compression_prompt_path = self._get_path(
            "compression_prompt_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "compression_prompt.yaml",
            ),
        )

        self.skip_planning = self._get_bool("skip_planning", False)
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dummy_plan.json",
            ),
            "r",
        ) as f:
            self.dummy_plan = json.load(f)


class Planner(Role):
    conversation_delimiter_message: str = "Let's start the new conversation!"
    ROLE_NAME: str = "Planner"

    @inject
    def __init__(
        self,
        config: PlannerConfig,
        logger: TelemetryLogger,
        event_emitter: SessionEventEmitter,
        llm_api: LLMApi,
        plugin_registry: PluginRegistry,
        round_compressor: Optional[RoundCompressor],
        post_translator: PostTranslator,
        plugin_only: bool = False,
    ):
        self.config = config
        self.logger = logger
        self.event_emitter = event_emitter
        self.llm_api = llm_api
        if plugin_only:
            self.available_plugins = [p for p in plugin_registry.get_list() if p.plugin_only is True]
        else:
            self.available_plugins = plugin_registry.get_list()

        self.planner_post_translator = post_translator

        self.prompt_data = read_yaml(self.config.prompt_file_path)

        if self.config.use_example:
            self.examples = self.get_examples()
        if len(self.available_plugins) == 0:
            self.logger.warning("No plugin is loaded for Planner.")
            self.plugin_description = "No plugin functions loaded."
        else:
            self.plugin_description = "    " + "\n    ".join(
                [f"{plugin.spec.plugin_description()}" for plugin in self.available_plugins],
            )
        self.instruction_template = self.prompt_data["instruction_template"]
        self.code_interpreter_introduction = self.prompt_data["code_interpreter_introduction"].format(
            plugin_description=self.plugin_description,
        )
        self.response_schema = self.prompt_data["planner_response_schema"]

        self.instruction = self.instruction_template.format(
            planner_response_schema=self.response_schema,
            CI_introduction=self.code_interpreter_introduction,
        )
        self.ask_self_cnt = 0
        self.max_self_ask_num = 3

        self.round_compressor = round_compressor
        self.compression_template = read_yaml(self.config.compression_prompt_path)["content"]

        self.logger.info("Planner initialized successfully")

    def compose_conversation_for_prompt(
        self,
        conv_rounds: List[Round],
        summary: Optional[str] = None,
    ) -> List[ChatMessageType]:
        conversation: List[ChatMessageType] = []

        for rnd_idx, chat_round in enumerate(conv_rounds):
            conv_init_message = None
            if rnd_idx == 0:
                conv_init_message = Planner.conversation_delimiter_message
                if summary is not None:
                    self.logger.debug(f"Summary: {summary}")
                    summary_message = (
                        f"\nThe context summary of the Planner's previous rounds" f" can refer to:\n{summary}\n\n"
                    )
                    conv_init_message += "\n" + summary_message

            for post in chat_round.post_list:
                if post.send_from == "Planner":
                    if post.send_to == "User" or post.send_to == "CodeInterpreter":
                        planner_message = self.planner_post_translator.post_to_raw_text(
                            post=post,
                        )
                        conversation.append(
                            format_chat_message(
                                role="assistant",
                                message=planner_message,
                            ),
                        )
                    elif (
                        post.send_to == "Planner"
                    ):  # self correction for planner response, e.g., format error/field check error
                        conversation.append(
                            format_chat_message(
                                role="assistant",
                                message=post.get_attachment(
                                    type=AttachmentType.invalid_response,
                                )[0],
                            ),
                        )  # append the invalid response to chat history
                        conversation.append(
                            format_chat_message(
                                role="user",
                                message="User: " + post.message,
                            ),
                        )  # append the self correction instruction message to chat history

                else:
                    if conv_init_message is not None:
                        message = post.send_from + ": " + conv_init_message + "\n" + post.message
                        conversation.append(
                            format_chat_message(role="user", message=message),
                        )
                        conv_init_message = None
                    else:
                        conversation.append(
                            format_chat_message(
                                role="user",
                                message=post.send_from + ": " + post.message,
                            ),
                        )

        return conversation

    def compose_prompt(self, rounds: List[Round]) -> List[ChatMessageType]:
        chat_history = [format_chat_message(role="system", message=self.instruction)]

        if self.config.use_example and len(self.examples) != 0:
            for conv_example in self.examples:
                conv_example_in_prompt = self.compose_conversation_for_prompt(
                    conv_example.rounds,
                )
                chat_history += conv_example_in_prompt

        summary = None
        if self.config.prompt_compression and self.round_compressor is not None:
            summary, rounds = self.round_compressor.compress_rounds(
                rounds,
                rounds_formatter=lambda _rounds: str(
                    self.compose_conversation_for_prompt(_rounds),
                ),
                use_back_up_engine=True,
                prompt_template=self.compression_template,
            )

        chat_history.extend(
            self.compose_conversation_for_prompt(
                rounds,
                summary=summary,
            ),
        )

        return chat_history

    def reply(
        self,
        memory: Memory,
        prompt_log_path: Optional[str] = None,
        use_back_up_engine: bool = False,
    ) -> Post:
        rounds = memory.get_role_rounds(role="Planner")
        assert len(rounds) != 0, "No chat rounds found for planner"
        new_post = self.event_emitter.create_post_proxy("Planner")

        new_post.update_status("composing prompt")
        chat_history = self.compose_prompt(rounds)

        def check_post_validity(post: Post):
            assert post.send_to is not None, "send_to field is None"
            assert post.send_to != "Planner", "send_to field should not be Planner"
            assert post.message is not None, "message field is None"
            assert (
                post.attachment_list[0].type == AttachmentType.init_plan
            ), f"attachment type {post.attachment_list[0].type} is not init_plan"
            assert (
                post.attachment_list[1].type == AttachmentType.plan
            ), f"attachment type {post.attachment_list[1].type} is not plan"
            assert (
                post.attachment_list[2].type == AttachmentType.current_plan_step
            ), "attachment type is not current_plan_step"

        new_post.update_status("calling LLM endpoint")
        if self.config.skip_planning and rounds[-1].post_list[-1].send_from == "User":
            self.config.dummy_plan["response"][0]["content"] += rounds[-1].post_list[-1].message
            llm_stream = [
                format_chat_message("assistant", json.dumps(self.config.dummy_plan)),
            ]
        else:
            llm_stream = self.llm_api.chat_completion_stream(
                chat_history,
                use_backup_engine=use_back_up_engine,
                use_smoother=True,
            )

        llm_output: List[str] = []
        try:

            def stream_filter(s: Iterable[ChatMessageType]):
                is_first_chunk = True
                for c in s:
                    if is_first_chunk:
                        new_post.update_status("receiving LLM response")
                        is_first_chunk = False
                    llm_output.append(c["content"])
                    yield c

            self.planner_post_translator.raw_text_to_post(
                post_proxy=new_post,
                llm_output=stream_filter(llm_stream),
                validation_func=check_post_validity,
            )
        except (JSONDecodeError, AssertionError) as e:
            self.logger.error(f"Failed to parse LLM output due to {str(e)}")
            new_post.error(f"failed to parse LLM output due to {str(e)}")
            new_post.update_attachment(
                "".join(llm_output),
                AttachmentType.invalid_response,
            )
            new_post.update_message(
                f"Failed to parse Planner output due to {str(e)}."
                f"The output format should follow the below format:"
                f"{self.prompt_data['planner_response_schema']}"
                "Please try to regenerate the output.",
            )
            new_post.update_send_to("Planner")
            self.ask_self_cnt += 1
            if self.ask_self_cnt > self.max_self_ask_num:  # if ask self too many times, return error message
                self.ask_self_cnt = 0
                new_post.end("Planner failed to generate response")
                raise Exception(f"Planner failed to generate response because {str(e)}")
        if prompt_log_path is not None:
            self.logger.dump_log_file(chat_history, prompt_log_path)
        return new_post.end()

    def get_examples(self) -> List[Conversation]:
        example_conv_list = load_examples(self.config.example_base_path)
        return example_conv_list
