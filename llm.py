import time # 用于可能的自定义重试逻辑中的休眠
from llama_cpp import Llama
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError # 导入具体的 OpenAI 错误类型
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 lang: str = "English",
                 openai_timeout: float = 60.0, # 增加 OpenAI 请求超时时间，单位秒 (原来 httpx 默认可能较短)
                 openai_max_retries: int = 3    # 增加 OpenAI 客户端最大重试次数
                ):
        self.model = model
        self.lang = lang
        self.api_key_provided = bool(api_key) # 标记是否提供了 api_key

        if self.api_key_provided:
            logger.info(f"Initializing OpenAI client with model: {model}, timeout: {openai_timeout}s, max_retries: {openai_max_retries}")
            try:
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                # 即使 OpenAI 初始化失败，也可能希望回退到 Llama (如果这是期望行为)
                # 或者在这里直接抛出异常，取决于项目的需求
                # 为了与原始逻辑保持相似，这里允许继续，但 generate 时会出问题
                self.llm = None # 明确标记llm未成功初始化
                self.api_key_provided = False # 更新标记
        else:
            logger.info("API key not provided. Initializing local Llama model Qwen/Qwen2.5-3B-Instruct-GGUF.")
            try:
                self.llm = Llama.from_pretrained(
                    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                    n_ctx=5_000,
                    n_threads=4,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Llama model: {e}")
                self.llm = None # 明确标记llm未成功初始化

        if self.llm is None:
            logger.error("LLM client (neither OpenAI nor Llama) could not be initialized!")
            # 根据项目需求，这里可以 raise RuntimeError("LLM could not be initialized")

    def generate(self, messages: list[dict]) -> str:
        if not self.llm:
            logger.error("LLM client is not initialized. Cannot generate text.")
            # 根据您的错误处理策略，可以返回空字符串或抛出异常
            # raise RuntimeError("LLM client not initialized, cannot generate.")
            return "Error: LLM not initialized."

        if isinstance(self.llm, OpenAI):
            logger.debug(f"Attempting to generate completion with OpenAI model: {self.model}. Messages: {messages}")
            try:
                response = self.llm.chat.completions.create(
                    messages=messages,
                    temperature=0,
                    model=self.model
                )
                content = response.choices[0].message.content
                logger.debug(f"Successfully received response from OpenAI: {content[:100]}...") # 日志截断，避免过长
                return content
            except APIConnectionError as e:
                logger.error(f"OpenAI API Connection Error during generation: {e}")
                raise # 将异常重新抛出，让上层调用者处理
            except APITimeoutError as e:
                logger.error(f"OpenAI API Timeout Error during generation: {e}")
                raise
            except RateLimitError as e:
                logger.error(f"OpenAI API Rate Limit Error during generation: {e}")
                # 这里可以考虑实现一个更复杂的退避重试逻辑，或者简单地让上层处理
                raise
            except APIStatusError as e: # 处理其他HTTP错误，如 4xx, 5xx
                logger.error(f"OpenAI API Status Error (status_code={e.status_code}) during generation: {e.message}")
                raise
            except Exception as e: # 捕获其他未预料到的 OpenAI 相关错误
                logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
                raise
        elif isinstance(self.llm, Llama):
            logger.debug(f"Attempting to generate completion with Llama model. Messages: {messages}")
            try:
                response = self.llm.create_chat_completion(messages=messages, temperature=0)
                content = response["choices"][0]["message"]["content"]
                logger.debug(f"Successfully received response from Llama: {content[:100]}...")
                return content
            except Exception as e:
                logger.error(f"An error occurred during Llama model generation: {e}")
                raise # 或者返回一个错误提示字符串
        else:
            logger.error("LLM type not recognized or not initialized properly.")
            return "Error: LLM type not recognized."


def set_global_llm(api_key: str = None,
                     base_url: str = None,
                     model: str = None,
                     lang: str = "English",
                     openai_timeout: float = 60.0, # 与 LLM 类保持一致
                     openai_max_retries: int = 3   # 与 LLM 类保持一致
                    ):
    global GLOBAL_LLM
    logger.info("Setting global LLM...")
    GLOBAL_LLM = LLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        lang=lang,
        openai_timeout=openai_timeout,
        openai_max_retries=openai_max_retries
    )

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.warning("Global LLM not set. Attempting to create a default one (likely Llama, unless OPENAI_API_KEY env var is picked up by OpenAI lib implicitly).")
        logger.warning("It's recommended to call `set_global_llm` explicitly with your desired configuration.")
        # 如果希望默认也尝试使用环境变量中的 OPENAI_API_KEY，可以这样做：
        # import os
        # default_api_key = os.getenv("OPENAI_API_KEY")
        # set_global_llm(api_key=default_api_key, model="gpt-3.5-turbo" if default_api_key else None)
        # 但为了更明确，这里还是让它默认创建 Llama (如果 api_key 为 None)
        set_global_llm() # 这会使用 LLM 构造函数中的默认 openai_timeout 和 openai_max_retries

    if GLOBAL_LLM and GLOBAL_LLM.llm is None: # 检查内部llm是否真的初始化成功
        logger.error("Global LLM object exists, but its internal llm client (OpenAI or Llama) failed to initialize.")
        # 在这种情况下，返回 GLOBAL_LLM 可能会导致后续调用 .generate() 时出错
        # 可以选择在这里抛出异常，或者允许返回，让 generate 方法中的检查来处理
        # raise RuntimeError("Failed to initialize the LLM client for the global LLM instance.")

    return GLOBAL_LLM
