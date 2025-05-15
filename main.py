import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm # 假设 llm.py 在同一目录或在 PYTHONPATH 中可访问
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    """
    检索和处理 Zotero 图书馆条目。
    """
    logger.debug(f"Initializing Zotero client for user ID: {id}")
    zot = zotero.Zotero(id, 'user', key)
    
    logger.debug("Fetching all collections from Zotero...")
    collections_raw = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections_raw}
    logger.debug(f"Fetched {len(collections)} collections.")

    logger.debug("Fetching all conference papers, journal articles, and preprints from Zotero...")
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    logger.debug(f"Fetched {len(corpus)} raw items.")
    
    # 筛选没有摘要的条目
    corpus = [c for c in corpus if c['data'].get('abstractNote', '') != '']
    logger.debug(f"{len(corpus)} items remaining after filtering for abstracts.")

    def get_collection_path(col_key:str) -> str:
        """
        递归构建 Zotero 集合的完整路径。
        """
        if not col_key or col_key not in collections:
            logger.warning(f"Collection key '{col_key}' not found in fetched collections. Returning empty path.")
            return ""
        collection_data = collections[col_key]['data']
        parent_collection_key = collection_data.get('parentCollection')
        current_collection_name = collection_data.get('name', 'Unnamed Collection')

        if parent_collection_key:
            parent_path = get_collection_path(parent_collection_key)
            return f"{parent_path}/{current_collection_name}" if parent_path else current_collection_name
        else:
            return current_collection_name

    logger.debug("Processing collection paths for corpus items...")
    for i, c in enumerate(corpus):
        item_collections = c['data'].get('collections', [])
        paths = []
        if item_collections: # Ensure there are collections to process
            paths = [get_collection_path(col) for col in item_collections if col] # Add check for valid col key
        c['paths'] = paths
        if i % 100 == 0: # Log progress for large libraries
             logger.trace(f"Processed paths for {i+1}/{len(corpus)} items.")
    logger.debug("Finished processing collection paths.")
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    """
    根据 gitignore 样式的集合路径模式筛选 Zotero 文集。
    """
    if not pattern:
        logger.info("No Zotero ignore pattern provided. Skipping filtering.")
        return corpus

    _,filename = mkstemp(prefix="zotero_ignore_", suffix=".txt")
    logger.debug(f"Created temporary ignore file: {filename}")
    try:
        with open(filename,'w') as file:
            file.write(pattern)
        
        base_dir = os.getcwd() 
        matcher = parse_gitignore(filename, base_dir=base_dir)
        
        new_corpus = []
        logger.debug(f"Filtering {len(corpus)} items based on pattern in '{filename}' relative to '{base_dir}'...")
        for c_idx, c in enumerate(corpus):
            item_paths = c.get('paths', [])
            if not item_paths: 
                new_corpus.append(c)
                logger.trace(f"Item {c_idx+1} has no paths, keeping.")
                continue

            match_results = [matcher(p) for p in item_paths]
            if not any(match_results):
                new_corpus.append(c)
                logger.trace(f"Item {c_idx+1} (paths: {item_paths}) did not match ignore patterns, keeping.")
            else:
                logger.trace(f"Item {c_idx+1} (paths: {item_paths}) matched ignore patterns, removing.")
        logger.info(f"Corpus filtered. Original size: {len(corpus)}, New size: {len(new_corpus)}")
    finally:
        os.remove(filename)
        logger.debug(f"Removed temporary ignore file: {filename}")
    return new_corpus


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    """
    根据查询从 arXiv 检索新论文。
    """
    logger.info(f"Retrieving arXiv papers with query: '{query}', debug mode: {debug}")
    client = arxiv.Client(num_retries=10, page_size=100, delay_seconds=10)
    
    rss_url = f"https://rss.arxiv.org/atom/{query}"
    logger.debug(f"Fetching RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if feed.bozo:
        logger.warning(f"RSS feed parsing issue: {feed.bozo_exception}")
    if 'Feed error for query' in feed.feed.get('title', ''):
        logger.error(f"Invalid ARXIV_QUERY: {query}. Feed title: {feed.feed.get('title', 'N/A')}")
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    papers = []
    if not debug:
        all_paper_ids = []
        if feed.entries:
            logger.debug(f"Processing {len(feed.entries)} entries from RSS feed to find new papers.")
            for entry_idx, entry in enumerate(feed.entries):
                announce_type = getattr(entry, 'arxiv_announce_type', None)
                if announce_type == 'new':
                    source_id_field = entry.get('id', '')  # Atom 中的 <id> 标签，例如 http://arxiv.org/abs/xxxx.yyyyvN 或 oai:arXiv.org:xxxx.yyyy
                    link_field = entry.get('link', '')    # <link rel="alternate">，通常是 http://arxiv.org/abs/xxxx.yyyyvN

                    processed_id_candidate = None

                    # 优先级 1: 从链接字段提取 (通常是 http://arxiv.org/abs/IDvV)
                    if link_field and '/abs/' in link_field:
                        processed_id_candidate = link_field.split('/abs/')[-1]
                    # 优先级 2: 如果是 URL，则从 id 字段提取 (http://arxiv.org/abs/IDvV)
                    elif source_id_field and '/abs/' in source_id_field:
                        processed_id_candidate = source_id_field.split('/abs/')[-1]
                    # 优先级 3: 如果是 OAI URN，则从 id 字段提取 (oai:arXiv.org:ID 或 oai:arXiv.org:IDvV)
                    elif source_id_field and source_id_field.startswith('oai:arXiv.org:'):
                        processed_id_candidate = source_id_field.replace('oai:arXiv.org:', '')
                    # source_id_field 中其他潜在简单 ID 格式的回退方案
                    elif source_id_field and not ('/' in source_id_field or ':' in source_id_field or ' ' in source_id_field):
                         # 假设它可能是纯 ID，如 "2401.12345" 或 "2401.12345v1"
                         processed_id_candidate = source_id_field

                    if processed_id_candidate:
                        # 为保持一致性，删除版本部分 (例如 v1, v2)，尽管 arxiv 库可以处理版本。
                        final_id = processed_id_candidate.split('v')[0]
                        if final_id: 
                            all_paper_ids.append(final_id)
                            logger.trace(f"Entry {entry_idx+1}: Extracted new paper ID '{final_id}' (from candidate '{processed_id_candidate}')")
                        else:
                            logger.warning(f"Entry {entry_idx+1}: Candidate ID '{processed_id_candidate}' became empty after version split. Source ID: '{source_id_field}', Link: '{link_field}'")
                    else:
                        logger.warning(f"Entry {entry_idx+1}: Could not extract valid paper ID. Announce type: '{announce_type}', Source ID: '{source_id_field}', Link: '{link_field}'")
        
        logger.info(f"Found {len(all_paper_ids)} new paper IDs from RSS feed for batch processing.")

        if not all_paper_ids:
            logger.info("No new paper IDs extracted from the RSS feed for the given query.")
            return []

        batch_size = 50 
        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv paper details")
        for i in range(0, len(all_paper_ids), batch_size):
            batch_ids = all_paper_ids[i:i+batch_size]
            logger.debug(f"Fetching batch {i//batch_size + 1} with IDs: {batch_ids}")
            try:
                search = arxiv.Search(id_list=batch_ids) # sort_by 对于 id_list 不是严格必需的
                batch_results = [ArxivPaper(p) for p in client.results(search)]
                papers.extend(batch_results)
                bar.update(len(batch_ids)) 
            except Exception as e:
                # 记录特定错误以及导致错误的 ID
                logger.error(f"Error fetching batch {i//batch_size + 1} of arXiv papers (IDs: {batch_ids}): {e}")
        bar.close()

    else: # Debug mode
        logger.debug("Debug mode: Retrieving latest 5 papers from 'cat:cs.AI'")
        try:
            search = arxiv.Search(query='cat:cs.AI', max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate)
            for result in client.results(search):
                papers.append(ArxivPaper(result))
                if len(papers) >= 5: 
                    break
        except Exception as e:
            logger.error(f"Error fetching debug arXiv papers: {e}")
    
    logger.info(f"Retrieved {len(papers)} ArxivPaper objects.")
    return papers


# 参数解析设置
# 以下部分涉及命令行参数的解析
parser = argparse.ArgumentParser(description='Recommender system for academic papers. 学术论文推荐系统。')

def add_argument_with_env_fallback(*args, **kwargs):
    """
    将参数添加到解析器，并从环境变量（如果可用）中设置其默认值。
    处理布尔型和其他类型的类型转换。
    """
    arg_dest_name = kwargs.get('dest')
    if not arg_dest_name:
        for arg_name in args:
            if arg_name.startswith('--'):
                arg_dest_name = arg_name.lstrip('-').replace('-', '_')
                break
        if not arg_dest_name and args: 
             arg_dest_name = args[0].lstrip('-').replace('-', '_')

    if not arg_dest_name:
        raise ValueError("Could not determine destination name for argument.")

    env_var_name = arg_dest_name.upper()
    env_value_str = os.environ.get(env_var_name)

    if env_value_str is not None and env_value_str != '':
        arg_type = kwargs.get('type', str) 
        try:
            if arg_type == bool:
                processed_env_value = env_value_str.lower() in ['true', '1', 'yes', 't']
            elif callable(arg_type):
                processed_env_value = arg_type(env_value_str)
            else: 
                processed_env_value = env_value_str
            kwargs['default'] = processed_env_value
            logger.debug(f"Argument '{arg_dest_name}' (env: {env_var_name}): Using value from environment: '{processed_env_value}' (original env string: '{env_value_str}')")
        except ValueError as e:
            logger.warning(f"Argument '{arg_dest_name}' (env: {env_var_name}): Could not convert environment variable value '{env_value_str}' to type {arg_type}. Using original default. Error: {e}")
    elif 'default' in kwargs:
         logger.trace(f"Argument '{arg_dest_name}' (env: {env_var_name}): No environment variable set or it's empty. Using provided default: '{kwargs['default']}'.")
    else:
         logger.trace(f"Argument '{arg_dest_name}' (env: {env_var_name}): No environment variable set and no default provided.")

    action = kwargs.get('action')
    if action in ['store_true', 'store_false'] and 'type' in kwargs:
        del kwargs['type']
        if 'default' in kwargs and isinstance(kwargs['default'], bool):
             pass
    parser.add_argument(*args, **kwargs)

if __name__ == '__main__':
    # 使用辅助函数定义参数
    add_argument_with_env_fallback('--zotero_id', type=str, help='Zotero user ID. Zotero 用户 ID。', required=False)
    add_argument_with_env_fallback('--zotero_key', type=str, help='Zotero API key. Zotero API 密钥。', required=False)
    add_argument_with_env_fallback('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern. 要忽略的 Zotero 集合，使用 gitignore 样式模式。', default="")
    add_argument_with_env_fallback('--send_empty', action='store_true', help='If get no arxiv paper, send empty email (default: False). 如果没有 arXiv 论文，则发送空邮件（默认值：False）。')
    add_argument_with_env_fallback('--max_paper_num', type=int, help='Maximum number of papers to recommend (default: 100, -1 for no limit). 推荐论文的最大数量（默认值：100，-1 表示无限制）。',default=100)
    add_argument_with_env_fallback('--arxiv_query', type=str, help='Arxiv search query. Arxiv 搜索查询。', required=False)
    add_argument_with_env_fallback('--smtp_server', type=str, help='SMTP server. SMTP 服务器。', required=False)
    add_argument_with_env_fallback('--smtp_port', type=int, help='SMTP port. SMTP 端口。', default=587)
    add_argument_with_env_fallback('--sender', type=str, help='Sender email address. 发件人电子邮件地址。', required=False)
    add_argument_with_env_fallback('--receiver', type=str, help='Receiver email address. 收件人电子邮件地址。', required=False)
    add_argument_with_env_fallback('--sender_password', type=str, help='Sender email password. 发件人电子邮件密码。', required=False)
    add_argument_with_env_fallback("--use_llm_api", action='store_true', help="Use OpenAI API to generate TLDR (default: False). 使用 OpenAI API 生成 TLDR（默认值：False）。")
    add_argument_with_env_fallback("--openai_api_key", type=str, help="OpenAI API key (required if --use_llm_api is set). OpenAI API 密钥（如果设置了 --use_llm_api，则为必需）。", default=None)
    add_argument_with_env_fallback("--openai_api_base", type=str, help="OpenAI API base URL. OpenAI API 基本 URL。", default="https://api.openai.com/v1")
    add_argument_with_env_fallback("--model_name", type=str, help="LLM Model Name. LLM 模型名称。", default="gpt-4o")
    add_argument_with_env_fallback("--language", type=str, help="Language of TLDR. TLDR 的语言。", default="English")
    add_argument_with_env_fallback("--openai_timeout", type=float, help="Timeout for OpenAI API requests in seconds (default: 60.0). OpenAI API 请求的超时时间（秒）（默认值：60.0）。", default=60.0)
    add_argument_with_env_fallback("--openai_max_retries", type=int, help="Maximum number of retries for OpenAI API requests (default: 3). OpenAI API 请求的最大重试次数（默认值：3）。", default=3)
    parser.add_argument('--debug', action='store_true', help='Debug mode (default: False). 调试模式（默认值：False）。')
    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    required_if_no_env = {
        'zotero_id': args.zotero_id,
        'zotero_key': args.zotero_key,
        'arxiv_query': args.arxiv_query,
        'smtp_server': args.smtp_server,
        'sender': args.sender,
        'receiver': args.receiver,
        'sender_password': args.sender_password
    }
    missing_args = [k for k, v in required_if_no_env.items() if v is None]
    if missing_args:
        parser.error(f"The following arguments (or their corresponding environment variables) are required: {', '.join(missing_args)}")

    if args.use_llm_api and args.openai_api_key is None:
        parser.error("--openai_api_key is required when --use_llm_api is set.")
    
    logger.info("Script started.")
    logger.debug(f"Arguments: {args}")

    try:
        logger.info("Retrieving Zotero corpus...")
        corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
        logger.info(f"Retrieved {len(corpus)} initial items from Zotero.")
        if args.zotero_ignore:
            logger.info(f"Applying Zotero ignore pattern: {args.zotero_ignore}")
            corpus = filter_corpus(corpus, args.zotero_ignore)
            logger.info(f"Remaining {len(corpus)} items in corpus after filtering.")
        
        logger.info("Retrieving Arxiv papers...")
        papers = get_arxiv_paper(args.arxiv_query, args.debug)
        
        if not papers: 
            logger.info("No new arXiv papers found for the given query.")
            if not args.send_empty:
                logger.info("SEND_EMPTY is false, exiting.")
                exit(0)
            html_content = render_email([]) 
        else:
            logger.info(f"Retrieved {len(papers)} papers from Arxiv. Reranking...")
            if not corpus:
                 logger.warning("Zotero corpus is empty. Reranking might not be effective or might be skipped.")
            papers = rerank_paper(papers, corpus) 
            
            if args.max_paper_num != -1 and len(papers) > args.max_paper_num:
                logger.info(f"Limiting papers to a maximum of {args.max_paper_num}.")
                papers = papers[:args.max_paper_num]
            
            if args.use_llm_api:
                logger.info(f"Setting up OpenAI API as global LLM. Model: {args.model_name}, Timeout: {args.openai_timeout}, Retries: {args.openai_max_retries}")
                set_global_llm(
                    api_key=args.openai_api_key,
                    base_url=args.openai_api_base,
                    model=args.model_name,
                    lang=args.language,
                    openai_timeout=args.openai_timeout,
                    openai_max_retries=args.openai_max_retries
                )
            else:
                logger.info("Setting up Local LLM as global LLM.")
                set_global_llm(lang=args.language) 
            
            logger.info("Rendering email content...")
            html_content = render_email(papers)

        logger.info("Sending email...")
        send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html_content)
        logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

    except Exception as e:
        logger.error(f"An error occurred in the main script: {e}")
        logger.exception("Traceback:") 
        sys.exit(1)
