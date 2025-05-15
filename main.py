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
from llm import set_global_llm # Assuming llm.py is in the same directory or accessible in PYTHONPATH
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    """
    Retrieves and processes Zotero library items.
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
    
    # Filter out items without abstracts
    corpus = [c for c in corpus if c['data'].get('abstractNote', '') != '']
    logger.debug(f"{len(corpus)} items remaining after filtering for abstracts.")

    def get_collection_path(col_key:str) -> str:
        """
        Recursively builds the full path for a Zotero collection.
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
    Filters the Zotero corpus based on a gitignore-style pattern for collection paths.
    """
    if not pattern:
        logger.info("No Zotero ignore pattern provided. Skipping filtering.")
        return corpus

    _,filename = mkstemp(prefix="zotero_ignore_", suffix=".txt")
    logger.debug(f"Created temporary ignore file: {filename}")
    try:
        with open(filename,'w') as file:
            file.write(pattern)
        
        # Ensure base_dir is a valid directory. Using current directory if not specified.
        base_dir = os.getcwd() 
        matcher = parse_gitignore(filename, base_dir=base_dir)
        
        new_corpus = []
        logger.debug(f"Filtering {len(corpus)} items based on pattern in '{filename}' relative to '{base_dir}'...")
        for c_idx, c in enumerate(corpus):
            item_paths = c.get('paths', [])
            if not item_paths: # If an item has no collection paths, it won't be matched by patterns usually targeting paths.
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
    Retrieves new papers from arXiv based on a query.
    """
    logger.info(f"Retrieving arXiv papers with query: '{query}', debug mode: {debug}")
    client = arxiv.Client(num_retries=10, page_size=100, delay_seconds=10) # Increased page_size for fewer requests
    
    # Use feedparser to get IDs of *new* papers first
    # This is generally more reliable for "new" papers than relying solely on arxiv.Search sort order
    rss_url = f"https://rss.arxiv.org/atom/{query}"
    logger.debug(f"Fetching RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if feed.bozo: # feed.bozo is true if a feed is not well-formed
        logger.warning(f"RSS feed parsing issue: {feed.bozo_exception}")
    if 'Feed error for query' in feed.feed.get('title', ''): # Check if title exists
        logger.error(f"Invalid ARXIV_QUERY: {query}. Feed title: {feed.feed.get('title', 'N/A')}")
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    papers = []
    if not debug:
        all_paper_ids = []
        if feed.entries:
            all_paper_ids = [entry.id.split('/abs/')[-1].split('v')[0] for entry in feed.entries if hasattr(entry, 'arxiv_announce_type') and entry.arxiv_announce_type == 'new']
            # Alternative ID extraction if the above is problematic:
            # all_paper_ids = [entry.link.split('/abs/')[-1] for entry in feed.entries if 'new' in entry.get('arxiv_announce_type','')]
        logger.info(f"Found {len(all_paper_ids)} new paper IDs from RSS feed.")

        if not all_paper_ids:
            logger.info("No new paper IDs found in the RSS feed for the given query.")
            return []

        # Fetch full paper details in batches using arxiv.Search
        # The arxiv library might handle batching internally with client.results,
        # but explicit batching gives more control over progress display.
        batch_size = 50 # Keep batch size reasonable
        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv paper details")
        for i in range(0, len(all_paper_ids), batch_size):
            batch_ids = all_paper_ids[i:i+batch_size]
            try:
                search = arxiv.Search(id_list=batch_ids, sort_by=arxiv.SortCriterion.SubmittedDate) # Sorting might not be strictly needed for id_list
                # client.results directly yields results
                batch_results = [ArxivPaper(p) for p in client.results(search)]
                papers.extend(batch_results)
                bar.update(len(batch_ids)) # Update by number of IDs processed
            except Exception as e:
                logger.error(f"Error fetching batch {i//batch_size + 1} of arXiv papers (IDs: {batch_ids}): {e}")
                # Optionally, continue to the next batch or re-raise
        bar.close()

    else: # Debug mode
        logger.debug("Debug mode: Retrieving latest 5 papers from 'cat:cs.AI'")
        try:
            search = arxiv.Search(query='cat:cs.AI', max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate)
            # client.results is a generator
            for result in client.results(search):
                papers.append(ArxivPaper(result))
                if len(papers) >= 5: # Ensure we don't exceed 5 in debug
                    break
        except Exception as e:
            logger.error(f"Error fetching debug arXiv papers: {e}")
    
    logger.info(f"Retrieved {len(papers)} ArxivPaper objects.")
    return papers


# Argument parsing setup
parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument_with_env_fallback(*args, **kwargs):
    """
    Adds an argument to the parser and sets its default value from an environment variable if available.
    Handles type conversion for boolean and other types.
    """
    # Determine the 'dest' for the argument, which is used for os.environ.get() and parser.set_defaults()
    # If 'dest' is explicitly provided in kwargs, use it.
    # Otherwise, derive it from the first long option (e.g., '--my-arg' -> 'my_arg').
    arg_dest_name = kwargs.get('dest')
    if not arg_dest_name:
        for arg_name in args:
            if arg_name.startswith('--'):
                arg_dest_name = arg_name.lstrip('-').replace('-', '_')
                break
        if not arg_dest_name and args: # Fallback for positional or short options if needed, though less common for env vars
             arg_dest_name = args[0].lstrip('-').replace('-', '_')


    if not arg_dest_name:
        raise ValueError("Could not determine destination name for argument.")

    env_var_name = arg_dest_name.upper()
    env_value_str = os.environ.get(env_var_name)

    # If environment variable is set and not just an empty string (which os.environ.get might return for unset in some shells)
    if env_value_str is not None and env_value_str != '':
        arg_type = kwargs.get('type', str) # Default to string type if not specified
        
        try:
            if arg_type == bool:
                # For bool, 'action="store_true"' or 'action="store_false"' is common.
                # If type=bool is used, we interpret common string representations.
                processed_env_value = env_value_str.lower() in ['true', '1', 'yes', 't']
            elif callable(arg_type):
                processed_env_value = arg_type(env_value_str)
            else: # Should not happen if type is bool or callable
                processed_env_value = env_value_str
            
            # Override the default specified in add_argument call with the environment variable's value
            kwargs['default'] = processed_env_value
            logger.debug(f"Argument '{arg_dest_name}' (env: {env_var_name}): Using value from environment: '{processed_env_value}' (original env string: '{env_value_str}')")

        except ValueError as e:
            logger.warning(f"Argument '{arg_dest_name}' (env: {env_var_name}): Could not convert environment variable value '{env_value_str}' to type {arg_type}. Using original default. Error: {e}")
    elif 'default' in kwargs:
         logger.trace(f"Argument '{arg_dest_name}' (env: {env_var_name}): No environment variable set or it's empty. Using provided default: '{kwargs['default']}'.")
    else:
         logger.trace(f"Argument '{arg_dest_name}' (env: {env_var_name}): No environment variable set and no default provided.")


    # Remove 'type' if action is store_true/store_false, as they are incompatible
    action = kwargs.get('action')
    if action in ['store_true', 'store_false'] and 'type' in kwargs:
        del kwargs['type']
        # If default was set from env var for a store_true/false action, it might need adjustment
        # For store_true, if env var was "false", default should be False. If "true", default should be True.
        # However, parser.set_defaults handles this correctly if the value is boolean.
        # The 'default' for store_true is False, for store_false is True.
        # If env var forces it, it will be set.
        if 'default' in kwargs and isinstance(kwargs['default'], bool):
             # For store_true, if env var led to False, default becomes False (which is natural)
             # If env var led to True, default becomes True.
             # This means 'default' kwarg will correctly set the initial state before parsing CLI args.
             pass


    parser.add_argument(*args, **kwargs)
    # No need for parser.set_defaults here if 'default' in kwargs is already updated.
    # If an env var was used, it has already modified kwargs['default'].
    # If no env var, the original default (or None) is used.


if __name__ == '__main__':
    # Define arguments using the helper
    add_argument_with_env_fallback('--zotero_id', type=str, help='Zotero user ID', required=False) # Made not required if env var is present
    add_argument_with_env_fallback('--zotero_key', type=str, help='Zotero API key', required=False)
    add_argument_with_env_fallback('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.', default="")
    
    # For boolean flags, action='store_true' or 'store_false' is often better than type=bool
    # The helper function handles env var "true"/"false" for these too.
    add_argument_with_env_fallback('--send_empty', action='store_true', help='If get no arxiv paper, send empty email (default: False)')
    
    add_argument_with_env_fallback('--max_paper_num', type=int, help='Maximum number of papers to recommend (default: 100, -1 for no limit)',default=100)
    add_argument_with_env_fallback('--arxiv_query', type=str, help='Arxiv search query', required=False)
    add_argument_with_env_fallback('--smtp_server', type=str, help='SMTP server', required=False)
    add_argument_with_env_fallback('--smtp_port', type=int, help='SMTP port', default=587) # Common default for TLS
    add_argument_with_env_fallback('--sender', type=str, help='Sender email address', required=False)
    add_argument_with_env_fallback('--receiver', type=str, help='Receiver email address', required=False)
    add_argument_with_env_fallback('--sender_password', type=str, help='Sender email password', required=False)
    
    add_argument_with_env_fallback(
        "--use_llm_api",
        action='store_true', # Changed to action for boolean flag
        help="Use OpenAI API to generate TLDR (default: False)",
    )
    add_argument_with_env_fallback(
        "--openai_api_key",
        type=str,
        help="OpenAI API key (required if --use_llm_api is set)",
        default=None,
    )
    add_argument_with_env_fallback(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument_with_env_fallback(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o", # Changed to a more recent default
    )
    add_argument_with_env_fallback(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    # New arguments for OpenAI timeout and retries
    add_argument_with_env_fallback(
        "--openai_timeout",
        type=float,
        help="Timeout for OpenAI API requests in seconds (default: 60.0)",
        default=60.0
    )
    add_argument_with_env_fallback(
        "--openai_max_retries",
        type=int,
        help="Maximum number of retries for OpenAI API requests (default: 3)",
        default=3 # Default from llm.py
    )

    parser.add_argument('--debug', action='store_true', help='Debug mode (default: False)')
    args = parser.parse_args()

    # Setup logger based on debug flag
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    # Validate required arguments if not provided by env vars or CLI
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
        
        if not papers: # Check if papers list is empty
            logger.info("No new arXiv papers found for the given query.")
            if not args.send_empty:
                logger.info("SEND_EMPTY is false, exiting.")
                exit(0)
            # If send_empty is true, proceed to render and send an empty email
            html_content = render_email([]) # Pass empty list
        else:
            logger.info(f"Retrieved {len(papers)} papers from Arxiv. Reranking...")
            if not corpus:
                 logger.warning("Zotero corpus is empty. Reranking might not be effective or might be skipped.")
                 # Decide if rerank_paper can handle an empty corpus or if you should skip it
            papers = rerank_paper(papers, corpus) # Assuming rerank_paper can handle empty corpus
            
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
                    openai_timeout=args.openai_timeout,         # Pass the timeout
                    openai_max_retries=args.openai_max_retries  # Pass the retries
                )
            else:
                logger.info("Setting up Local LLM as global LLM.")
                # For local LLM, timeout/retries for OpenAI are not applicable
                set_global_llm(lang=args.language) 
            
            logger.info("Rendering email content...")
            html_content = render_email(papers)

        logger.info("Sending email...")
        send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html_content)
        logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

    except Exception as e:
        logger.error(f"An error occurred in the main script: {e}")
        logger.exception("Traceback:") # This will print the full traceback
        sys.exit(1) # Exit with a non-zero code to indicate failure

