import argparse
import os
import json
from ingest import ingest_file
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--txt_path', type=str, help='Path to the TXT file')
    parser.add_argument('--media_path', type=str, help='Path to the media file')

    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model to use')
    parser.add_argument('--llm-max-concurrency', type=int, default=5,
                      help='Maximum number of concurrent LLM requests in async TOC pipeline')

    parser.add_argument('--toc-check-pages', type=int, default=20, 
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--toc-page-range', type=str, default=None,
                      help='TOC page range in PDF, e.g. 3-8. If provided, TOC is extracted directly from this range.')
    parser.add_argument('--max-pages-per-node', type=int, default=10,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=20000,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default='yes',
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='yes',
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default='yes',
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default='yes',
                      help='Whether to add text to the node')
    parser.add_argument('--if-add-page-labels', type=str, default='no',
                      help='Whether to mark page labels in node text (PDF only)')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    parser.add_argument('--txt-chunk-chars', type=int, default=12000,
                      help='Character count per chunk for TOC extraction from TXT (txt only)')
    args = parser.parse_args()
    
    # Validate that exactly one file type is specified
    selected_input_count = sum(bool(x) for x in [args.pdf_path, args.md_path, args.txt_path, args.media_path])
    if selected_input_count == 0:
        raise ValueError("One of --pdf_path or --md_path or --txt_path or --media_path must be specified")
    if selected_input_count > 1:
        raise ValueError("Only one of --pdf_path or --md_path or --txt_path or --media_path can be specified")

    config_loader = ConfigLoader()

    # Build common options from YAML defaults + CLI overrides.
    user_opt = {
        'model': args.model,
        'llm_max_concurrency': args.llm_max_concurrency,
        'toc_check_page_num': args.toc_check_pages,
        'toc_page_range': args.toc_page_range,
        'max_page_num_each_node': args.max_pages_per_node,
        'max_token_num_each_node': args.max_tokens_per_node,
        'if_add_node_id': args.if_add_node_id,
        'if_add_node_summary': args.if_add_node_summary,
        'if_add_doc_description': args.if_add_doc_description,
        'if_add_node_text': args.if_add_node_text,
        'if_add_page_labels': args.if_add_page_labels,
    }
    opt = config_loader.load(user_opt)
    output_dir = getattr(opt, 'output_dir', './files') or './files'
    
    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process the PDF
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')
        
    elif args.media_path:
        args.media_path = os.path.abspath(args.media_path)
        # Process media file
        print('Processing media file...')
        text_path = ingest_file(args.media_path)
        user_opt = {
            'audio_json_path': text_path,
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        audio_json_to_tree(**user_opt)
        

            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')

    elif args.txt_path:
        # Validate TXT file
        if not args.txt_path.lower().endswith('.txt'):
            raise ValueError("TXT file must have .txt extension")
        if not os.path.isfile(args.txt_path):
            raise ValueError(f"TXT file not found: {args.txt_path}")

        print('Processing txt file...')
        import asyncio

        toc_with_page_number = asyncio.run(txt_to_tree(
            txt_path=args.txt_path,
            chunk_chars=args.txt_chunk_chars,
            llm_max_concurrency=args.llm_max_concurrency,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))

        print('Parsing done, saving to file...')

        txt_name = os.path.splitext(os.path.basename(args.txt_path))[0]
        output_file = f'{output_dir}/{txt_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f'Tree structure saved to: {output_file}')