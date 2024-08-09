"""
MIT License

© [2024] [Dataworkz]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from evaluate import Evaluate
from utils import extract_response, get_golden_response, write_answers_to_csv, read_openai_key
import argparse, logging, os
    
def evaluate_results(args):
    """
        Evaluate results.

        Args:
            args: Contains all file names to be used for evaluation
    """
    
    response_file = './data/apple10k_collected_response.csv'
    #Pre-processing
    # Extract answers and write to CSV
    question,cand_resp = extract_response(args.dataworkz_response_file, "question :","answer :","links :")
    golden_resp,golden_ctxt = get_golden_response(args.benchmark)

    logging.debug("question:{}, golden response:{}, golden context:{}, candidate response:{}".format(len(question),len(golden_resp),len(golden_ctxt),len(cand_resp)))
    write_answers_to_csv(question, golden_ctxt, golden_resp, cand_resp, response_file)

    logging.debug(f"Intermediate response file saved to {response_file}.")
    logging.info("Extraction Successful.")

    eval = Evaluate()
    eval.evaluate(response_file, args.evaluation_file)
    logging.info("Evaluation completed successfully.")

def _disp_response(llm_response):
    g_claims = llm_response["Golden Response Claims"]
    cand_claims = llm_response["Candidate Response Claims"]
    co_claims = llm_response["Common Claims"]
    g_cnt = llm_response["No of Golden Response Claims"]
    cand_cnt = llm_response["No of Candidate Response Claims"]
    co_cnt = llm_response["No of Common Claims"]
    # this means that the candidate claims cover all the common claims which are part of the 
    # golden claims.
    if co_cnt > cand_cnt:
        cand_cnt = co_cnt

    print("Golden Claims:\n",g_claims)
    print("Candidate Claims:\n",cand_claims)
    print("Common Claims:\n",co_claims)

def evaluate_question(args):
    """
        Evaluate response to a single question.

        Args:
            args: Contains a single question, its corresponding golden answer and candidate answer for evaluation
    """

    recall, precision, f1, response = eval.evaluate_via_llm(args.question, args.golden_response, args.candidate_response)
    _disp_response(response)
    logging.info("Recall:{}, Precision:{}, f1:{}".format(recall, precision,f1))
    

def main():

    benchmark_file = "./data/rag_benchmark_apple_10k_2022_with_context.xlsx"
    response_file = './data/apple10k_dataworkz_qna_response.txt'
    eval_file = './data/apple10k_evaluation_result.csv'


    parser = argparse.ArgumentParser(description="Welcome to Dataworkz Evaluation Framework.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for command c1
    parser_eval = subparsers.add_parser('evaluate', help='Run command evaluate')
    parser_eval.add_argument('--benchmark', type=str, default=benchmark_file, help=f'Benchmark file. (default: {benchmark_file})')
    parser_eval.add_argument('--dataworkz_response_file', type=str, default=response_file, help=f'Response file generated from Dataworkz QnA. (default: {response_file})')
    parser_eval.add_argument('--evaluation_file', type=str, default=eval_file, help=f'Evaluation file. (default: {eval_file})')
    parser_eval.set_defaults(func=evaluate_results)

    parser_eval = subparsers.add_parser('evaluate_question', help='Evaluate a single question')
    parser_eval.add_argument('--question', type=str, required=True, help='Benchmark file.')
    parser_eval.add_argument('--golden_response', type=str, required=True, help='Golden response.')
    parser_eval.add_argument('--candidate_response', type=str, default=eval_file, help='Candidate response for the question.')
    parser_eval.set_defaults(func=evaluate_question)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()
    
def check_openai_api_key():
    api_key = read_openai_key()
    if api_key is None or api_key == "":
        print("The OPENAI_API_KEY is not set in the config file. Please set it to continue.")
        return False
    else:
        os.environ['OPENAI_API_KEY'] = api_key
        print("OPENAI_API_KEY is set.")
        return True

if __name__ == "__main__":
    if not check_openai_api_key():
        exit(1)  # Exit the script if the API key is not set
    main()
