
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

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction
import time, json
from utils import get_openai_response
import logging, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Evaluate:
    
    def evaluate(self, response_file, eval_file):
        """
        Evaluate response to a single question.

        Args:
            response_file: temporary csv file generated with collated information from the benchmark file and the generated response file
            eval_file: csv file which would contain the final output of the evaluation result.
        """
        df = pd.read_csv(response_file)

        results = {
            'bs': [],
            'r1': [],
            'rL': [],
            'bert_p': [],
            'bert_r': [],
            'bert_f1': [],
            'sim': [],
            'llm_recall': [],
            'llm_precision': [],
            'llm_f1': [],
            'g_cnt':[],
            'cand_cnt':[],
            'co_cnt':[],
            'g_claims':[],
            'cand_claims':[],
            'co_claims':[]
        }

        for index, row in df.iterrows():
            print("Response:",row["SNo."])

            question = row["Question"]
            golden_resp = row["Golden Response"]
            cand_resp = row["Candidate Response"]

            bs, r1, rL, bert_p, bert_r, bert_f1 = self.__evaluate_metrics(golden_resp,cand_resp)
            sim = self.__evaluate_similarity([golden_resp],[cand_resp])
            results['bs'].append(bs)
            results['r1'].append(r1)
            results['rL'].append(rL)
            results['bert_p'].append(bert_p)
            results['bert_r'].append(bert_r)
            results['bert_f1'].append(bert_f1)
            results['sim'].append(sim)

            llm_recall, llm_precision, llm_f1, llm_response = self.evaluate_via_llm(question, golden_resp, cand_resp)
            results['llm_recall'].append(llm_recall)
            results['llm_precision'].append(llm_precision)
            results['llm_f1'].append(llm_f1)
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
            results['g_claims'].append(g_claims)
            results['cand_claims'].append(cand_claims)
            results['co_claims'].append(co_claims)
            results['g_cnt'].append(g_cnt)
            results['cand_cnt'].append(cand_cnt)
            results['co_cnt'].append(co_cnt)                        
            
        df['Bleu Score'] = results['bs']
        df['Rouge-1'] = results['r1']
        df['Rouge-L'] = results['rL']
        df['Bert Precision'] = results['bert_p']
        df['Bert Recall'] = results['bert_r']
        df['Bert Score F1'] = results['bert_f1']
        df['Similarity Score'] = results['sim']
        df['LLM Recall'] = results['llm_recall']
        df['LLM Precision'] = results['llm_precision']
        df['LLM F1'] = results['llm_f1']
        df['Golden Response Claim Count'] = results['g_cnt']
        df['Candidate Response Claim Count'] = results['cand_cnt']
        df['Common Claim Count'] = results['co_cnt']
        df['Golden Response Claims'] = results['g_claims']
        df['Candidate Response Claims'] = results['cand_claims']
        df['Common Claims'] = results['co_claims']
        
        df.to_csv(eval_file, index=False)
    
    def __evaluate_similarity(self, reference_sentence,candidate_sentence):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Compute embeddings for both lists
        embeddings1 = model.encode(reference_sentence)
        embeddings2 = model.encode(candidate_sentence)
        # Compute cosine similarities
        similarities = model.similarity(embeddings1, embeddings2)
        return similarities[0][0].item()

    def __evaluate_metrics(self, reference, candidate):
        
        # Calculate BLEU score
        ref_bleu = reference.split()
        cand_bleu = candidate.split()
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu([ref_bleu], cand_bleu, smoothing_function=chencherry.method1)
        logging.debug(f'BLEU score: {bleu_score:.4f}')

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        logging.debug(f'ROUGE-1 score: {scores["rouge1"].fmeasure:.4f}')
        logging.debug(f'ROUGE-L score: {scores["rougeL"].fmeasure:.4f}')

        # Calculate BERT score
        P, R, F1 = bert_score.score([candidate], [reference], lang="en", rescale_with_baseline=True)
        logging.debug(f'BERT Precision: {P.mean().item():.4f}')
        logging.debug(f'BERT Recall: {R.mean().item():.4f}')
        logging.debug(f'BERT F1: {F1.mean().item():.4f}')

        return bleu_score, scores["rouge1"].fmeasure, scores["rougeL"].fmeasure, P.item(), R.item(), F1.item()
    
    def evaluate_via_llm(self, question, golden_response, candidate_response):
        """
        Evaluate response to a single question.

        Args:
            response_file: temporary csv file generated with collated information from the benchmark file and the generated response file
            eval_file: csv file which would contain the final output of the evaluation result.
        """
        prompt_template5 = """
            Given the following question:

            ###  Start Question:
            {}
            End Question

            and a golden response and a candidate response respectively. 

            ### Start Golden Response:
            {}
            End Golden Response

            ### Start Candidate Response:
            {}
            End Candidate Response

            ### Evaluate the two responses using the Evaluation Method below. 
            The responses could be numerical, specific (e.g., names or dates), or descriptive.

            ### Evaluation Method:
            1. Create a list of individual claims that can be inferred from the golden response with respect to the question.
            2. Create a list of individual claims that can be inferred from the candidate response with respect to the question.
            3. Calculate the total number of claims of the golden response present in the candidate response based on the following rules:
                - the complete statement of each claim in golden response should be checked against the complete statement of each claim in candidate response. 
                - If the golden response claim is specific in nature like numerical, names or dates then the candidate response claim should contains the exact value present in the golden response.
            
            ### For creating the individual claims follow the following instructions:
             - Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
             - Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
             - For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
             - Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.

            ### After creating the list, perform the following:
            1. In the golden response claims, if any claim can be directly inferred from the question only then remove it from the list.
            2. In the candidate repsonse claims, if any claim can be directly inferred from the question, only then remove it from the list.
            
            ### The final output should contain the explanation of the evaluation method and the numerical value in the following json format:
            
            {{
                Golden Response Claims: {{ <list of claims from the golden response> }}
                Candidate Response Claims: {{ <list of claims from the candidate response> }}
                Common Claims: {{ <list of claims from golden response present in candidate > }}
                No of Golden Response Claims: <value>
                No of Candidate Response Claims: <value>
                No of Common Claims: <value>
            }}
            
            ### Example:
            {{
                "Golden Response Claims": {{ 
                                                "1": The Mac line includes laptops.,
                                                "2": The laptops mentioned are MacBook Air and MacBook Pro.,
                                                "3": The Mac line includes desktops.,
                                                "4": The desktops mentioned are iMac, Mac mini, Mac Studio, and Mac Pro.
                                            }},
                "Candidate Response Claims":   {{
                                                "1": The company's line of personal computers is called Mac.,
                                                "2": It includes laptops.,
                                                "3": The laptops included are MacBook Air and MacBook Pro.,
                                                "4": It includes desktops.,
                                                "5": The desktops included are iMac, Mac mini, Mac Studio, and Mac Pro.,
                                                }},
                "No of Golden Response Claims": 4,
                "No of Candidate Response Claims": 5,
                "No of Common Claims": 4
            }}

            ### Please strictly adhere to the json format specified above. please provide the complete response
            in json format.

            """

        prompt = prompt_template5.format(question, golden_response, candidate_response)
        logging.debug("Prompt:\n",prompt)
        response = get_openai_response(prompt)
        json_response = None
        try:
            logging.debug("\nResponse before extract:\n",response)
            response = self.__extract_json(response)
            logging.debug("\nResponse after extract:\n",response)
            json_response = json.loads(response)
            logging.debug(json_response)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print("Raw output:", response)
            return

        if json_response is None or json_response == '':
            print("Empty response\n")
            exit(1)
        
        logging.debug("Response:\n", response)

        golden_cnt = json_response["No of Golden Response Claims"]
        candidate_cnt = json_response["No of Candidate Response Claims"]
        common_cnt = json_response["No of Common Claims"]
        # this means that the cnadidate claims cover all the common claims which are part of the 
        # golden claims.
        if common_cnt > candidate_cnt:
            candidate_cnt = common_cnt
        recall, precision, f1 = self.__calculate_llm_metrics(golden_cnt, candidate_cnt, common_cnt)
        
        time.sleep(1)

        return recall, precision, f1, json_response

    def __calculate_llm_metrics(self, golden_cnt, candidate_cnt, common_cnt):
        recall = common_cnt/golden_cnt
        if candidate_cnt == 0:
            precision = 0
        else:
            precision = common_cnt/candidate_cnt
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = (2*recall*precision)/(precision+recall)
        return recall, precision, f1

    def __extract_json(self, response):
            start_tag = '```json'
            end_tag = '```'
            start_tag_idx = response.find(start_tag)
            end_tag_idx = response.rfind(end_tag)
            if start_tag_idx != -1 and end_tag_idx != -1:
                return response[start_tag_idx+len(start_tag):end_tag_idx].strip()
            else:
                return response


prompt_template4 = """
            Given the following question:

            ###  Start Question:
            {}
            End Question

            and a golden response and a candidate response respectively. 

            ### Start Golden Response:
            {}
            End Golden Response

            ### Start Candidate Response:
            {}
            End Candidate Response

            ### Evaluate the two responses using the Evaluation Method below. 
            The responses could be numerical, specific (e.g., names or dates), or descriptive.

            ### Evaluation Method:
            1. Calculate the total number of claims of the golden response with respect to the question.
            2. Calculate the total number of claims of the candidate response with respect to the question..
            3. Calculate the total number of claims of the golden response present in the candidate response based on the following rules:
                - the complete statement of each claim in golden response should be checked against the complete statement of each claim in candidate response. 
                - If the golden response claim is specific in nature like numerical, names or dates then the candidate response claim should contains the exact value present in the golden response.
            4. If there are additional claims in the golden response, can they be inferred from the candidate response? If yes then include them in the common claims and in candidate claims.
            5. If there are additional claims in the candidate response, can they be inferred from the golden response? If yes then include them in the common claims and in golden response claims. 
            
            ### The final output should contain the explanation of the evaluation method and the numerical value in the following json format:
            
            {{
                Golden Response Claims: {{ <list of claims from the golden response> }}
                Candidate Response Claims: {{ <list of claims from the candidate response> }}
                Common Claims: {{ <list of claims from golden response present in candidate > }}
                No of Golden Response Claims: <value>
                No of Candidate Response Claims: <value>
                No of Common Claims: <value>
            }}
            
            ### Example:
            {{
                "Golden Response Claims": {{ 
                                                "1": The Mac line includes laptops.,
                                                "2": The laptops mentioned are MacBook Air and MacBook Pro.,
                                                "3": The Mac line includes desktops.,
                                                "4": The desktops mentioned are iMac, Mac mini, Mac Studio, and Mac Pro.
                                            }},
                "Candidate Response Claims":   {{
                                                "1": The company's line of personal computers is called Mac.,
                                                "2": It includes laptops.,
                                                "3": The laptops included are MacBook Air and MacBook Pro.,
                                                "4": It includes desktops.,
                                                "5": The desktops included are iMac, Mac mini, Mac Studio, and Mac Pro.,
                                                }},
                "No of Golden Response Claims": 4,
                "No of Candidate Response Claims": 5,
                "No of Common Claims": 4
            }}

            ### Please strictly adhere to the json format specified above. please provide the complete response
            in json format.

            """

prompt_template1 = """
        following is the question:
            
        question: {}
                
        and given the golden response,

        golden response: {}

        and the candidate response,

        candidate response: {}

        # first identify whether the golden response contains numerical results.
        # if it is asking for specific information then compare the "candidate response" to the "golden response" and determine whether the "candidate response" contains all the specific information present in "golden response". 
        # if the question is asking for descriptive or generative answer then see whether the   

        Break the golden response and the candidate response into two sets of separate points 
        only if the answer is comprehensive.
        Use similar langugage so that when we run evaluation metrics like bertScore or semantic
        similarity then two similar sentences should give us higher score.

        Provide the points under two separate tags: "golden_response_points" and "candidate_response_points".
    """

prompt_template2 = """
        following is the question:
            
        question: {}
        
        You are given a golden response and a candidate response. Your task is to compare the two responses 
        based on accuracy, completeness, and relevance. The responses could be numerical, 
        specific (e.g., names or dates), or descriptive.

        ### Golden Response:
        {}

        ### Candidate Response:
        {}

        ### Comparison Criteria:
        1. **Accuracy**: Does the candidate response provide the correct information?
        2. **Completeness**: Does the candidate response cover all aspects of the golden response?
        3. **Relevance**: Is the information in the candidate response relevant to the question?

        ### Evaluation Metrics:
        1. **Recall**: Number of claims of the golden reponse present in the candidate response/ Total number of claims in golden response  
        2. **Precision**: Number of claims of the golden reponse present in the candidate response/ Total number of claims in candidate response
        3. **F1 Score**: Harmonic mean of Recall and Precision calculated above.

        ### Evaluation:
        Provide a detailed comparison of the two responses based on the criteria above. 
        Indicate any missing or incorrect information in the candidate response. 
        If the candidate response provides additional relevant information, mention that as well.
        """

prompt_template3 = """
        You are given a golden response and a candidate response. Your task is to evaluate the two responses 
        using the Evaluation Method below and then calculate the metrics using the evaluation metrics set below. 
        The responses could be numerical, specific (e.g., names or dates), or descriptive.

        ### Golden Response:
        {}

        ### Candidate Response:
        {}

        ### Evaluation Method:
        1. Calculate the total number of claims of the golden response.
        2. Calculate the total number of claims of the candidate response.
        3. Calculate the total number of claims of the golden response present in the candidate response.

        ### Evaluation Metrics:
        1. **Recall**: Number of claims of the golden response present in the candidate response/ Total number of claims in golden response  
        2. **Precision**: Number of claims of the golden reponse present in the candidate response/ Total number of claims in candidate response
        3. **F1 Score**: Harmonic mean of Recall and Precision calculated above.

        The final output should only contain the evaluation metrics in the following format:
        Recall:
        Precision:
        F1:
        """
