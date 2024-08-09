# Dataworkz Evaluation Framework
We introduce a state-of-the-art evaluation method (DEF) using LLM-as-a-Judge that captures the essence of a generated answer when compared to a golden response and also provides a measurable metric. We run our evaluation metrics across a publicly available finance dataset, [Apple 10-K filing for the fiscal year 2022](https://investor.apple.com/sec-filings/default.asp). We leveraged [FinQABench](https://huggingface.co/datasets/lighthouzai/finqabench), a QA benchmark specifically designed for financial applications, which is based on Apple’s 2022 10-K filing. FinQABench comprises 100 questions related to the Apple 10-K 2022 filing, along with golden answers and corresponding golden context. We tested all the FinQABench questions against the Dataworkz (DW) RAG system and evaluated the generated DW answers by comparing them to the FinQABench golden answers.

The result is available at: [Apple 10k evaluation result](https://github.com/dataworkz-dev/dataworkz-eval/blob/8f4ad7ec0a9227998aaf3c2550cd51c893aacd00/data/apple10k_evaluation_result.csv)

## Prompt used for LLM-as-a-Judge
To build our prompt we adapt the prompt template present in Chen et. al's paper [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648).

```
@misc{chen2023dense,
      title={Dense X Retrieval: What Retrieval Granularity Should We Use?},
      author={Tong Chen and Hongwei Wang and Sihao Chen and Wenhao Yu and Kaixin Ma and Xinran Zhao and Hongming Zhang and Dong Yu},
      year={2023},
      eprint={2312.06648},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Our final prompt template for generating metrics:

```python
prompt =    """
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
```

## Setup
1. Configure your OpenAI API key in config/config.json
2. Install package dependencies by running the following command from terminal:
	> pip install -r requirements.txt

## Usage
Following files are essential to run the evaluation:
   - Benchmark File: Benchmark tests available publicly.
   - Response File: Response generated by Dataworkz QnA.
   - Evaluation File: Final evaluation results would be stored in this file.
	
Note: The Benchmark File and the Response File should contain equal number of rows and each row 
should correspond to the response of the same question.

