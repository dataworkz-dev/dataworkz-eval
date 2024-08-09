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

import csv, logging
import pandas as pd
import os, openai
import json

def read_openai_key():
    file_path = "./config/config.json"
    key = "OPENAI_API_KEY"
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if key in data:
            return data[key]
        else:
            print(f"The key '{key}' is not present in the JSON file.")
            return None
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
        return None

def get_openai_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=messages,
        temperature=0,
    )
    logging.debug("\nOpenAI Response:\n", response)
    return response.choices[0].message.content


def extract_response(file_path,qtag, atag, ntag):
    qtag = qtag.lower()
    atag = atag.lower()
    ntag = ntag.lower()

    questions = []
    answers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        qcapture = False
        acapture = False
        qlines = []
        alines = []
        
        for line in lines:
            line = line.strip()
            logging.debug("Line:\n",line)
            
            if line.lower().startswith(qtag):
                qlines.append(line[len(qtag):].strip())
                qcapture = True
            elif line.lower().startswith(atag):
                logging.debug("Question added:", " ".join(qlines).strip())
                questions.append(" ".join(qlines).strip())
                qlines = []
                alines.append(line[len(atag):].strip())
                qcapture = False
                acapture = True
            elif line.lower().startswith(ntag):
                logging.debug("Answer added:"," ".join(alines).strip())
                answers.append(" ".join(alines).strip())
                alines = []
                qcapture = False
                acapture = False
            elif qcapture == True:
                qlines.append(line)
            elif acapture == True:
                alines.append(line)

    return questions, answers
            
def write_answers_to_csv(question, golden_ctxt, golden_resp, cand_resp, response_file):
    with open(response_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SNo.","Question", "Golden Context", "Golden Response", "Candidate Response"])  # Write header
        sno = 0
        for q,gc,gr,cr in zip(question,golden_ctxt,golden_resp, cand_resp):
            sno += 1
            writer.writerow([sno,q,gc,gr,cr])  # Write each answer in a new row

def get_golden_response(golden_file):
    # Read the Excel file
    df = pd.read_excel(golden_file)

    # Retrieve the data from the specified column
    golden_response = df['Golden Response']
    golden_context = df['Golden Context']
    
    return golden_response, golden_context
