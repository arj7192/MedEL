import json
from concurrent.futures import ThreadPoolExecutor

import PyPDF2
import gradio as gr
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def initialize_llm_model(model_path: str = "mistral-7b-openorca.Q4_0.gguf") -> LlamaCpp:
    llm = LlamaCpp(model_path=model_path,
                   n_gpu_layers=1,
                   n_batch=512,
                   n_ctx=2048,
                   f16_kv=True,
                   callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                   seed=10)
    return llm


# create a global LLM model
llm = initialize_llm_model()

# can be increased as per available resources
executor = ThreadPoolExecutor(max_workers=1)


def extract_pdf_info(pdf_file: gr.File) -> dict:
    # Open the PDF file
    with open(pdf_file.name, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        number_of_pages = len(pdf_reader.pages)
        text = ""
        for i in range(number_of_pages):
            page = pdf_reader.pages[i]
            text += page.extract_text()

    # Create a JSON object containing PDF information
    pdf_info = {
        "text": text,
        "num_pages": number_of_pages
    }

    return pdf_info


def final_score(output_payload: list) -> float:
    answers = []
    for output in output_payload:
        # we need a try catch to cover for cases where LLM hallucinates non JSON output
        try:
            answers += [json.loads(output.get("answer", "{}"))]
        except json.decoder.JSONDecodeError as e:
            continue
    answer_scores = [(int(answer.get("confidence", 0)) / 10) * (answer.get("decision", "").lower() == "yes") for answer
                     in answers]
    final_score = sum(answer_scores) / len(answer_scores) if len(answer_scores) else 0.
    return final_score * 100.


def prepare_report(input_text: str) -> (dict, dict, float, str):
    prompt_2 = PromptTemplate(
        input_variables=["question"],
        template="""{input_text}\n\n{question}\n\n Answer in a sentence or as a list""")
    llm_chain_2 = LLMChain(prompt=prompt_2, llm=llm)
    questions_2 = ["What is the patient’s chief complaint?",
                   "What treatment plan is the doctor suggesting?",
                   "A list of allergies the patient has",
                   "A list of medications the patient is taking, with any known side-effects", ]

    answers_2 = [llm_chain_2.run({"input_text": input_text, "question": question}) for question in questions_2]
    json_output_2 = [{"question": question, "answer": answer} for (question, answer) in zip(questions_2, answers_2)]
    prompt_3 = PromptTemplate(
        input_variables=["question"],
        template="""{input_text}\n\n{question}\n\nAnswer precisely in json format with following 
        fields - decision (string with value yes or no), confidence (integer between 0 and 10), 
        and justification (one sentence)""")
    llm_chain_3 = LLMChain(prompt=prompt_3, llm=llm)
    questions_3 = ["Does the patient have a family history of colon cancer in their first-degree relatives?",
                   "Has the patient experienced minimal bright red blood per rectum?",
                   "Has the patient had significant loss of blood?",
                   "Does the patient have a history of skin problems?",
                   "Has the patient used hydrocortisone cream for the haemorrhoids "
                   "that they are currently experiencing?",
                   "Were any high risk traits found on the patient’s genetic test?",
                   "Has the patient had a colonoscopy in the last 5 years?",
                   "Has the patient had any recent foreign travel?",
                   "How long has the patient been known to healthcare services?", ]
    answers_3 = [executor.submit(llm_chain_3.run, {"input_text": input_text, "question": question}) for question in
                 questions_3]
    json_output_3 = [{"question": question, "answer": answer.result()} for (question, answer) in
                     zip(questions_3, answers_3)]
    overall_score = final_score(json_output_3)
    with open("output.json", "w") as f:
        json.dump({"general_info": json_output_2, "specific_info": json_output_3, "overall_score": overall_score}, f)
    return json_output_2, json_output_3, overall_score, "output.json"


def app(pdf_file: gr.File) -> (dict, dict, float, str):
    pdf_info = extract_pdf_info(pdf_file)
    return prepare_report(pdf_info.get("text"))


iface = gr.Interface(
    fn=app,
    inputs="file",
    outputs=[
        gr.JSON(label="General patient information"),
        gr.JSON(label="Answers to specific questions"),
        gr.Number(label="Treatment plan appropriateness score (%)"),
        gr.File(label="Download JSON", file_types=[".json"])
    ],
    title="MedEL: Medical document data Extraction using LLM",
    description="Upload a PDF medical report to extract information as JSON using a simple LLM-powered pipeline."
)

iface.launch(server_name="0.0.0.0", server_port=8082)
