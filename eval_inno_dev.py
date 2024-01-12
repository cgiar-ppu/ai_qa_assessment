import glob
import json
import os
from typing import List

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document

from utils import InnovationProfile, NewBedrock, Readiness

VERBOSE = False

AZURE_OPENAI_API_BASE = "https://so-azure-openai-dev.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
AZURE_DEPLOYMENT_NAME = "gpt35-baseline"
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_TYPE = "azure"

OPENAI_API_KEY = ""


llm_azure = AzureChatOpenAI(
    openai_api_base=AZURE_OPENAI_API_BASE,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_type=AZURE_OPENAI_API_TYPE,
    temperature=0.0,
)

llm_openai = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0
)

llm_bedrock = NewBedrock(model_id="anthropic.claude-v2")

result_output_parser = PydanticOutputParser(pydantic_object=InnovationProfile)
result_format_instructions = result_output_parser.get_format_instructions()

readiness_output_parser = PydanticOutputParser(pydantic_object=Readiness)
readiness_format_instructions = readiness_output_parser.get_format_instructions()

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

extraction_template = (
    "You are a researcher at CGIAR. Your task is to review reports submitted by other researchers and evaluate "
    "how innovative they are across a number of dimensions. A researcher has submitted the following research results "
    "containing key points about the research they conducted. Extract the necessary information from the document provided "
    "and returning this information as a JSON instance.\n\n"
    "{format_instructions}\n\n"
    "The text you are extracting information from can be found below.\n\n"
    "{text}"
)

extraction_prompt = PromptTemplate.from_template(
    template=extraction_template,
    partial_variables={"format_instructions": result_format_instructions},
)

summary_template = (
    "Human:<admin>You are a researcher at CGIAR. Your task is to review projects submitted by other researchers and evaluate "
    "them across a number of dimensions.<admin>\n\nHuman: "
    "This task consist of the following steps:\n"
    "Step 1: Review the project title and its description. Stop and think about it. This will provide you a general understanding of the evidence that will be presented in the next step.\n"
    "Step 2: Review the evidence provided.\n"
    "Step 3: Review the table inside the XML tags <innovation_levels></innovation_levels> This table will be useful in carrying out the next step.\n"
    "Step 4: Generate a list of quotes from the evidence that highlight the important activities that took place as well as the findings made and any results. "
    "Identify if activites carried out are ideations, the development of basic principles, or the validation or testing of well-defined hypotheses. "
    "Also include quotes that characterize the setting for the activities carried out. "
    "Reflect on if the evidence contains well-defined hypotheses that are being validated or tested under fully-controlled conditions, semi-controlled conditions or uncontrolled conditions. "
    "It may be useful to reference the innovation levels (1 through 9) during this step."
    "Finally, make sure to take note of if this innovation is a technological innovation, a capacity development innovation or a policy/organizational/institutional innovation."
    "Write quotes down word for word inside <thinking></thinking> XML tags. This is a space for you to write down relevant content and will not be shown to the user.\n"
    "Step 5: Using the quotes inside <thinking></thinking>, write a 500 word summary in a professional, academic 3rd person voice. "
    "IMPORTANT: It is critical that you distinguish activities which have been carried out from activities which have been planned but not yet carried out."
    "Make sure to recount all major activities that took place within the evidence. "
    "In addition to all major activites, the summary should identify any potential innovations that might result from the activities noted and make sure to explicitly state them. "
    "Finally, close the summary by restating all key findings and next steps discussed.\n"
    "Step 6: Review the summary written in step 5. If necessary re-write it to emphasize conciseness without losing any details. Return the summary without any introduction between the XML tag <summary>.\n"
    "IMPORTANT: Keep in mind that the audience consist of academic researchers. Never refer to yourself in the first person in this summary. "
    "Be sure to read the entire set of instructions carefully before beginning. "
    "Do not go to the next step without making sure the previous step has been completed.\n"
    "<project_title>\n"
    "{short_title}\n"
    "</project_title>\n"
    "<description>\n"
    "{description}\n"
    "</description>\n"
    "<innovation_levels>\n"
    "<row>"
    "<level>0</level>"
    "<title>Idea</title>"
    "<definition>The innovation is in the idea stage. The innovation is not yet being implemented.</definition>"
    "</row>"
    "<row>"
    "<level>1</level>"
    "<title>Basic Research</title>"
    "<definition>The innovation's basic principles are being researched for their ability to achieve an impact.</definition>"
    "</row>"
    "<row>"
    "<level>2</level>"
    "<title>Formulation</title>"
    "<definition>The innovation's basic principles are being formulated or designed.</definition>"
    "</row>"
    "<row>"
    "<level>3</level>"
    "<title>Proof of Concept</title>"
    "<definition>The innovation's key concepts have been validated for their ability to achieve a specific impact.</definition>"
    "</row>"
    "<row>"
    "<level>4</level>"
    "<title>Controlled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under fully-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>5</level>"
    "<title>Model/Early Prototype</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under fully-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>6</level>"
    "<title>Semi-controlled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under semi-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>7</level>"
    "<title>Prototype</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under semi-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<field>8</level>"
    "<title>Uncontrolled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under uncontrolled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>9</level>"
    "<title>Proven Innovation</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under uncontrolled conditions.</definition>"
    "</row>\n"
    "<innovation_levels>\n"
    "<evidence>\n"
    "{text}\n"
    "</evidence>"
    "\n\nAssistant:\n<summary>\n"
)

readiness_template = (
    "You are a researcher at CGIAR. Your task is to review projects submitted by other researchers and evaluate "
    "them across a number of dimensions. This task consist of the following steps:\n"
    "Step 1: Review the summary provided. It summarizes the work carried out as part of the project. Make sure to distinguish activities which were carried out from activities which were only planned.\n"
    "Step 2: Review the table inside the XML tags <innovation_levels></innovation_levels>. It contains a scale of innovation readiness that ranges from 1 to 9.\n"
    "Step 3: Use the innovation readiness scale to determine the cumulative readiness level of COMPLETED activities conducted as part of the project. IMPORTANT: No 'planned, but not-yet-completed' activities should be considered when determining the readiness level.\n"
    "Step 4: In a professional, academic 3rd person voice, concisely justify why you selected this readiness level in at most 300 words.\n"
    "IMPORTANT: Keep in mind that the audience consist of academic researchers. Never refer to yourself in the first person in this summary. "
    "Refer to all actions in the past tense. Be sure to read the entire set of instructions carefully before beginning. "
    "Do not go to the next step without making sure the previous step has been completed.\n\n"
    "Innovation development refers to a new, improved, or adapted output or groups of outputs such as technologies, products and services, policies, "
    "and other organizational and institutional arrangements with high potential to contribute to positive impacts when used at scale. "
    "Innovations may be at early stages of readiness (ideation or basic research) or at more mature stages of readiness (delivery and scaling).\n\n"
    "<innovation_levels>\n"
    "<row>"
    "<level>0</level>"
    "<title>Idea</title>"
    "<definition>The innovation is in the idea stage. The innovation is not yet being implemented.</definition>"
    "</row>"
    "<row>"
    "<level>1</level>"
    "<title>Basic Research</title>"
    "<definition>The innovation's basic principles are being researched for their ability to achieve an impact.</definition>"
    "</row>"
    "<row>"
    "<level>2</level>"
    "<title>Formulation</title>"
    "<definition>The innovation's basic principles are being formulated or designed.</definition>"
    "</row>"
    "<row>"
    "<level>3</level>"
    "<title>Proof of Concept</title>"
    "<definition>The innovation's key concepts have been validated for their ability to achieve a specific impact.</definition>"
    "</row>"
    "<row>"
    "<level>4</level>"
    "<title>Controlled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under fully-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>5</level>"
    "<title>Model/Early Prototype</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under fully-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>6</level>"
    "<title>Semi-controlled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under semi-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>7</level>"
    "<title>Prototype</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under semi-controlled conditions.</definition>"
    "</row>"
    "<row>"
    "<field>8</level>"
    "<title>Uncontrolled Testing</title>"
    "<definition>The innovation is being tested for its ability to achieve a specific impact under uncontrolled conditions.</definition>"
    "</row>"
    "<row>"
    "<level>9</level>"
    "<title>Proven Innovation</title>"
    "<definition>The innovation is validated for its ability to achieve a specific impact under uncontrolled conditions.</definition>"
    "</row>\n"
    "<innovation_levels>\n"
    "<summary>\n"
    "{text}\n"
    "</summary>"
    "{format_instructions}"
    "Only return the resulting JSON object. DO NOT return any other text."
)

root_path = "/home/ubuntu/data/2022"

_folder_paths = [f.path for f in os.scandir(root_path) if f.is_dir()]
result_codes = [os.path.basename(f) for f in _folder_paths]

ID_CODE_RESULT_MAP = {r: None for r in result_codes}
ID_CODE_EVIDENCE_MAP = {r: [] for r in result_codes}

for code in result_codes:
    path = f"{root_path}/{code}/result.pdf"
    if os.path.exists(path):
        ID_CODE_RESULT_MAP[code] = path

for p in _folder_paths:
    base = os.path.basename(p)
    files = [
        f
        for f in (
            glob.glob(os.path.join(p, "**", "*.xlsx"), recursive=True)
            + glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True)
            + glob.glob(os.path.join(p, "**", "*.pptx"), recursive=True)
        )
        if not f.endswith("result.pdf")
    ]
    ID_CODE_EVIDENCE_MAP[base].extend(files)


def load_data(result_id) -> tuple[List[Document], List[Document]]:
    result = PyPDFLoader(ID_CODE_RESULT_MAP.get(result_id)).load_and_split()
    evidence_list = ID_CODE_EVIDENCE_MAP.get(result_id)

    evidence_docs = []
    if evidence_list is not None:
        for path in evidence_list:
            _evidence_docs = PyPDFLoader(path).load_and_split()
            for _doc in _evidence_docs:
                evidence_docs.append(_doc)

    return result, evidence_docs


def get_structured_result(result) -> dict:
    llm_chain = LLMChain(llm=llm_azure, prompt=extraction_prompt, verbose=VERBOSE)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="text",
        verbose=VERBOSE,
    )
    _structured_result = stuff_chain.run({"input_documents": result})
    structured_result = json.loads(_structured_result)

    return structured_result


def get_evidence_summary(evidence_docs, structured_result) -> str:
    short_title = structured_result["short_title"]
    description = structured_result["description"]
    summary_prompt = PromptTemplate.from_template(
        template=summary_template,
        partial_variables={"short_title": short_title, "description": description},
    )
    llm_chain = LLMChain(llm=llm_bedrock, prompt=summary_prompt, verbose=VERBOSE)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="text",
        verbose=VERBOSE,
    )
    evidence_summary = stuff_chain.run({"input_documents": evidence_docs})

    return evidence_summary


def get_readiness_level(evidence_summary) -> str:
    readiness_prompt = PromptTemplate.from_template(
        template=readiness_template,
        partial_variables={
            "format_instructions": readiness_format_instructions,
        },
    )
    llm_chain = LLMChain(llm=llm_openai, prompt=readiness_prompt, verbose=VERBOSE)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="text",
        verbose=VERBOSE,
    )
    _evidence_summary = [
        Document(page_content=evidence_summary, metadata={"source": "Claude v2"})
    ]
    _structured_readiness_eval = stuff_chain.run({"input_documents": _evidence_summary})
    structured_readiness_eval = json.loads(_structured_readiness_eval)
    # structured_readiness_eval = ast.literal_eval(_structured_readiness_eval)

    return structured_readiness_eval


def evaluate_results() -> None:
    for result_id in ID_CODE_RESULT_MAP.keys():
        # if (
        #     result_id
        #     in (
        #     )
        # ):  # 791, 1876, 2097 - Bedrock timeout; 1035 - Bedrock input too long; 2171, 3418, 1103 - .pptx; 856, 2168, 2104, 3018, 2050 - paywall or no evidence
        #     continue
        print(f"Evaluating result {result_id}...")
        print("Loading PRMS result and evidence...")
        result, evidence_docs = load_data(result_id)
        print("Getting structured result...")
        structured_result = get_structured_result(result)
        print("Getting summary of evidence...")
        evidence_summary = get_evidence_summary(evidence_docs, structured_result)
        print("Getting readiness level...")
        structured_readiness_eval = get_readiness_level(evidence_summary)
        output = {
            "result_id": result_id,
            "reported_readiness_level": structured_result["readiness_level"],
            "reported_readiness_justif": structured_result["readiness_justif"],
            "ai_readiness_level": structured_readiness_eval["readiness_level"],
            "ai_readiness_justif": structured_readiness_eval["readiness_level_summary"],
        }

        with open(
            "/home/ubuntu/readiness_09_01_24.jsonl", "a"
        ) as f:  # updates to .jsonl
            json.dump(output, f)
            f.write("\n")

        print(output)

    return output


if __name__ == "__main__":
    result = evaluate_results()
