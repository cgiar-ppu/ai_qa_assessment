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

from utils import ImpactAreas, ImpactAreaTags, NewBedrock

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

llm_openai_35 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k", temperature=0
)

llm_bedrock = NewBedrock(model_id="anthropic.claude-v2")

result_output_parser = PydanticOutputParser(pydantic_object=ImpactAreas)
result_format_instructions = result_output_parser.get_format_instructions()

geo_loc_ia_tags_output_parser = PydanticOutputParser(pydantic_object=ImpactAreaTags)
geo_loc_ia_tags_format_instructions = (
    geo_loc_ia_tags_output_parser.get_format_instructions()
)

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
    "Step 3: Review the texts inside the <geographic_focus></geographic_focus> and <topics_of_interest></topics_of_interest> XML tags. These texts will be useful in carrying out the next step.\n"
    "Step 4: Generate a list of quotes from the evidence that highlight the important activities that took place as well as the findings made and any results. "
    "Also include quotes that characterize the geographic focus of the aforementioned activities and any topics of interest that are explicitly referenced in the evidence. "
    "Write them down word for word inside <thinking></thinking> XML tags. This is a space for you to write down relevant content and will not be shown to the user.\n"
    "Step 5: Using the quotes inside <thinking></thinking>, write a 750 word summary in a professional, academic 3rd person voice. "
    "Make sure to note all major activities that took place within the evidence. "
    "In addition to all major activites, the summary should identify the geographic focus of these activities and make sure to explicitly state "
    "any of the topics of interest if and only if they are referenced in the evidence. Both geographic focus and topics of interest are clearly defined below."
    "Make sure to justify why all particular topics of interest are identified. Finally, close the summary by restating all key findings.\n"
    "Step 6: Review the summary written in step 5. If necessary re-write it to emphasize conciseness without losing any details. Return the summary without any introduction between the XML tag <summary>.\n"
    "IMPORTANT: Keep in mind that the audience consist of academic researchers. Never refer to yourself in the first person in this summary. "
    "Refer to all actions in the past tense. Be sure to read the entire set of instructions carefully before beginning. "
    "Do not go to the next step without making sure the previous step has been completed.\n"
    "<project_title>\n"
    "{project_title}\n"
    "</project_title>\n"
    "<description>\n"
    "{description}\n"
    "</description>\n"
    "<geographic_focus>\n"
    "'Geographic focus' consist of determining if the major activities are 'Global', 'Regional', 'National', or 'Sub-national'. "
    "'Regional' refers to whether the activities focus on a United Nations geoscheme subregion. "
    "'National' refers to whether the activities focus on a country. "
    "'Sub-national' refers to whether the activites focus on a WHO subnational region without a country. "
    "The geographic focus is the largest category that is explicitly mentioned in the context of an activity carried out in the evidence.\n"
    "</geographic_focus>\n"
    "<topics_of_interest>\n"
    "'Topics of interest' are topics that should be highlighted in the summary if they are discussed because the impact of activites on these topics "
    "are especially important. These topics are nutrition, health, food security, poverty reduction, livelihood, jobs, gender equality, youth inlcusion, social inclusion, "
    "climate adaptation, climate mitigation, environmental health, and biodiversity.\n"
    "</topics_of_interest>\n"
    "<evidence>\n"
    "{text}\n"
    "</evidence>"
    "\n\nAssistant:\n<summary>\n"
)

GEO_LOC_LABELS = """
The labels for Geographic focus are: 'Global', 'Regional', 'National', 'Sub-national'.
The labels for Region are all the United Nations geoscheme subregions".
The labels for Country are any of the countries in the world.
"""

IA_OBJECTIVES = """
Below is a lookup table of CGIAR strategic objectives and their definitions.

| Impact Area | Impact Area Abbreviation | Objective |
| Gender equality, youth and social inclusion | Gender | To close the gender gap in rights to economic resources, access to ownership and control over land and natural resources for over 500 million women who work in food, land and water systems. |
| Gender equality, youth and social inclusion | Gender | To offer rewardable opportunities to 267 million young people who are not in employment, education or training. |
| Climate adaptation and mitigation | Climate | Turn agriculture and forest systems into a net sink for carbon by 2050 (Climate Mitigation target). |
| Climate adaptation and mitigation | Climate | Equip 500 million small-scale producers to be more resilient by 2030 to climate shocks, with climate adaptation solutions available through national innovation systems (Climate Adaptation target). |
| Climate adaptation and mitigation | Climate | Support countries in implementing National Adaptation Plans and Nationally Determined Contributions, and increased ambition in climate actions by 2030 (Climate Policy target). This support could possibly be in the form of education or training. |
| Nutrition, health and food security | Nutrition | To end hunger for all and enable affordable, healthy diets for the 3 billion people who do not currently have access to safe and nutritious food. |
| Nutrition, health and food security | Nutrition | To reduce cases of foodborne illness (600 million annually) and zoonotic disease (1 billion annually) by one third. |
| Environmental health and biodiversity | Environment | Stay within planetary and regional environmental boundaries: 1. consumptive water use in food production of less than 2,500 km3 per year (with a focus on the most stressed basins) 2. zero net deforestation 3. nitrogen application of 90 Tg per year (with a redistribution towards low-input farming systems) and increased use efficiency; and phosphorus application of 10 Tg per year. |
| Environmental health and biodiversity | Environment | Maintain the genetic diversity of seed varieties, cultivated plants and farmed and domesticated animals and their related wild species, including through soundly managed genebanks at the national, regional, and international levels. |
| Environmental health and biodiversity | Environment | In addition, water conservation and management, restoration of degraded lands/soils, restoration of biodiversity in situ, and management of pollution related to food systems are key areas of environmental impacts to which the CGIAR should contribute. |
| Environmental health and biodiversity | Environment | All of the 2030 targets that are organized as part of the 4 long-term goals for 2050 included in the Kunming-Montreal Global Biodiversity Framework. |
| Environmental health and biodiversity | Environment | All of the targets and indicators included in UN Sustainable Development Goals 2, 6, 12, 14, 15 and 17 |
| Poverty reduction, livelihoods and jobs | Poverty | Lift at least 500 million people living in rural areas above the extreme poverty line of US $1.90 per day (2011 PPP). |
| Poverty reduction, livelihoods and jobs | Poverty | Reduce by at least half the proportion of men, women and children of all ages living in poverty in all its dimensions, according to national definitions. |
"""

IA_LABELS = """
The Impact Area labels are defined as follows:

| Tag | Objective |
| Principal | The research activity principally addressses one of the objectives for the impact area. The impact area is the main objective of the research activity and fundamental to its design and expected results. The research activity would not have been undertaken without consideration of this impact area. |
| Significant | The research activity contributes in significant ways to the impact area, even though it is not the principal focus of the activity. The impact area is an important and deliberate objective of the research activity but not the main reason for its undertaking. |
| Not Targeted | The research activity has not been found to target any aspect of the impact area. |
"""

geo_loc_ia_tags_template = (
    "You are a researcher at CGIAR. Your task is to review projects submitted by other researchers and determine "
    "the appropriate label for these projects for 8 dimensions."
    "Three of these dimensions are 'Geographic focus', 'Region' and 'Country'. They are collectively referred to as 'Geographic Location'.\n"
    "The remaining 5 dimensions are 'Gender equality, youth and social inclusion', 'Climate adaptation and mitigation', "
    "'Nutrition, health and food security', 'Environmental health and biodiversity' and 'Poverty reduction, livelihoods and jobs'. "
    "These dimensions are collectively referred to as 'Impact Areas'.\n"
    "Another researcher has already prepared a summary of the project for you to use as a reference for selecting the appropriate label for each dimension.\n\n"
    "The steps for the first task are as follows:\n"
    "This task consist of two sub-tasks. The first is to determine the Geographical Location labels. The second sub-task is to determine the Impact Area labels.\n"
    "Step 1: Review the sections labeled Project Title, Description, and Geographic Location Labels. Geographic Location Labels provides a list of the labels for each label.\n"
    "Step 2: Review the summary provided and return the labels that corresponds with it for 'Geographic focus', 'Region' and 'Country'.\n"
    "The steps for the second task are as follows:\n"
    "Step 1: Review the section labeled Impact Area Objectives. It contains a table of the impact areas and the objectives that comprise each of them.\n"
    "Step 2: Review the section labeled Impact Area Labels. It provides definitions of the labels for labeling the Impact Areas.\n"
    "IMPORTANT: Keep in mind that the audience consist of academic researchers. Never refer to yourself in the first person in this summary. "
    "Refer to all actions in the past tense. Be sure to read the entire set of instructions carefully before beginning. "
    "Do not go to the next step without making sure the previous step has been completed.\n\n"
    "Geographic Location Labels:\n {geo_loc_labels}\n\n"
    "Impact Area Objectives:\n {ia_objectives}\n\n"
    "Impact Area Labels:\n {ia_labels}\n\n"
    "Summary:\n{text}\n\n"
    "{format_instructions}"
    "Only return the resulting JSON object. DO NOT return any other text."
)


root_path = "/home/ubuntu/data/2023"

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
    llm_chain = LLMChain(
        llm=llm_openai_35, prompt=extraction_prompt, verbose=VERBOSE
    )  # previously - llm=llm_azure
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
    project_title = structured_result["project_title"]
    description = structured_result["description"]["description"]
    summary_prompt = PromptTemplate.from_template(
        template=summary_template,
        partial_variables={"project_title": project_title, "description": description},
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


def get_geo_loc_ia_tags(evidence_summary) -> str:
    geo_loc_ia_tags_prompt = PromptTemplate.from_template(
        template=geo_loc_ia_tags_template,
        partial_variables={
            "geo_loc_labels": GEO_LOC_LABELS,
            "ia_objectives": IA_OBJECTIVES,
            "ia_labels": IA_LABELS,
            "format_instructions": geo_loc_ia_tags_format_instructions,
        },
    )
    llm_chain = LLMChain(llm=llm_openai, prompt=geo_loc_ia_tags_prompt, verbose=VERBOSE)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="text",
        verbose=VERBOSE,
    )
    _evidence_summary = [
        Document(page_content=evidence_summary, metadata={"source": "Claude v2"})
    ]
    _structured_geo_loc_ia_tags = stuff_chain.run(
        {"input_documents": _evidence_summary}
    )
    structured_geo_loc_ia_tags = json.loads(_structured_geo_loc_ia_tags)

    return structured_geo_loc_ia_tags


def evaluate_results() -> None:
    for result_id in ID_CODE_RESULT_MAP.keys():
        print(f"Evaluating result {result_id}...")
        print("Loading PRMS result and evidence...")
        result, evidence_docs = load_data(result_id)
        print("Getting structured result...")
        structured_result = get_structured_result(result)
        print("Getting summary of evidence...")
        evidence_summary = get_evidence_summary(evidence_docs, structured_result)
        print("Getting geographic location and impact area tags...")
        structured_geo_loc_ia_tags = get_geo_loc_ia_tags(evidence_summary)

        output = {
            "result_id": result_id,
            "reported_geographic_focus": structured_result["geographic_location"][
                "geographic_focus"
            ],
            "reported_region": structured_result["geographic_location"]["region"],
            "reported_country": structured_result["geographic_location"]["country"],
            "reported_gender_tag": structured_result["impact_areas"]["gender_tag"],
            "reported_climate_tag": structured_result["impact_areas"][
                "climate_change_tag"
            ],
            "reported_nutrition_tag": structured_result["impact_areas"][
                "nutrition_tag"
            ],
            "reported_environment_tag": structured_result["impact_areas"][
                "environment_tag"
            ],
            "reported_poverty_tag": structured_result["impact_areas"]["poverty_tag"],
            "ai_geographic_focus": structured_geo_loc_ia_tags["geographic_location"][
                "geographic_focus"
            ],
            "ai_region": structured_geo_loc_ia_tags["geographic_location"]["region"],
            "ai_country": structured_geo_loc_ia_tags["geographic_location"]["country"],
            "ai_gender_tag": structured_geo_loc_ia_tags["impact_areas"]["gender_tag"],
            "ai_climate_tag": structured_geo_loc_ia_tags["impact_areas"][
                "climate_change_tag"
            ],
            "ai_nutrition_tag": structured_geo_loc_ia_tags["impact_areas"][
                "nutrition_tag"
            ],
            "ai_environment_tag": structured_geo_loc_ia_tags["impact_areas"][
                "environment_tag"
            ],
            "ai_poverty_tag": structured_geo_loc_ia_tags["impact_areas"]["poverty_tag"],
            "ai_gender_tag_just": structured_geo_loc_ia_tags["impact_justifications"][
                "gender_tag_just"
            ],
            "ai_climate_tag_just": structured_geo_loc_ia_tags["impact_justifications"][
                "climate_change_tag_just"
            ],
            "ai_nutrition_tag_just": structured_geo_loc_ia_tags[
                "impact_justifications"
            ]["nutrition_tag_just"],
            "ai_environment_tag_just": structured_geo_loc_ia_tags[
                "impact_justifications"
            ]["environment_tag_just"],
            "ai_poverty_tag_just": structured_geo_loc_ia_tags["impact_justifications"][
                "poverty_tag_just"
            ],
        }

        with open("/home/ubuntu/geo_loc_ia_tags_09_01_24.jsonl", "a") as f:
            json.dump(output, f)
            f.write("\n")

        print(output)

    return output


if __name__ == "__main__":
    result = evaluate_results()
