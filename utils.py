import json
from typing import Any, Iterator, List

import boto3
from langchain.llms.bedrock import Bedrock, LLMInputOutputAdapter
from langchain.schema.output import GenerationChunk
from pydantic.v1 import BaseModel, Field

session = boto3.Session(region_name="us-east-1")
boto3_bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")
boto3_bedrock_stream = boto3.client(
    service_name="bedrock-runtime", region_name="us-east-1"
)


class Description(BaseModel):
    description: str = Field(
        title="Description", description="A brief description of the innovation"
    )


class ImpactAreasResults(BaseModel):
    gender_tag: str = Field(
        title="Gender tag",
        description='The gender tag of the research results. Choices are: "Principal", "Significant", "Not Targeted", "Not provided". Select only the choice that is relevant given the description.',
    )
    climate_change_tag: str = Field(
        title="Climate Change tag",
        description='The climate change tag of the research results. Choices are: "Principal", "Significant", "Not Targeted", "Not provided". Select only the choice that is relevant given the description.',
    )
    nutrition_tag: str = Field(
        title="Nutrition tag",
        description='The nutrition tag of the research results. Choices are: "Principal", "Significant", "Not Targeted", "Not provided". Select only the choice that is relevant given the description.',
    )
    environment_tag: str = Field(
        title="Environment and/or biodiversity tag",
        description='The environment tag of the research results. Choices are: "Principal", "Significant", "Not Targeted", "Not provided". Select only the choice that is relevant given the description.',
    )
    poverty_tag: str = Field(
        title="Poverty tag",
        description='The poverty tag of the research results. Choices are: "Principal", "Significant", "Not Targeted", "Not provided". Select only the choice that is relevant given the description.',
    )


class ImpactAreasOutput(BaseModel):
    gender_tag: str = Field(
        title="Gender tag",
        description="The gender tag of the research results.",
    )
    climate_change_tag: str = Field(
        title="Climate Change tag",
        description="The climate change tag of the research results.",
    )
    nutrition_tag: str = Field(
        title="Nutrition tag",
        description="The nutrition tag of the research results.",
    )
    environment_tag: str = Field(
        title="Environment and/or biodiversity tag",
        description="The environment tag of the research results.",
    )
    poverty_tag: str = Field(
        title="Poverty tag",
        description="The poverty tag of the research results.",
    )


class ActionAreas(BaseModel):
    action_areas: List[str] = Field(
        title="Action Areas",
        description='The action areas relevant to the research results. Choices are: "Systems Transformation", "Resilient Agrifood Systems", "Genetic Innovation". Select only the choices that are relevant given the description.',
    )


class SDGs(BaseModel):
    sdgs: List[str] = Field(
        title="Sustainable Development Goals",
        description='The relevant UN 2030 Sustainable Development Goals. Choices are: "Goal 1: No Poverty", "Goal 2: Zero Hunger", "Goal 3: Good Health and Well-being", "Goal 4: Quality Education", "Goal 5: Gender Equality", "Goal 6: Clean Water and Sanitation", "Goal 7: Affordable and Clean Energy", "Goal 8: Decent Work and Economic Growth", "Goal 9: Industry, Innovation and Infrastructure", "Goal 10: Reduced Inequalities", "Goal 11: Sustainable Cities and Communities", "Goal 12: Responsible Consumption and Production", "Goal 13: Climate Action", "Goal 14: Life Below Water", "Goal 15: Life On Land", "Goal 16: Peace, Justice and Strong Institutions", "Goal 17: Partnerships for the Goals". Select only the choices that are relevant given the description.',
    )


class TheoryOfChange(BaseModel):
    name: str = Field(
        title="Name",
        description="The name of the initiative which begins with the prefix 'INIT-'.",
    )


class GeographicLocationResults(BaseModel):
    geographic_focus: str = Field(
        title="Geographic Focus",
        description='The geographic focus of the innovation. Choices are: "Global", "Regional", "National", "Sub-national", "Not provided". Select only the choice that is relevant given the description.',
    )
    region: List[str] = Field(
        title="Region",
        description='The region(s) where the innovation is being implemented. Choices are: "Africa", "Asia", "Europe", "Latin America and the Caribbean", "North America", "Oceania", "Not provided". Select only the choices that is relevant given the description.',
    )
    country: List[str] = Field(
        title="Country",
        description="The country(ies) that the innovation targets. Choices are any of the countries in the world or 'Not provided'. Select only the choices that is relevant given the description.",
    )


class GeographicLocationOutput(BaseModel):
    geographic_focus: str = Field(
        title="Geographic Focus",
        description="The geographic focus of the project.",
    )
    region: str = Field(
        title="Region",
        description="The region where the project is being implemented.",
    )
    country: str = Field(
        title="Country",
        description="The country where the project took place.",
    )


class Readiness(BaseModel):
    readiness_level: str = Field(
        title="Readiness Level", description="The readiness level selected."
    )
    readiness_level_summary: str = Field(
        title="Readiness Level Summary",
        description="The summary which justifies the readiness level selected.",
    )


class InnovationProfile(BaseModel):
    description: str = Field(
        title="Description", description="A brief description of the innovation"
    )
    long_title: str = Field(
        title="Long Title",
        description="The long title of the innovation. Otherwise return 'Not Provided.'",
    )
    short_title: str = Field(
        title="Short Title",
        description="The short title of the innovation. Otherwise return 'Not Provided.'",
    )
    innovation_character: str = Field(
        title="Innovation Characterization",
        description="The characterization that best categorizes the degree of innovation. Choices are: 'Incremental innovation', 'Radical innovation', 'Disruptive innovation'. Select only the choice that is relevant given the description.",
    )
    innovation_typology: str = Field(
        title="Innovation Topology",
        description="The typology that best characertizes the innovation. Choices are: 'Technological innovation', 'Capacity development innovation', 'Policy, organizational or institutional innovation'. Select only the choice that is relevant given the description.",
    )
    readiness_level: str = Field(
        title="Readiness Level",
        description="The readiness level of the innovation. Choices are: 'Level 1 - Basic Research', 'Level 2 - Formulation', 'Level 3 - Proof of Concept', 'Level 4 - Controlled Testing Demonstration', 'Level 5 - Model/Early Prototype', 'Level 6 - Semi-controlled Testing', 'Level 7 - Prototype', 'Level 8 - Uncontrolled Testing', 'Level 9 - Proven Innovation'. Select only the choice that is relevant given the description.",
    )
    readiness_justif: str = Field(
        title="Readiness Justification",
        description="A brief explanation of how the provided evidence justifies the readiness level of the innovation.",
    )


class ImpactAreas(BaseModel):
    project_title: str = Field(
        title="Project Title", description="The title of the project"
    )
    description: Description
    geographic_location: GeographicLocationResults
    impact_areas: ImpactAreasResults


class ImpactJustifications(BaseModel):
    gender_tag_just: str = Field(
        title="Gender tag justification",
        description="A single sentence justification for why the gender tag label was selected.",
    )
    climate_change_tag_just: str = Field(
        title="Climate Change tag justification",
        description="A single sentence justification for why the climate change tag label was selected.",
    )
    nutrition_tag_just: str = Field(
        title="Nutrition tag justification",
        description="A single sentence justification for why the nutrition tag label was selected.",
    )
    environment_tag_just: str = Field(
        title="Environment and/or biodiversity tag justification",
        description="A single sentence justification for why the environment tag label was selected.",
    )
    poverty_tag_just: str = Field(
        title="Poverty tag justification",
        description="A single sentence justification for why the poverty tag label was selected.",
    )


class GeoLocTags(BaseModel):
    geographic_location: GeographicLocationOutput


class ImpactAreaTags(BaseModel):
    geographic_location: GeographicLocationOutput
    impact_areas: ImpactAreasOutput
    impact_justifications: ImpactJustifications


class NewBedrock(Bedrock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_input_and_invoke_stream(
        self, prompt, stop=None, run_manager=None, **kwargs
    ) -> Iterator[GenerationChunk]:
        _model_kwargs = {
            "prompt": Any,
            "max_tokens_to_sample": 8192,
            "temperature": 0.0,
        }
        provider = "anthropic"
        model_kwargs = {**_model_kwargs, **kwargs}
        input_body = LLMInputOutputAdapter.prepare_input(provider, prompt, model_kwargs)
        body = json.dumps(input_body)
        try:
            response = boto3_bedrock_stream.invoke_model(
                body=body,
                modelId="anthropic.claude-v2",
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        for chunk in LLMInputOutputAdapter.prepare_output_stream(
            provider, response, stop
        ):
            yield chunk

    def _prepare_input_and_invoke(self, prompt, stop=None, run_manager=None, **kwargs):
        _model_kwargs = {
            "prompt": Any,
            "max_tokens_to_sample": 8192,
            "temperature": 0.2,
        }
        provider = "anthropic"
        model_kwargs = {**_model_kwargs, **kwargs}
        input_body = LLMInputOutputAdapter.prepare_input(provider, prompt, model_kwargs)
        body = json.dumps(input_body)
        try:
            response = boto3_bedrock_stream.invoke_model(
                body=body,
                modelId="anthropic.claude-v2",
                accept="application/json",
                contentType="application/json",
            )
            text = LLMInputOutputAdapter.prepare_output(provider, response)

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        return text
