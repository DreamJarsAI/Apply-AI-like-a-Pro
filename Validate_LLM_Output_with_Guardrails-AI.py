# Medium article: https://ai.plainenglish.io/ensuring-accuracy-a-guide-to-validating-large-language-model-outputs-b24ea780aff7
# GuardrailAI document: https://www.guardrailsai.com/docs/guardrails_ai/getting_started


# Import modules
import os
from dotenv import load_dotenv
import openai
from guardrails import Guard
# In the terminal, run: guardrails hub install hub://guardrails/valid_range
# In the terminal, run: guardrails hub install hub://guardrails/valid_choices
from guardrails.hub import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List


# A doctor's note
doctors_notes = """49 y/o Male with chronic macular rash to face & hair, worse in beard, eyebrows & nares.
Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream"""


# Use Pydantic to define a schema
class Symptoms(BaseModel):
    symptom: str = Field(..., description="Symptom that a patient is experiencing")
    affected_area: str = Field(
        ..., 
        description="What part of the body the symptom is affecting", 
        # Note: if the LLM fails to output a valid choice, the validator will ask the LLM to regenerate a response
        validators=[ValidChoices(["head", "neck", "chest"], on_fail="reask")]
    )

class CurrentMeds(BaseModel):
    medication: str = Field(..., description="Name of the medication the patient is taking")
    response: str = Field(..., description="How the patient is responding to the medication")

class PatientInfo(BaseModel):
    gender: str = Field(..., description="Patient's gender")
    age: int = Field(..., description="Patient's age", validators=[ValidRange(0, 100)])
    # Note: symptoms will be a list of Symptoms objects
    symptoms: List[Symptoms] = Field(..., description="Symptoms that the patient is experiencing")
    # Note: current_meds will be a list of CurrentMeds objects
    current_meds: List[CurrentMeds] = Field(..., description="Medications that the patient is currently taking")


# Set the OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Draft a prompt to guide the data extraction process
# Note: ${gr.complete_json_suffix_v2} guides the GuardrailsAI to output a valid JSON
prompt = f"""Given the following doctor's notes about a patient,
please extract a dictionary that contains the patient's information:

{doctors_notes}

@complete_json_suffix_v2
"""


# Initialize a Guard object from the PatientInfo Pydantic model
guard = Guard.from_pydantic(output_class=PatientInfo, prompt=prompt)


# Wrap an LLM call with the Guard object
validated_output, *rest = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4",
)


# Print the validated output
print(validated_output)