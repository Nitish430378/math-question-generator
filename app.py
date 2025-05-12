
import streamlit as st
import json
import re
import random
from typing import Dict, List, Optional, Set
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Nitish035/gguf_4",
        filename="unsloth.Q4_K_M.gguf",
        local_dir=".",
    )
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=35,
        verbose=False,
    )
    return model

model = load_model()

def get_difficulty(dok):
    difficulty_map = {
        1: "Easy",
        2: "Medium",
        3: "Hard",
        4: "Very Hard"
    }
    return difficulty_map.get(dok, "Medium")

def generate_response(prompt_dict, max_tokens=3024):
    dok_descriptions = {
        1: "Basic recall of math facts, definitions, or simple calculations",
        2: "Application of math concepts requiring 1-2 step procedures",
        3: "Multi-step math problems requiring analysis and justification",
        4: "Complex math problems needing synthesis and creative approaches"
    }
    
    difficulty_levels = {
        "Easy": "Multi-step problems requiring reasoning and understanding of concepts - easy; Straightforward problems with obvious approaches.",
        "Medium": "Multi-step problems requiring moderate reasoning and understanding of concepts - medium; Requires careful analysis but standard methods apply.",
        "Hard": "Complex multi-step problems with multiple variables and operations - hard; Demands innovative thinking and multiple concepts.",
        "Very Hard": "Advanced problems requiring systems of equations, conditional logic, and optimization - very hard; Requires advanced reasoning, optimization strategies, and integration of multiple topics."
    }

    contexts = [
        "a school fundraiser",
        "a community bake sale",
        "a sports team's snack stand",
        "a charity event",
        "a classroom project",
        "",
        "",
        ""
    ]

    operations = [
        "addition and subtraction",
        "multiplication and division",
        "fractions and percentages",
        "ratios and proportions",
        "algebraic equations",
        "",
        ""
    ]

    prompt = f"""<|im_start|>user 
Generate a {prompt_dict['Difficulty']} difficulty math multiple-choice question with options and correct answer with these specifications:

* Grade Level: {prompt_dict['Grade']}
* Topic: {prompt_dict['Topic']} (align with appropriate CCSS standard)
* Depth of Knowledge (DOK): Level {prompt_dict['DOK']} ({dok_descriptions[prompt_dict['DOK']]})
* Difficulty: {prompt_dict['Difficulty']} ({difficulty_levels[prompt_dict['Difficulty']]})
* Context: {random.choice(contexts)}
* Math Operations: {random.choice(operations)}

1. Create a unique word problem based on the context and operations
2. Design a question that matches DOK level {prompt_dict['DOK']}
3. Create four plausible options with one clearly correct answer
4. Format as a clean multiple-choice question

# Requirements:

1. The question must be unique and different from previous questions
2. Make sure the final answer computed in the explanation is inserted into one of the 4 options
3. The `correct_answer` key must match the option letter that holds the correct answer
4. Options should reflect common student misconceptions
5. Format the response as a JSON object with these keys: 'question', 'options', 'correct_answer', 'explanation'

<|im_end|>
<|im_start|>assistant
"""

    response = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        echo=False
    )
    
    return response['choices'][0]['text']

def try_parse_json(response_text):
    try:
        match = re.search(r'{.*}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}")
        return None

def main():
    st.title("DOK-Based Math Question Generator")
    st.write("Generate math questions based on a Depth of Knowledge level")

    dok_level = st.selectbox("Select DOK Level:", [1, 2, 3, 4])
    no_of_questions = st.slider("Number of Questions:", 1, 2)
    topic = st.selectbox("Select Topic:", [
        "Functions",
        "Statistics and Probability",
        "Geometry",
        "Expressions and Equations",
        "Number System",
        "Ratios and Proportional"
    ])
    grade_level = st.selectbox("Select Grade Level:", [6, 7, 8])

    if st.button("Generate Questions"):
        difficulty = get_difficulty(dok_level)
        all_questions = []
        generated_questions = set()

        for i in range(no_of_questions):
            attempts = 0
            while attempts < 3:
                prompt_dict = {
                    "Grade": str(grade_level),
                    "Topic": topic,
                    "DOK": dok_level,
                    "Difficulty": difficulty
                }
                
                response_text = generate_response(prompt_dict)
                parsed_json = try_parse_json(response_text)
                
                if parsed_json and parsed_json.get('question') and parsed_json['question'] not in generated_questions:
                    generated_questions.add(parsed_json['question'])
                    all_questions.append(parsed_json)
                    break
                attempts += 1

        st.subheader("Generated Questions:")
        if all_questions:
            st.json(all_questions)
        else:
            st.error("Failed to generate unique questions. Please try again.")

if __name__ == "__main__":
    main()