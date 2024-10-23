import streamlit as st
import pandas as pd
import PyPDF2
import io
from anthropic import Anthropic
from openai import OpenAI
import tempfile
import os

# Model configurations
MODELS = {
    "OpenAI": {
        "name": "OpenAI GPT-3.5",
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "requires_key": True,
        "base_url": None
    },
    "Anthropic": {
        "name": "Anthropic Claude",
        "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        "requires_key": True,
        "base_url": None
    },
    "Llama": {
        "name": "Meta Llama",
        "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
        "requires_key": False,
        "base_url": "http://3.15.181.146:8000/v1/"
    }
}

# Define default prompts
DEFAULT_PROMPTS = {
    "Study characteristics": [
        {
            "title": "First author last name",
            "prompt": "State the last name of first author only, formatted with first letter capitalized.",
            "format": "Text with first letter capitalized"
        },
        {
            "title": "Publication year",
            "prompt": "State the publication year (4 digits only), formatted 'Publication Year: XXXX'. Use 'Online ahead of print' year if not in final format in journal issue with page numbers.",
            "format": "Publication Year: XXXX"
        },
        {
            "title": "Journal",
            "prompt": "State the full journal name without explanation, formatted 'Journal: Journal Name'",
            "format": "Journal: Journal Name"
        },
        {
            "title": "Country of corresponding author",
            "prompt": "State the country of corresponding author (if there is more than one corresponding author, prioritize the one that is first author; if none are first, then prioritize the last author), response formatted 'Country of Corresponding Author: Country Name'",
            "format": "Country of Corresponding Author: Country Name"
        },
        {
            "title": "Funding source",
            "prompt": "State the article funding source from one of these choices without explanation: Industry / Non-industry / Combined industry and non-industry / No funding / Not reported; formatted 'Funding source: source'",
            "format": "Funding source: source"
        },
        {
            "title": "Author financial conflicts of interest",
            "prompt": "State whether there is a conflict of interest without explanation. If yes, indicate which author(s). If no conflict of interest, response formatted 'No'; if there is conflict of interest, response formatted '[Author last name]; [conflict]'.",
            "format": "No or [Author last name]; [conflict]"
        }
    ],
    "Participants": [
        {
            "title": "Main eligibility criteria",
            "prompt": "State the study eligibility criteria. Separate by a semi-column and list each criterion on a separate row. Only include key criteria and avoid long lists of minor, less important criteria. Response should be 30 words or less.",
            "format": "Criterion 1; Criterion 2; etc."
        },
        {
            "title": "Country(ies) of participants",
            "prompt": "State the country(es) of participants (all listed), response formatted 'Countries of participants: Country Name 1; Country Name 2; etc.'",
            "format": "Countries of participants: Country Name 1; Country Name 2; etc."
        },
        {
            "title": "N included",
            "prompt": "Indicate the number of participants included in the trial (use number randomized if provided and the larger number of any other designation otherwise). Response formatted 'N = Number'",
            "format": "N = Number"
        },
        {
            "title": "N (%) females/women",
            "prompt": "Indicate the number and percentage of female or women participants. Response formatted 'Females/women: Number (percentage)'",
            "format": "Females/women: Number (percentage)"
        }
    ],
    "Trial arms": [
        {
            "title": "Trial arm name",
            "prompt": "Provide trial arm names (e.g., Pill placebo, sham, usual care, waitlist, physical therapy, cognitive behaviour therapy). Response formatted 'Trial arm name 1 vs. Trial arm name 2 vs. Trial arm name 3 etc.'",
            "format": "Trial arm name 1 vs. Trial arm name 2 vs. Trial arm name 3"
        },
        {
            "title": "Group description",
            "prompt": "Provide a description of the groups. Summarize succinctly. Response should be 60 words or less. Response formatted 'Group 1 description: Description; Group 2 description: Description; etc.'",
            "format": "Group 1 description: Description; Group 2 description: Description"
        }
    ]
}

def initialize_session_state():
    """Initialize all session state variables."""
    if 'prompts_dict' not in st.session_state:
        st.session_state.prompts_dict = DEFAULT_PROMPTS.copy()
    if 'categories' not in st.session_state:
        st.session_state.categories = list(DEFAULT_PROMPTS.keys())
    if 'show_add_category' not in st.session_state:
        st.session_state.show_add_category = False
    if 'show_add_prompt' not in st.session_state:
        st.session_state.show_add_prompt = {}
        for category in st.session_state.categories:
            st.session_state.show_add_prompt[category] = False
    if 'editing_category' not in st.session_state:
        st.session_state.editing_category = {}
        for category in st.session_state.categories:
            st.session_state.editing_category[category] = False

def get_client(provider):
    """Initialize and return the appropriate client based on the selected provider."""
    if provider == "OpenAI":
        return OpenAI(api_key=st.session_state.api_keys['openai'])
    elif provider == "Anthropic":
        return Anthropic(api_key=st.session_state.api_keys['anthropic'])
    elif provider == "Llama":
        return OpenAI(base_url="http://3.15.181.146:8000/v1/", api_key='None')
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def extract_info_from_pdf(pdf_file, prompts_dict, provider, model):
    """Extract information from a PDF file using the selected model."""
    # Read PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file.seek(0)
        
        pdf_reader = PyPDF2.PdfReader(tmp_file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    os.unlink(tmp_file.name)
    
    # Flatten prompts from all categories
    all_prompts = []
    for category, prompts in prompts_dict.items():
        all_prompts.extend(prompts)
    
    # Get appropriate client
    client = get_client(provider)
    
    # Prepare the extraction prompt
    system_message = """You are a research assistant analyzing academic papers. Your task is to extract specific information following exact formatting requirements. 
    If information is not found, respond with 'Not found in paper'.
    Be precise and ensure responses match the specified formats exactly.
    Provide each answer on a new line starting with the number and a period."""
    
    extraction_prompt = f"""Here is the paper content to analyze: {text[:15000]}...

Please extract the following information, following the exact format specified for each:

{chr(10).join([f'{i+1}. {prompt["prompt"]} Format: {prompt["format"]}' for i, prompt in enumerate(all_prompts)])}"""

    try:
        if provider == "Anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                system=system_message,
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ]
            )
            response_text = response.content[0].text
        else:  # OpenAI and Llama use the same API format
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": extraction_prompt}
                ]
            )
            response_text = response.choices[0].message.content
        
        # Extract answers from the response
        answers = response_text.split('\n')
        answers = [ans.split('. ', 1)[1] if '. ' in ans else ans 
                  for ans in answers if ans.strip() and not ans.startswith('Here')]
        
        return answers
    
    except Exception as e:
        st.error(f"Error with {provider} API: {str(e)}")
        return None

def manage_categories():
    """Handle category management UI and logic."""
    st.subheader("Manage Categories")
    
    # Add new category
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Add New Category"):
            st.session_state.show_add_category = True
    
    if st.session_state.show_add_category:
        with st.form(key="add_category_form"):
            new_category = st.text_input("New Category Name")
            submitted = st.form_submit_button("Save Category")
            if submitted and new_category:
                if new_category not in st.session_state.categories:
                    st.session_state.categories.append(new_category)
                    st.session_state.prompts_dict[new_category] = []
                    st.session_state.show_add_prompt[new_category] = False
                    st.session_state.editing_category[new_category] = False
                    st.session_state.show_add_category = False
                    st.success(f"Category '{new_category}' added!")
                    st.rerun()
                else:
                    st.error("Category already exists!")
    
    # Edit existing categories
    st.markdown("### Edit Categories")
    for category in st.session_state.categories:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if st.session_state.editing_category.get(category, False):
                new_name = st.text_input(f"Edit category name", value=category, key=f"edit_{category}")
                if new_name != category and new_name in st.session_state.categories:
                    st.error("Category name already exists!")
                elif new_name != category:
                    prompts = st.session_state.prompts_dict.pop(category)
                    st.session_state.prompts_dict[new_name] = prompts
                    idx = st.session_state.categories.index(category)
                    st.session_state.categories[idx] = new_name
                    st.session_state.editing_category[new_name] = False
                    st.session_state.editing_category.pop(category)
                    st.session_state.show_add_prompt[new_name] = st.session_state.show_add_prompt.pop(category)
                    st.success(f"Category renamed to '{new_name}'!")
                    st.rerun()
            else:
                st.markdown(f"**{category}**")
        
        with col2:
            if not st.session_state.editing_category.get(category, False):
                if st.button("Edit", key=f"edit_btn_{category}"):
                    st.session_state.editing_category[category] = True
                    st.rerun()
            else:
                if st.button("Save", key=f"save_btn_{category}"):
                    st.session_state.editing_category[category] = False
                    st.rerun()
        
        with col3:
            if st.button("Delete", key=f"delete_btn_{category}"):
                if len(st.session_state.categories) > 1:
                    st.session_state.categories.remove(category)
                    st.session_state.prompts_dict.pop(category)
                    st.session_state.show_add_prompt.pop(category)
                    st.session_state.editing_category.pop(category)
                    st.success(f"Category '{category}' deleted!")
                    st.rerun()
                else:
                    st.error("Cannot delete the last category!")

def main():
    st.title("PDF Data Extractor")
    
    # Initialize session state
    initialize_session_state()
    
    # Model selection
    st.subheader("Model Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox(
            "Select Provider",
            options=list(MODELS.keys()),
            key="provider"
        )
    
    with col2:
        model = st.selectbox(
            "Select Model",
            options=MODELS[provider]["models"],
            key="model"
        )
    
    # Initialize API keys in session state if not present
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'openai': '',
            'anthropic': ''
        }
    
    # API key input if required
    if MODELS[provider]["requires_key"]:
        if provider == "OpenAI":
            api_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                key="openai_key_input",
                value=st.session_state.api_keys['openai']
            )
            st.session_state.api_keys['openai'] = api_key
        elif provider == "Anthropic":
            api_key = st.text_input(
                "Enter your Anthropic API key:",
                type="password",
                key="anthropic_key_input",
                value=st.session_state.api_keys['anthropic']
            )
            st.session_state.api_keys['anthropic'] = api_key
        
        if not api_key:
            st.warning(f"Please enter your {provider} API key to proceed.")
            return
    
    # Category Management
    manage_categories()
    
    # Prompt Management
    st.subheader("Manage Prompts")
    
    # Display and edit prompts by category
    for category in st.session_state.categories:
        st.markdown(f"### {category}")
        
        # Add prompt button for category
        if st.button(f"Add New Prompt to {category}", key=f"add_prompt_btn_{category}"):
            st.session_state.show_add_prompt[category] = True
        
        # Add new prompt form
        if st.session_state.show_add_prompt[category]:
            with st.form(key=f"add_prompt_form_{category}"):
                title = st.text_input("Title", key=f"new_prompt_title_{category}")
                prompt = st.text_area("Prompt", key=f"new_prompt_text_{category}")
                format_str = st.text_input("Format", key=f"new_prompt_format_{category}")
                
                submitted = st.form_submit_button("Save Prompt")
                if submitted and title and prompt and format_str:
                    st.session_state.prompts_dict[category].append({
                        "title": title,
                        "prompt": prompt,
                        "format": format_str
                    })
                    st.session_state.show_add_prompt[category] = False
                    st.success(f"Prompt '{title}' added to {category}!")
                    st.rerun()
        

        # Display existing prompts
        for i, prompt in enumerate(st.session_state.prompts_dict[category]):
            with st.expander(f"{prompt['title']}"):
                # Edit fields
                new_title = st.text_input("Title", value=prompt["title"], key=f"title_{category}_{i}")
                new_prompt = st.text_area("Prompt", value=prompt["prompt"], key=f"prompt_{category}_{i}")
                new_format = st.text_input("Format", value=prompt["format"], key=f"format_{category}_{i}")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Update prompt
                    if st.button("Update", key=f"update_{category}_{i}"):
                        st.session_state.prompts_dict[category][i] = {
                            "title": new_title,
                            "prompt": new_prompt,
                            "format": new_format
                        }
                        st.success("Prompt updated!")
                
                with col2:
                    # Delete prompt
                    if st.button("Delete", key=f"delete_{category}_{i}"):
                        st.session_state.prompts_dict[category].pop(i)
                        st.success(f"Prompt '{prompt['title']}' deleted!")
                        st.rerun()
    # File uploader
    st.subheader("Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Extract Data"):
        # Process files and create DataFrame
        data = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            progress_text = st.empty()
            progress_text.text(f"Processing {file.name}...")
            
            try:
                answers = extract_info_from_pdf(
                    file, 
                    st.session_state.prompts_dict,
                    provider,
                    model
                )
                
                if answers:
                    file_data = {"Filename": file.name}
                    
                    # Flatten prompts for data assignment
                    all_prompts = []
                    for category_prompts in st.session_state.prompts_dict.values():
                        all_prompts.extend(category_prompts)
                    
                    for prompt, answer in zip(all_prompts, answers):
                        file_data[prompt["title"]] = answer
                    data.append(file_data)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            progress_text.empty()
        
        if data:
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Display the results
            st.subheader("Extracted Data")
            st.dataframe(df)
            
            # Download button
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            
            st.download_button(
                label="Download Excel file",
                data=excel_buffer,
                file_name="extracted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()