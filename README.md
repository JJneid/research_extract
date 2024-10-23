# PDF Data Extractor

A Streamlit application that extracts structured information from academic PDFs using various LLM providers (OpenAI, Anthropic Claude, or Meta LLama). The app allows users to customize extraction prompts, manage categories, and export results to Excel.

## Features

### Multiple LLM Providers
- OpenAI (GPT-3.5, GPT-4)
- Anthropic Claude (Claude 3 Haiku, Claude 3 Sonnet)
- Meta LLama (Meta-Llama-3.1-8B-Instruct)

### Prompt Management
- Organize prompts into customizable categories
- Add, edit, and delete categories
- Add, edit, and delete prompts within categories
- Specify exact format requirements for each prompt

### Data Extraction
- Process multiple PDFs in batch
- Extract information based on customized prompts
- Format validation for extracted data
- Progress tracking during extraction

### Export
- View results in an interactive table
- Export results to Excel
- Each PDF gets its own row with columns matching prompts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-data-extractor.git
cd pdf-data-extractor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit pandas PyPDF2 anthropic openai openpyxl
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Configure the application:
   - Select your preferred LLM provider
   - Enter API key if required (not needed for Llama)
   - Customize categories and prompts if needed

4. Upload PDFs and extract data:
   - Click "Upload PDF Files" to select one or more PDFs
   - Click "Extract Data" to begin processing
   - Monitor progress in the progress bar
   - View results in the interactive table
   - Download results as Excel file

## Default Categories and Prompts

The application comes with three default categories:

### Study Characteristics
- First author last name
- Publication year
- Journal
- Country of corresponding author
- Funding source
- Author financial conflicts of interest

### Participants
- Main eligibility criteria
- Country(ies) of participants
- N included
- N (%) females/women

### Trial Arms
- Trial arm name
- Group description

## Customization

### Adding New Categories
1. Click "Add New Category"
2. Enter category name
3. Click "Save Category"

### Adding New Prompts
1. Navigate to desired category
2. Click "Add New Prompt"
3. Fill in:
   - Title: Column name in results
   - Prompt: Instructions for the LLM
   - Format: Expected format of the response
4. Click "Save Prompt"

### Editing Categories/Prompts
- Use "Edit" buttons to modify existing categories
- Use expanders to modify existing prompts
- Click "Update" to save changes

## API Keys

### OpenAI
- Obtain API key from: https://platform.openai.com/api-keys
- Required for GPT-3.5 and GPT-4 models

### Anthropic
- Obtain API key from: https://console.anthropic.com/
- Required for Claude models

### Llama
- No API key required
- Uses provided endpoint: http://3.15.181.146:8000/v1/

## Configuration

The application uses several configuration dictionaries that can be modified in the code:

### Models Configuration
```python
MODELS = {
    "OpenAI": {
        "name": "OpenAI GPT-3.5",
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "requires_key": True,
        "base_url": None
    },
    # ... other providers
}
```

### Default Prompts
```python
DEFAULT_PROMPTS = {
    "Study characteristics": [
        {
            "title": "First author last name",
            "prompt": "State the last name of first author only...",
            "format": "Text with first letter capitalized"
        },
        # ... other prompts
    ],
    # ... other categories
}
```

## Error Handling

The application includes error handling for:
- Invalid API keys
- Failed API calls
- PDF processing errors
- Duplicate category names
- Missing required fields

## Limitations

- PDF text extraction quality depends on the PDF format
- Maximum context length varies by model
- Processing time increases with document length
- Session state is not persistent between restarts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses OpenAI, Anthropic, and Meta Llama APIs
- PDF processing with PyPDF2
- Excel export with openpyxl
