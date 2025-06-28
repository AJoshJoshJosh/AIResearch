import csv
import os
import asyncio
import logging
import aiohttp
import json
import time
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================
# CONSTANTS
# ======================
class APIConstants:
    PERPLEXITY_MODEL = "llama-3.1-sonar-small-128k-online"
    OPENAI_MODEL = "gpt-4o-mini"
    RATE_LIMIT_STATUS = 429
    PERPLEXITY_MAX_TOKENS = 2000
    OPENAI_MAX_TOKENS = 3000
    REQUEST_TIMEOUT = 30
    SESSION_TIMEOUT = 60
    CONNECTOR_LIMIT = 10
    CONNECTOR_PER_HOST = 5

# ======================
# DATA STRUCTURES
# ======================
@dataclass
class CompanyData:
    name: str
    id: int

@dataclass
class APIResponse:
    content: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class APIConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int

@dataclass
class ProcessingConfig:
    batch_size: int
    request_delay: float
    max_retries: int
    retry_delay: float
    temperature: float

@dataclass
class FileConfig:
    input_path: str
    output_path: str

# ======================
# PROMPT TEMPLATES
# ======================
class PromptTemplates:
    PERPLEXITY_SEARCH = (
        "{company_name} upcoming building construction projects and programs in North America including retrofit programs. "
        "Find the project start date, completion date, and the status of whether the project is finished. Include only references "
        "to projects that are relevant now or in the future. If a project is finished, include that detail and be clear that it is finished. Do not speculate."
    )
    
    CHATGPT_ANALYSIS = """Do the following for {company_name}: Using the information from {perplexity_summary} only. 

Summarize up to the five most relevant construction programs or projects this company is directly undertaking.

Apply the following criteria strictly:

Overall, we are looking to find companies that are building physical space for their own employees. Only include projects that apply.

1. Relevance to the Company: Only include building construction projects directly undertaken by this company for its own operations, 
students, facilities, or strategic initiatives. Exclude general industry initiatives, government programs, or external projects 
where the company is not the owner, developer, or primary beneficiary.

2. Alignment with Strategic Objectives: Focus on construction related to building space owned or directly operated by the company.

3. Stage of Construction:

    • Prioritize pre-bid or early-phase projects that suggest future work, which end in 2026 or 2027 or later.
    • Projects in the bidding phase should increase the likelihood of relevance.

4. Unified Programs: Roll up individual projects into unified programs when applicable (e.g., multiple sites across a region).

5. Size and Scale: Highlight projects with significant planned square footage, budget, or a large aggregate size if multiple small 
projects combine to create substantial output.

Response Format:

For each relevant program or project, provide:

    • Name
    • 2-sentence description
    • Start date
    • End date
    • Planned square footage / size
    • Planned budget
    • General contractor
    • Electrical engineer (if selected)

Check if the projects are the same, and combine them if they are.

Scoring:

Assign a score from 1 to 5 to each project based on the following:

Date range: projects that end in 2026, 2027, 2028, or 2029 are acceptable. You can summarize projects that end later than that or don't have a clear end date, but do not count them in the final score. Scores above 1 can only be assigned to projects that occur in North America. Scores above 1 can only be assigned to office, retail, education, manufacturing, or warehouse space. Projects ending in 2025 should receive a score of 2 or less.

    • 5: The company is actively engaged in pre-bid or early-phase construction projects aligned with strategic goals. The project 
      or program is over 100,000 sq ft, or the size is not defined but likely larger than 100,000 sq ft for the project or program 
      overall.
    • 4: The company is actively engaged in pre-bid or early-phase construction projects aligned with strategic goals and the project 
      or program is under 100,000 sq ft.
    • 3: The company is actively engaged in pre-bid or early-phase construction projects aligned with strategic goals but specific 
      sizes are not defined.
    • 2: Projects are nearing completion or are unrelated to company-owned facilities.
    • 1: No relevant or direct construction projects found.

Score each project.

Scoring is based on the logic of OR. Start each analysis with a score of 1. Any disqualification criteria (date, type of building, type of initiative) means that the score is a 1.

Then, take the highest score of all the projects.

End your statement with a repeat of score on a new line in this format: Score: [1-5]. Strictly follow this."""

# ======================
# CONFIGURATION
# ======================
class Config:
    def __init__(self):
        self.api_configs = self._create_api_configs()
        self.processing_config = self._create_processing_config()
        self.file_config = self._create_file_config()
        self.prompts = PromptTemplates()

    def _create_api_configs(self) -> Dict[str, APIConfig]:
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not perplexity_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        return {
            "perplexity": APIConfig(
                api_key=perplexity_key,
                base_url="https://api.perplexity.ai/chat/completions",
                model=APIConstants.PERPLEXITY_MODEL,
                max_tokens=APIConstants.PERPLEXITY_MAX_TOKENS
            ),
            "openai": APIConfig(
                api_key=openai_key,
                base_url="https://api.openai.com/v1/chat/completions",
                model=APIConstants.OPENAI_MODEL,
                max_tokens=APIConstants.OPENAI_MAX_TOKENS
            )
        }

    def _create_processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            batch_size=1,
            request_delay=3.0,
            max_retries=3,
            retry_delay=6.0,
            temperature=0.1
        )

    def _create_file_config(self) -> FileConfig:
        return FileConfig(
            input_path="input_companies.csv",
            output_path="output_results.csv"
        )

    def get_perplexity_prompt(self, company_name: str) -> str:
        """Get the Perplexity API prompt for a company"""
        return self.prompts.PERPLEXITY_SEARCH.format(company_name=company_name)

    def get_chatgpt_prompt(self, company_name: str, perplexity_summary: str) -> str:
        """Get the ChatGPT API prompt for analyzing Perplexity results"""
        return self.prompts.CHATGPT_ANALYSIS.format(
            company_name=company_name,
            perplexity_summary=perplexity_summary
        )

config = Config()

# ======================
# LOGGING CONFIGURATION
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================
# API CLIENTS
# ======================
class APIClient:
    """Base class for API clients"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_config.api_key}",
            "Content-Type": "application/json"
        }
    
    def build_request_data(self, prompt: str, temperature: float) -> Dict[str, Any]:
        return {
            "model": self.api_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.api_config.max_tokens,
            "temperature": temperature
        }

class PerplexityClient(APIClient):
    """Client for interacting with Perplexity API"""
    pass

class OpenAIClient(APIClient):
    """Client for interacting with OpenAI API"""
    pass

# ======================
# API FUNCTIONS
# ======================
async def make_api_request(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], 
                          data: Dict[str, Any], retries: int = 0) -> Dict[str, Any]:
    """Make an API request with retry logic"""
    try:
        async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=APIConstants.REQUEST_TIMEOUT)) as response:
            if response.status == 200:
                result = await response.json()
                
                # Log rate limit headers for monitoring (even on success)
                rate_limit_headers = {
                    'x-ratelimit-remaining-requests': response.headers.get('x-ratelimit-remaining-requests', 'N/A'),
                    'x-ratelimit-remaining-tokens': response.headers.get('x-ratelimit-remaining-tokens', 'N/A'),
                    'x-ratelimit-reset-requests': response.headers.get('x-ratelimit-reset-requests', 'N/A'),
                    'x-ratelimit-reset-tokens': response.headers.get('x-ratelimit-reset-tokens', 'N/A'),
                    'x-ratelimit-limit-requests': response.headers.get('x-ratelimit-limit-requests', 'N/A'),
                    'x-ratelimit-limit-tokens': response.headers.get('x-ratelimit-limit-tokens', 'N/A'),
                }
                
                # Only log if we have rate limit headers (OpenAI responses)
                if any(v != 'N/A' for v in rate_limit_headers.values()):
                    logging.info(f"Rate limit status: {rate_limit_headers}")
                
                return result
            elif response.status == APIConstants.RATE_LIMIT_STATUS:  # Rate limit exceeded
                # Log detailed rate limit information
                rate_limit_headers = {
                    'x-ratelimit-remaining-requests': response.headers.get('x-ratelimit-remaining-requests', 'N/A'),
                    'x-ratelimit-remaining-tokens': response.headers.get('x-ratelimit-remaining-tokens', 'N/A'),
                    'x-ratelimit-reset-requests': response.headers.get('x-ratelimit-reset-requests', 'N/A'),
                    'x-ratelimit-reset-tokens': response.headers.get('x-ratelimit-reset-tokens', 'N/A'),
                    'x-ratelimit-limit-requests': response.headers.get('x-ratelimit-limit-requests', 'N/A'),
                    'x-ratelimit-limit-tokens': response.headers.get('x-ratelimit-limit-tokens', 'N/A'),
                }
                
                # Get the actual error response body
                error_body = await response.text()
                logging.warning(f"Rate limit headers: {rate_limit_headers}")
                logging.warning(f"Rate limit error body: {error_body}")
                
                if retries < config.processing_config.max_retries:
                    wait_time = config.processing_config.retry_delay * (2 ** retries)  # Exponential backoff
                    logging.warning(f"Rate limit hit, waiting {wait_time}s before retry {retries + 1}")
                    await asyncio.sleep(wait_time)
                    return await make_api_request(session, url, headers, data, retries + 1)
                else:
                    raise Exception(f"Rate limit exceeded after {config.processing_config.max_retries} retries")
            else:
                error_text = await response.text()
                logging.error(f"API request failed with status {response.status}")
                logging.error(f"Response headers: {dict(response.headers)}")
                logging.error(f"Response body: {error_text}")
                raise Exception(f"API request failed with status {response.status}: {error_text}")
    except asyncio.TimeoutError:
        if retries < config.processing_config.max_retries:
            logging.warning(f"Request timeout, retrying {retries + 1}/{config.processing_config.max_retries}")
            await asyncio.sleep(config.processing_config.retry_delay)
            return await make_api_request(session, url, headers, data, retries + 1)
        else:
            raise Exception("Request timeout after maximum retries")
    except Exception as e:
        if retries < config.processing_config.max_retries:
            logging.warning(f"Request failed: {e}, retrying {retries + 1}/{config.processing_config.max_retries}")
            await asyncio.sleep(config.processing_config.retry_delay)
            return await make_api_request(session, url, headers, data, retries + 1)
        else:
            raise e

async def query_perplexity(company_name: str, session: aiohttp.ClientSession) -> str:
    """Query Perplexity API for company construction projects"""
    client = PerplexityClient(config.api_configs["perplexity"])
    prompt = config.get_perplexity_prompt(company_name)
    
    headers = client.get_headers()
    data = client.build_request_data(prompt, config.processing_config.temperature)
    
    try:
        result = await make_api_request(session, config.api_configs["perplexity"].base_url, headers, data)
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            logging.info(f"Perplexity response for {company_name}: {len(content)} characters")
            return content
        else:
            logging.error(f"Unexpected Perplexity response format for {company_name}: {result}")
            return f"Error: Unexpected response format from Perplexity API"
            
    except Exception as e:
        logging.error(f"Perplexity API error for {company_name}: {e}")
        return f"Error: {str(e)}"

async def query_chatgpt(company_name: str, perplexity_summary: str, session: aiohttp.ClientSession) -> str:
    """Query OpenAI API for analysis of Perplexity results"""
    client = OpenAIClient(config.api_configs["openai"])
    prompt = config.get_chatgpt_prompt(company_name, perplexity_summary)
    
    headers = client.get_headers()
    data = client.build_request_data(prompt, config.processing_config.temperature)
    
    try:
        result = await make_api_request(session, config.api_configs["openai"].base_url, headers, data)
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            logging.info(f"OpenAI response for {company_name}: {len(content)} characters")
            return content
        else:
            logging.error(f"Unexpected OpenAI response format for {company_name}: {result}")
            return f"Error: Unexpected response format from OpenAI API"
            
    except Exception as e:
        logging.error(f"OpenAI API error for {company_name}: {e}")
        return f"Error: {str(e)}"

# ======================
# PROCESSING FUNCTIONS
# ======================
def read_input_csv(filepath: str) -> list:
    df = pd.read_csv(filepath)
    if "id" not in df.columns:
        df.insert(0, "id", pd.Series(range(1, len(df) + 1)))
        df.to_csv(filepath, index=False)
    return df.to_dict(orient="records")

def get_processed_ids(output_filepath: str) -> set:
    if not Path(output_filepath).exists():
        return set()
    with open(output_filepath, "r", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(outfile)
        return {int(row["id"]) for row in reader if row["id"].isdigit()}

async def process_company(company_name: str, session: aiohttp.ClientSession) -> Dict[str, str]:
    """Process a single company with rate limiting"""
    logging.info(f"Processing company: {company_name}")
    
    # Query Perplexity
    perplexity_summary = await query_perplexity(company_name, session)
    await asyncio.sleep(config.processing_config.request_delay)  # Rate limiting
    
    # Query ChatGPT
    chatgpt_summary = await query_chatgpt(company_name, perplexity_summary, session)
    await asyncio.sleep(config.processing_config.request_delay)  # Rate limiting
    
    return {
        "company_name": company_name,
        "perplexity_output": perplexity_summary,
        "chatgpt_output": chatgpt_summary,
    }

async def process_batch(companies: list, writer: csv.DictWriter, session: aiohttp.ClientSession) -> None:
    """Process a batch of companies"""
    tasks = [process_company(company["company_name"], session) for company in companies]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Error processing company {companies[i]['company_name']}: {result}")
            # Write error result to maintain data integrity
            error_row: Dict[str, str] = {
                "company_name": companies[i]["company_name"],
                "perplexity_output": f"Error: {str(result)}",
                "chatgpt_output": f"Error: {str(result)}",
            }
            writer.writerow(error_row)
        elif isinstance(result, dict):
            writer.writerow(result)

async def main_async() -> None:
    logging.info("Starting processing...")
    companies = read_input_csv(config.file_config.input_path)
    processed_ids = get_processed_ids(config.file_config.output_path)
    unprocessed_companies = [row for row in companies if row["id"] not in processed_ids]

    if not unprocessed_companies:
        logging.info("All companies have already been processed.")
        return

    # Create aiohttp session for connection pooling
    connector = aiohttp.TCPConnector(
        limit=APIConstants.CONNECTOR_LIMIT, 
        limit_per_host=APIConstants.CONNECTOR_PER_HOST
    )
    timeout = aiohttp.ClientTimeout(total=APIConstants.SESSION_TIMEOUT)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with open(config.file_config.output_path, "a", newline="", encoding="utf-8") as outfile:
            fieldnames = ["id", "company_name", "perplexity_output", "chatgpt_output"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            if not processed_ids:
                writer.writeheader()

            async def process_all_batches():
                for i in tqdm(range(0, len(unprocessed_companies), config.processing_config.batch_size), desc="Processing Batches"):
                    batch = unprocessed_companies[i:i + config.processing_config.batch_size]
                    await process_batch(batch, writer, session)
                    # Additional delay between batches - increased to 6 seconds
                    await asyncio.sleep(config.processing_config.request_delay * 2)

            await process_all_batches()

    logging.info(f"Processing complete. Results saved to {config.file_config.output_path}.")

# ======================
# MOCK INPUT CREATION
# ======================
def create_mock_input() -> None:
    sample_data = {"company_name": ["Apple", "Google", "Microsoft", "Amazon", "Facebook", "Tesla", "Netflix", "Uber", "Airbnb", "Spotify"]}
    df = pd.DataFrame(sample_data)
    df.to_csv(config.file_config.input_path, index=False)

# ======================
# RUN THE SCRIPT
# ======================
if __name__ == "__main__":
    create_mock_input()
    asyncio.run(main_async())