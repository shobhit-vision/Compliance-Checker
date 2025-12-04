import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import zlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io as reportlab_io

# Email Imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# LangChain imports
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import cassio
from astrapy import DataAPIClient

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# Email Notification System (PRODUCTION READY)
# -----------------------------------------------------------------------------
class EmailNotificationSystem:
    def __init__(self):
        # Load SMTP credentials from api.txt
        self.sender_email = None
        self.sender_password = None
        self.load_smtp_credentials()

    def load_smtp_credentials(self):
        """Load SMTP credentials from api.txt file"""
        try:
            self.sender_email = os.getenv('SENDER_EMAIL')
            self.sender_password = os.getenv('SENDER_PASSWORD')

            if not all([self.sender_email, self.sender_password]):
                st.warning("⚠️ SMTP credentials incomplete . Email alerts will be disabled.")

        except Exception as e:
            st.error(f"❌ Error loading SMTP credentials: {str(e)}")


    def send_alert(self, to_email, subject, body, pdf_attachments=None):
        """
        Sends an email alert - PRODUCTION VERSION
        Only sends when contract is successfully revised and saved
        """
        if not to_email:
            st.warning("⚠️ No recipient email provided")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add body to email
            msg.attach(MIMEText(body, 'plain'))

            # Add PDF attachments if provided
            if pdf_attachments:
                for attachment_name, pdf_buffer in pdf_attachments.items():
                    if pdf_buffer:
                        # Create attachment
                        attachment = MIMEText(pdf_buffer.getvalue(), 'plain', 'utf-8')
                        attachment.add_header('Content-Disposition', 
                                            'attachment', 
                                            filename=attachment_name)
                        msg.attach(attachment)

            # SPECIFIC GMAIL LOGIC REQUESTED
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, to_email, msg.as_string())
            server.quit()

            return True

        except Exception as e:
            st.error(f"❌ Failed to send email: {str(e)}")
            return False

# -----------------------------------------------------------------------------
# Global Contract Manager (State Management)
# -----------------------------------------------------------------------------
class GlobalContractManager:
    """Manages the in-memory state of all contracts for quick access."""
    def __init__(self):
        self.contracts = {}
        self.revision_data = {}
        self.compliance_data = {}

    def initialize_contracts_from_collections(self, db_collections):
        """Initialize contracts from database collections starting with 'contract_'"""
        # Reset current contracts to avoid duplicates during re-init
        self.contracts = {}
        for collection_name in db_collections:
            if collection_name.startswith('contract_'):
                contract_name = collection_name.replace('contract_', '').replace('_', ' ').title()
                self.contracts[contract_name] = {
                    'collection_name': collection_name,
                    'data': {},
                    'metadata': {},
                    'status': {
                        'needs_revision': False,
                        'revision_status': 'Current',
                        'updation_status': 'Up to date',
                        'risk_score': 0,
                        'last_updated': datetime.now().isoformat()
                    },
                    'revision_history': []
                }

    def update_contract_status(self, contract_name, needs_revision=False, revision_status="Current",
                             updation_status="Up to date", risk_score=0):
        """Update contract status in the global object"""
        if contract_name in self.contracts:
            self.contracts[contract_name]['status'].update({
                'needs_revision': needs_revision,
                'revision_status': revision_status,
                'updation_status': updation_status,
                'risk_score': risk_score,
                'last_updated': datetime.now().isoformat()
            })

    def get_contracts_needing_revision(self):
        """Get contracts that need revision"""
        return {name: data for name, data in self.contracts.items()
                if data['status']['needs_revision']}

    def update_contract_data(self, contract_name, new_data, new_metadata=None):
        """Update contract data in the global object"""
        if contract_name in self.contracts:
            self.contracts[contract_name]['data'] = new_data
            if new_metadata:
                self.contracts[contract_name]['metadata'] = new_metadata
            self.update_contract_status(contract_name, needs_revision=False,
                                      revision_status="Updated", updation_status="Current")

    def add_revision_history(self, contract_name, revision_data):
        """Add revision history to contract"""
        if contract_name in self.contracts:
            if 'revision_history' not in self.contracts[contract_name]:
                self.contracts[contract_name]['revision_history'] = []
            self.contracts[contract_name]['revision_history'].append(revision_data)

    def delete_contract(self, contract_name):
        """Delete contract from global manager"""
        if contract_name in self.contracts:
            del self.contracts[contract_name]
            return True
        return False

# Initialize global contract manager
if 'global_contracts_manager' not in st.session_state:
    st.session_state.global_contracts_manager = GlobalContractManager()

global_contracts = st.session_state.global_contracts_manager

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------
class ConfigManager:
    def __init__(self):
        # Explicitly initialize attributes to None to prevent AttributeError
        self.astra_token = None
        self.astra_db_id = None
        self.groq_api_key = None
        self.sender_email = None
        self.sender_password = None

        self.regulatory_standards = ["GDPR", "DPDPA"]
        self.similarity_threshold = 0.7
        self.max_document_size = 7000
        self.load_credentials()

    def load_credentials(self):
        """Load credentials from a file or environment variables."""
        try:
            self.astra_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
            self.astra_db_id = os.getenv('ASTRA_DB_ID')
            self.groq_api_key = os.getenv('GROQ_API_KEY')
            self.sender_email = os.getenv('SENDER_EMAIL')
            self.sender_password = os.getenv('SENDER_PASSWORD')
        except Exception:
            # Keep defaults as None if loading fails
            pass

# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------
class DocumentProcessor:
    @staticmethod
    def compress_text(text):
        """Compress text using zlib compression"""
        try:
            if isinstance(text, str):
                compressed = zlib.compress(text.encode('utf-8'))
                return compressed
            return text
        except:
            return text

    @staticmethod
    def decompress_text(compressed_data):
        """Decompress text using zlib"""
        try:
            if isinstance(compressed_data, bytes):
                decompressed = zlib.decompress(compressed_data).decode('utf-8')
                return decompressed
            return compressed_data
        except:
            return compressed_data

    @staticmethod
    def optimize_text_for_storage(text):
        """Optimize text for storage while preserving paragraph structure"""
        if not isinstance(text, str):
            return text
        
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Clean up extra spaces within the line
            line = re.sub(r'[ \t]+', ' ', line.strip())
            if line:  # Keep non-empty lines
                optimized_lines.append(line)
        
        # Join with single newlines (preserving paragraph breaks)
        # This assumes paragraphs are separated by blank lines
        result = '\n'.join(optimized_lines)
        
        # Clean up multiple consecutive newlines (more than 2)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

    @staticmethod
    def calculate_text_size(text):
        """Calculate the size of text in bytes"""
        if isinstance(text, str):
            return len(text.encode('utf-8'))
        elif isinstance(text, bytes):
            return len(text)
        return 0

# -----------------------------------------------------------------------------
# AstraDB Management (Optimized with Caching)
# -----------------------------------------------------------------------------
class AstraDBManager:
    def __init__(self, token, db_id):
        self.token = token
        self.db_id = db_id
        # Use fallback if token/db_id are missing to prevent crash during init
        if self.db_id:
            self.api_endpoint = f"https://{self.db_id}-us-east1.apps.astra.datastax.com"
        else:
            self.api_endpoint = ""

        self.client = None
        self.db = None
        self.document_processor = DocumentProcessor()
        self.init_client()

    def init_client(self):
        """Initialize AstraDB client"""
        try:
            if self.token and self.db_id:
                cassio.init(token=self.token, database_id=self.db_id)
                self.client = DataAPIClient(self.token)
                self.db = self.client.get_database_by_api_endpoint(self.api_endpoint)
                return True
        except Exception as e:
            st.error(f"Error initializing AstraDB client: {str(e)}")
        return False

    def collection_exists(self, collection_name):
        """Check if collection exists"""
        if not self.db: return False
        try:
            collections = self.db.list_collection_names()
            return collection_name in collections
        except:
            return False

    def delete_collection(self, collection_name):
        """Delete collection by name"""
        if not self.db: return False
        try:
            if self.collection_exists(collection_name):
                collection = self.db.get_collection(collection_name)
                collection.drop()
                return True
        except Exception as e:
            st.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False

    def prepare_contract_data(self, contract_data, metadata):
        """Prepare contract data for storage, handling large documents"""
        if isinstance(contract_data, str):
            contract_data = self.document_processor.optimize_text_for_storage(contract_data)

        original_size = self.document_processor.calculate_text_size(str(contract_data))

        if original_size > 7000:
            compressed_data = self.document_processor.compress_text(str(contract_data))
            summary = str(contract_data)[:500] + "..." if len(str(contract_data)) > 500 else str(contract_data)
            compressed_size = self.document_processor.calculate_text_size(compressed_data)

            return {
                "data_type": "compressed_large_document",
                "compressed_data": compressed_data,
                "summary": summary,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "needs_decompression": True
            }
        else:
            return {
                "data_type": "direct_storage",
                "contract_data": contract_data,
                "needs_decompression": False
            }

    def save_contract_data(self, collection_name, contract_data, metadata, analysis_data=None):
        """Save complete contract data to AstraDB with large document handling"""
        if not self.db: return False
        try:
            if self.collection_exists(collection_name):
                self.delete_collection(collection_name)

            collection = self.db.create_collection(collection_name)
            prepared_data = self.prepare_contract_data(contract_data, metadata)

            if analysis_data:
                analysis_size = self.document_processor.calculate_text_size(str(analysis_data))
                if analysis_size > 2 * 1024 * 1024:  # 2MB limit
                     analysis_data = str(analysis_data)[:2000000] + "... (Truncated)"

            serialized_metadata = self.serialize_metadata(metadata)

            contract_document = {
                "document_type": "complete_contract",
                **prepared_data,
                "metadata": serialized_metadata,
                "analysis_data": analysis_data,
                "upload_time": datetime.now().isoformat(),
                "contract_name": metadata.get("contract_name", "Unknown"),
                "revision": metadata.get("revision", 0),
                "last_updated": datetime.now().isoformat(),
                "needs_revision": metadata.get("needs_revision", False),
                "risk_score": metadata.get("risk_score", 0),
                "revision_data": metadata.get("revision_data", ""),
                "prev_contract_data": metadata.get("prev_contract_data", ""),
                "revision_details": metadata.get("revision_details", [])
            }

            result = collection.insert_one(contract_document)

            if result.inserted_id:
                # Invalidate cache if data changes
                st.cache_data.clear()
                return True
            else:
                return False

        except Exception as e:
            st.error(f"Error saving contract data: {str(e)}")
            return False

    def serialize_metadata(self, metadata):
        """Serialize metadata with size optimization"""
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serializable_metadata[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                serializable_metadata[key] = json.loads(json.dumps(value, default=str))
            else:
                str_value = str(value)
                if len(str_value) > 200:
                    serializable_metadata[key] = str_value[:200] + "..."
                else:
                    serializable_metadata[key] = value
        return serializable_metadata

    # CACHING: Speed up retrieval by caching results
    @st.cache_data(ttl=1, show_spinner=False)
    def get_contract_data(_self, collection_name):
        """Retrieve complete contract data from AstraDB"""
        if not _self.db: return None, None, None, None, None, None
        try:
            if not _self.collection_exists(collection_name):
                return None, None, None, None, None, None

            collection = _self.db.get_collection(collection_name)
            documents = collection.find()

            for doc in documents:
                if doc.get("document_type") == "complete_contract":
                    if doc.get("needs_decompression", False):
                        compressed_data = doc.get("compressed_data")
                        if compressed_data:
                            contract_data = _self.document_processor.decompress_text(compressed_data)
                        else:
                            contract_data = doc.get("summary", "Data not available")
                    else:
                        contract_data = doc.get("contract_data")

                    metadata = doc.get("metadata", {})
                    analysis_data = doc.get("analysis_data")
                    revision_data = doc.get("revision_data", "")
                    prev_contract_data = doc.get("prev_contract_data", "")
                    revision_details = doc.get("revision_details", [])

                    return contract_data, metadata, analysis_data, revision_data, prev_contract_data, revision_details

            return None, None, None, None, None, None
        except Exception as e:
            return None, None, None, None, None, None

    def update_contract_revision(self, collection_name, new_contract_data, new_metadata, new_analysis_data=None, revision_details=None):
        """Update contract with new revision"""
        if not self.db: return False
        try:
            if not self.collection_exists(collection_name):
                st.error(f"Collection {collection_name} does not exist")
                return False

            collection = self.db.get_collection(collection_name)

            # We use the internal method to avoid cache for update operations, but read initial data
            current_data, current_metadata, current_analysis, current_revision_data, current_prev_data, current_revision_details = self.get_contract_data(collection_name)

            delete_result = collection.delete_many({"document_type": "complete_contract"})

            prepared_data = self.prepare_contract_data(new_contract_data, new_metadata)
            serialized_metadata = self.serialize_metadata(new_metadata)
            prev_contract_data = current_data if current_data else ""

            # Preserve existing revision_data if not explicitly overwritten in metadata
            revision_data_to_store = new_metadata.get("revision_data", current_revision_data)

            updated_document = {
                "document_type": "complete_contract",
                **prepared_data,
                "metadata": serialized_metadata,
                "analysis_data": new_analysis_data if new_analysis_data else current_analysis,
                "upload_time": datetime.now().isoformat(),
                "contract_name": new_metadata.get("contract_name", "Unknown"),
                "revision": new_metadata.get("revision", 1),
                "last_updated": datetime.now().isoformat(),
                "needs_revision": new_metadata.get("needs_revision", False),
                "risk_score": new_metadata.get("risk_score", 0),
                "revision_data": revision_data_to_store,
                "prev_contract_data": prev_contract_data,
                "revision_details": revision_details if revision_details else current_revision_details
            }

            insert_result = collection.insert_one(updated_document)

            if insert_result.inserted_id:
                st.cache_data.clear() # Clear cache on update
                return True
            else:
                st.error("❌ Failed to insert updated contract document")
                return False

        except Exception as e:
            st.error(f"Error updating contract revision: {str(e)}")
            return False

    def get_contracts_needing_revision(self):
        """Get contracts that need revision based on risk score"""
        if not self.db: return []
        try:
            collections = self.db.list_collection_names()
            contract_collections = [col for col in collections if col.startswith('contract_')]

            contracts_needing_revision = []
            for collection_name in contract_collections:
                contract_data, metadata, analysis_data, revision_data, prev_contract_data, revision_details = self.get_contract_data(collection_name)

                if metadata and metadata.get('needs_revision', False):
                    contracts_needing_revision.append({
                        'name': metadata.get('contract_name', collection_name),
                        'risk_score': metadata.get('risk_score', 0),
                        'collection_name': collection_name,
                        'metadata': metadata,
                        'contract_data': contract_data,
                        'revision_data': revision_data,
                        'prev_contract_data': prev_contract_data
                    })

            return contracts_needing_revision
        except Exception as e:
            st.error(f"Error getting contracts needing revision: {str(e)}")
            return []

    def get_contract_count(self):
        """Get number of contracts in database"""
        if not self.db: return 0
        try:
            collections = self.db.list_collection_names()
            contract_collections = [col for col in collections if col.startswith('contract_')]
            return len(contract_collections)
        except Exception as e:
            return 0

    @st.cache_data(ttl=1) # Short TTL for live updates
    def get_all_contract_collections(_self):
        """Get all contract collections (Cached)"""
        if not _self.db: return []
        try:
            collections = _self.db.list_collection_names()
            return [col for col in collections if col.startswith('contract_')]
        except Exception as e:
            st.error(f"Error getting contract collections: {str(e)}")
            return []

# -----------------------------------------------------------------------------
# Enhanced LLM Metadata Extractor - FIXED VERSION
# -----------------------------------------------------------------------------
class LLMMetadataExtractor:
    """Enhanced class for extracting metadata using LLM with structured output"""
    
    @staticmethod
    def extract_with_llm(contract_text, llm, contract_name="", owner_email=""):
        """Extract comprehensive metadata using LLM with structured output"""
        if not llm:
            return None
            
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal document analyzer. Extract the following information from the contract text:

EXTRACT THE FOLLOWING INFORMATION:
1. PARTIES: List all parties mentioned as parties to the agreement (e.g., "Company ABC", "John Doe"). Format: "Party 1, Party 2"
2. SIGN_DATE: Format: "YYYY-MM-DD" or "Month Day, Year" not add hour minute or second
3. CONTRACT_TYPE: Type of contract (e.g., "Service Agreement", "NDA", "Employment Contract")
4. TERMINATION_DATE: The date when the contract terminates or expires. Format: "YYYY-MM-DD" or "Month Day, Year"
5. KEY_CLAUSES: Important clauses in the contract (e.g., Confidentiality, Termination, Payment)

RETURN FORMAT - Provide exactly one line for each field:
PARTIES: [list of parties]
SIGN_DATE: [sign date]
CONTRACT_TYPE: [contract type]
TERMINATION_DATE: [termination date]
KEY_CLAUSES: [key clauses]

If you cannot determine a value, use "Unknown" for that field.
Do not include any additional text, explanations, or formatting beyond these 5 lines."""),
                ("human", f"Contract Text:\n{contract_text[:8000]}")
            ])

            extraction_chain = prompt | llm | StrOutputParser()
            response = extraction_chain.invoke({})
            
            return LLMMetadataExtractor.parse_structured_response(response, contract_name, owner_email)
            
        except Exception as e:
            st.error(f"Error in LLM metadata extraction: {str(e)}")
            return None
    
    @staticmethod
    def parse_structured_response(response_text, contract_name, owner_email):
        """Parse the structured LLM response"""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'parties': [],
                'sign_date': 'Unknown',
                'contract_type': 'Unknown',
                'termination_date': 'Unknown',
                'key_clauses': [],
                'contract_name': contract_name,
                'owner_email': owner_email,
                'revision': 0,
                'needs_revision': False,
                'risk_score': 0,
                'last_updated': datetime.now().isoformat(),
                'revision_history': [],
                'revision_data': "",
                'prev_contract_data': "",
                'revision_details': []
            }
            
            for line in lines:
                line = line.strip()
                
                if line.upper().startswith('PARTIES:'):
                    parties_text = line.split(':', 1)[1].strip()
                    if parties_text and parties_text.upper() != 'UNKNOWN':
                        # Split by commas and clean
                        parties = [p.strip() for p in parties_text.split(',')]
                        result['parties'] = parties
                
                elif line.upper().startswith('SIGN_DATE:'):
                    date_text = line.split(':', 1)[1].strip()
                    if date_text and date_text.upper() != 'UNKNOWN':
                        result['sign_date'] = date_text
                
                elif line.upper().startswith('CONTRACT_TYPE:'):
                    type_text = line.split(':', 1)[1].strip()
                    if type_text and type_text.upper() != 'UNKNOWN':
                        result['contract_type'] = type_text
                
                elif line.upper().startswith('TERMINATION_DATE:'):
                    term_text = line.split(':', 1)[1].strip()
                    if term_text and term_text.upper() != 'UNKNOWN':
                        result['termination_date'] = term_text
                
                elif line.upper().startswith('KEY_CLAUSES:'):
                    clauses_text = line.split(':', 1)[1].strip()
                    if clauses_text and clauses_text.upper() != 'UNKNOWN':
                        # Split by commas and clean
                        clauses = [c.strip() for c in clauses_text.split(',')]
                        result['key_clauses'] = clauses
            
            # Parse dates and calculate days left
            result = LLMMetadataExtractor.process_dates(result)
            
            # Add other required fields
            result['contract_size'] = 0  # Will be set later
            result['word_count'] = 0  # Will be set later
            result['regulatory_standards'] = ["GDPR", "DPDPA"]  # Default
            result['status'] = "Active" if result.get('days_left', 'Unknown') != 'Unknown' and result.get('days_left', 0) > 0 else "Expired"
            result['needs_update'] = result.get('days_left', 'Unknown') != 'Unknown' and result.get('days_left', 0) <= 30
            
            return result
            
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
            # Return basic structure on error
            return {
                'parties': [],
                'sign_date': 'Unknown',
                'contract_type': 'Unknown',
                'termination_date': 'Unknown',
                'key_clauses': [],
                'contract_name': contract_name,
                'owner_email': owner_email,
                'revision': 0,
                'needs_revision': False,
                'risk_score': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'days_left': 'Unknown',
                'status': 'Unknown',
                'needs_update': False,
                'contract_size': 0,
                'word_count': 0,
                'regulatory_standards': ["GDPR", "DPDPA"],
                'revision_history': [],
                'revision_data': "",
                'prev_contract_data': "",
                'revision_details': []
            }
    
    @staticmethod
    def process_dates(metadata):
        """Process dates from extracted metadata - SIMPLE 180 DAY RULE"""
        try:
            # Parse sign date
            sign_date_str = metadata.get('sign_date', 'Unknown')
            
            if sign_date_str != 'Unknown':
                sign_date = LLMMetadataExtractor.parse_date(sign_date_str)
                if sign_date:
                    # Format sign date as YYYY-MM-DD
                    metadata['sign_date'] = sign_date.strftime('%Y-%m-%d')
                    metadata['contract_date'] = sign_date.strftime('%Y-%m-%d')
                    
                    # Calculate termination date (180 days from sign date)
                    termination_date = sign_date + timedelta(days=180)
                    metadata['termination_date'] = termination_date.strftime('%Y-%m-%d')
                    
                    # Calculate days left
                    days_left = (termination_date - datetime.now()).days
                    metadata['days_left'] = max(0, days_left)
                    
                    # Set status
                    if days_left > 0:
                        metadata['status'] = "Active"
                        metadata['needs_update'] = days_left <= 30
                    else:
                        metadata['status'] = "Expired"
                        metadata['needs_update'] = False
                else:
                    metadata['sign_date'] = 'Unknown'
                    metadata['contract_date'] = 'Unknown'
                    metadata['termination_date'] = 'Unknown'
                    metadata['days_left'] = 'Unknown'
                    metadata['status'] = 'Unknown'
                    metadata['needs_update'] = False
            else:
                metadata['sign_date'] = 'Unknown'
                metadata['contract_date'] = 'Unknown'
                metadata['termination_date'] = 'Unknown'
                metadata['days_left'] = 'Unknown'
                metadata['status'] = 'Unknown'
                metadata['needs_update'] = False
                
            return metadata
                
        except Exception as e:
            # Set defaults on error
            metadata['sign_date'] = 'Unknown'
            metadata['contract_date'] = 'Unknown'
            metadata['termination_date'] = 'Unknown'
            metadata['days_left'] = 'Unknown'
            metadata['status'] = 'Unknown'
            metadata['needs_update'] = False
            return metadata
    
    @staticmethod
    def parse_date(date_str):
        """Parse date from various formats - SAFE VERSION"""
        if not date_str or date_str == 'Unknown':
            return None
        
        date_str = str(date_str).strip()
        
        # First try to parse as ISO format
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            pass
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',          # 2024-01-15
            '%B %d, %Y',         # January 15, 2024
            '%b %d, %Y',         # Jan 15, 2024
            '%d/%m/%Y',          # 15/01/2024
            '%m/%d/%Y',          # 01/15/2024
            '%d-%m-%Y',          # 15-01-2024
            '%m-%d-%Y',          # 01-15-2024
            '%Y/%m/%d',          # 2024/01/15
            '%d %B %Y',          # 15 January 2024
            '%d %b %Y',          # 15 Jan 2024
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # Try to extract date from string using regex
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(\d{1,2}-\d{1,2}-\d{4})\b',
            r'\b(\w+ \d{1,2}, \d{4})\b',
            r'\b(\d{1,2} \w+ \d{4})\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                extracted_date = match.group(1)
                for fmt in date_formats:
                    try:
                        return datetime.strptime(extracted_date, fmt)
                    except:
                        continue
        
        return None

# -----------------------------------------------------------------------------
# Contract Manager
# -----------------------------------------------------------------------------
class ContractManager:
    def __init__(self, database, global_contracts):
        self.database = database
        self.config = database.config
        self.revision_data = {}
        self.name_manager = ComplianceNameManager()
        self.global_contracts = global_contracts
        self.document_processor = DocumentProcessor()
        # Initialize Email System
        self.email_system = EmailNotificationSystem()
        # Initialize LLM Metadata Extractor
        self.metadata_extractor = LLMMetadataExtractor()

    def initialize_global_contracts(self, force_refresh=False):
        """Initialize global contracts from database collections"""
        try:
            if not force_refresh and self.global_contracts.contracts:
                return self.global_contracts.contracts

            # Clear cache to ensure live data
            if force_refresh:
                st.cache_data.clear()

            collections = self.database.astra_db.get_all_contract_collections()
            self.global_contracts.initialize_contracts_from_collections(collections)

            for collection_name in collections:
                contract_name = collection_name.replace('contract_', '').replace('_', ' ').title()
                # Get cached data (cache cleared if force_refresh)
                contract_data, metadata, analysis_data, revision_data, prev_contract_data, revision_details = self.database.astra_db.get_contract_data(collection_name)

                if contract_data and metadata:
                    self.global_contracts.contracts[contract_name]['data'] = contract_data
                    self.global_contracts.contracts[contract_name]['metadata'] = metadata
                    self.global_contracts.contracts[contract_name]['status'].update({
                        'needs_revision': metadata.get('needs_revision', False),
                        'risk_score': metadata.get('risk_score', 0),
                        'revision': metadata.get('revision', 0),
                        'last_updated': metadata.get('last_updated', datetime.now().isoformat())
                    })
                    self.global_contracts.contracts[contract_name]['analysis_data'] = analysis_data
                    self.global_contracts.contracts[contract_name]['revision_data'] = revision_data
                    self.global_contracts.contracts[contract_name]['prev_contract_data'] = prev_contract_data
                    self.global_contracts.contracts[contract_name]['revision_details'] = revision_details

            return self.global_contracts.contracts

        except Exception as e:
            st.error(f"Error initializing global contracts: {str(e)}")
            return {}

    # Validation Function for GDPR/DPDPA
    def validate_contract_compliance(self, text):
        """Ensure contract is strictly GDPR or DPDPA related."""
        keywords = [
            "gdpr", "general data protection regulation",
            "dpdpa", "digital personal data protection",
            "data protection act"
        ]
        return any(keyword in text.lower() for keyword in keywords)

    def extract_contract_metadata_with_llm(self, contract_text, contract_name="", owner_email="", llm=None):
        """Extract comprehensive metadata from contract using LLM"""
        try:
            if not llm:
                # Fallback to regex if LLM not available
                return self.extract_contract_metadata_fallback(contract_text, contract_name, owner_email)
            
            # Use enhanced LLM metadata extractor
            metadata = self.metadata_extractor.extract_with_llm(contract_text, llm, contract_name, owner_email)
            
            if metadata:
                # Add additional fields
                metadata['contract_size'] = len(contract_text)
                metadata['word_count'] = len(contract_text.split())
                metadata['regulatory_standards'] = [self.detect_regulatory_standards(contract_text)]
                metadata['upload_time'] = datetime.now().isoformat()
                metadata['last_updated'] = datetime.now().isoformat()
                
                return metadata
            else:
                # Fallback if LLM extraction fails
                return self.extract_contract_metadata_fallback(contract_text, contract_name, owner_email)
            
        except Exception as e:
            st.error(f"Error in LLM metadata extraction: {str(e)}")
            # Fallback to regex-based extraction
            return self.extract_contract_metadata_fallback(contract_text, contract_name, owner_email)

    def extract_contract_metadata_fallback(self, contract_text, contract_name="", owner_email=""):
        """Fallback metadata extraction using regex when LLM fails"""
        try:
            contract_date = self.extract_contract_date(contract_text)
            parties = self.extract_parties(contract_text)
            contract_type = self.extract_contract_type(contract_text)
            key_clauses = self.extract_key_clauses(contract_text)

            # Try to extract termination date
            termination_date = self.extract_termination_date(contract_text)
            if not termination_date:
                # Fallback: 180 days from sign date
                termination_date = contract_date + timedelta(days=180)
            
            days_left = (termination_date - datetime.now()).days

            return {
                'contract_date': contract_date.isoformat(),
                'sign_date': contract_date.isoformat(),
                'termination_date': termination_date.isoformat(),
                'days_left': max(0, days_left),
                'status': "Active" if days_left > 0 else "Expired",
                'needs_update': days_left <= 30,
                'parties': parties,
                'contract_type': contract_type,
                'key_clauses': key_clauses,
                'last_updated': datetime.now().isoformat(),
                'contract_size': len(contract_text),
                'word_count': len(contract_text.split()),
                'regulatory_standards': [self.detect_regulatory_standards(contract_text)],
                'revision': 0,
                'needs_revision': False,
                'risk_score': 0,
                'contract_name': contract_name,
                'owner_email': owner_email,
                'revision_history': [],
                'revision_data': "",
                'prev_contract_data': "",
                'revision_details': []
            }
        except Exception as e:
            st.error(f"Error in fallback metadata extraction: {str(e)}")
            # Return minimal metadata on error
            return {
                'contract_date': datetime.now().isoformat(),
                'sign_date': datetime.now().isoformat(),
                'termination_date': (datetime.now() + timedelta(days=180)).isoformat(),
                'days_left': 180,
                'status': "Active",
                'needs_update': False,
                'parties': [],
                'contract_type': 'Unknown',
                'key_clauses': [],
                'last_updated': datetime.now().isoformat(),
                'contract_size': len(contract_text),
                'word_count': len(contract_text.split()),
                'regulatory_standards': [self.detect_regulatory_standards(contract_text)],
                'revision': 0,
                'needs_revision': False,
                'risk_score': 0,
                'contract_name': contract_name,
                'owner_email': owner_email,
                'revision_history': [],
                'revision_data': "",
                'prev_contract_data': "",
                'revision_details': []
            }

    def extract_contract_metadata(self, contract_text, contract_name="", owner_email=""):
        """Wrapper for backward compatibility"""
        return self.extract_contract_metadata_fallback(contract_text, contract_name, owner_email)

    def detect_regulatory_standards(self, contract_text):
        """Detect which regulatory standards apply to this contract"""
        return self.name_manager.detect_compliance_standard(contract_text)

    def extract_contract_date(self, contract_text):
        """Enhanced contract date extraction with multiple patterns"""
        patterns = [
            r'\b(?:signed|dated|executed|effective)\s+on\s+(\w+\s+\d{1,2},\s*\d{4})',
            r'\b(\w+\s+\d{1,2},\s*\d{4})\s*(?:\(?date\)?)?',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\b\d{4}-\d{1,2}-\d{1,2}',
            r'date\s*[.:]\s*(\w+\s+\d{1,2},\s*\d{4})',
            r'this\s+agreement.*?(\w+\s+\d{1,2},\s*\d{4})',
            r'made\s+.*?(\w+\s+\d{1,2},\s*\d{4})'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE)
            if matches:
                try:
                    date_str = matches[0]
                    for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    continue

        return datetime.now()

    def extract_termination_date(self, contract_text):
        """Extract termination date from contract text"""
        patterns = [
            r'terminat(?:ion|es|ing)\s+on\s+(\w+\s+\d{1,2},\s*\d{4})',
            r'expir(?:ation|es|y)\s+on\s+(\w+\s+\d{1,2},\s*\d{4})',
            r'end(?:s|ing)?\s+on\s+(\w+\s+\d{1,2},\s*\d{4})',
            r'term\s+.*?(\w+\s+\d{1,2},\s*\d{4})',
            r'valid\s+until\s+(\w+\s+\d{1,2},\s*\d{4})'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE)
            if matches:
                try:
                    date_str = matches[0]
                    for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    continue

        return None

    def extract_parties(self, contract_text):
        """Extract parties from contract text"""
        parties = []
        patterns = [
            r'between\s+([^,]+?)\s+and\s+([^,\.]+)',
            r'parties:\s*(.+?)\s*and\s*(.+)',
            r'this\s+agreement.*?between\s+([^,]+?)\s+and\s+([^,\.]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE | re.DOTALL)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        parties.extend([party.strip() for party in match])
                    else:
                        parties.append(match.strip())

        parties = list(set([p for p in parties if p and len(p) > 2]))

        if not parties:
            company_pattern = r'\b(?:LLC|Inc|Corp|Corporation|Company|Ltd)\b'
            potential_parties = re.findall(r'([A-Z][a-zA-Z\s]+?' + company_pattern + r')', contract_text)
            parties = list(set(potential_parties))

        return parties[:4]

    def extract_contract_type(self, contract_text):
        """Extract contract type from content"""
        text_lower = contract_text.lower()

        type_patterns = {
            'Service Agreement': ['service agreement', 'service contract', 'services agreement'],
            'Employment Contract': ['employment agreement', 'employment contract', 'employee agreement'],
            'NDA': ['non-disclosure', 'confidentiality agreement', 'nda'],
            'Partnership': ['partnership agreement', 'joint venture', 'collaboration agreement'],
            'License Agreement': ['license agreement', 'licensing agreement', 'software license'],
            'Purchase Agreement': ['purchase agreement', 'sales agreement', 'purchase contract'],
            'Lease Agreement': ['lease agreement', 'rental agreement', 'tenancy agreement']
        }

        for contract_type, keywords in type_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type

        return 'Other Agreement'

    def extract_key_clauses(self, contract_text):
        """Extract key clauses from contract"""
        clauses = []
        clause_patterns = {
            'Confidentiality': ['confidential', 'non-disclosure', 'proprietary information'],
            'Termination': ['termination', 'term of agreement', 'duration'],
            'Payment': ['payment', 'compensation', 'fees', 'price'],
            'Liability': ['liability', 'indemnification', 'warranty'],
            'Intellectual Property': ['intellectual property', 'ip', 'copyright', 'patent'],
            'Governing Law': ['governing law', 'jurisdiction', 'dispute resolution']
        }

        for clause_type, keywords in clause_patterns.items():
            if any(keyword in contract_text.lower() for keyword in keywords):
                clauses.append(clause_type)

        return clauses[:6]

    def save_contract_to_astra(self, contract_name, contract_data, metadata, analysis_data=None):
        """Save complete contract to AstraDB with size optimization and valid schema naming"""
        try:
            # FIXED: Handle file names with extensions like contract3.pdf
            base_name_without_ext = os.path.splitext(contract_name)[0]

            # FIXED: Sanitize name strictly (replace spaces/symbols with _, only alphanumeric allowed)
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name_without_ext).lower()

            # Create valid collection name
            collection_name = f"contract_{clean_name}"

            enhanced_metadata = {
                **metadata,
                "contract_name": contract_name, # Keep original name for display
                "upload_time": datetime.now().isoformat(),
                "collection_name": collection_name,
                "revision": metadata.get("revision", 0),
                "needs_revision": metadata.get("needs_revision", False),
                "risk_score": metadata.get("risk_score", 0),
                "revision_data": metadata.get("revision_data", ""),
                "prev_contract_data": metadata.get("prev_contract_data", ""),
                "revision_details": metadata.get("revision_details", [])
            }

            success = self.database.astra_db.save_contract_data(
                collection_name,
                contract_data,
                enhanced_metadata,
                analysis_data
            )

            if success:
                # Use cleaned name title for dictionary key to match collection
                contract_key = clean_name.replace('_', ' ').title()

                self.global_contracts.contracts[contract_key] = {
                    'collection_name': collection_name,
                    'data': contract_data,
                    'metadata': enhanced_metadata,
                    'status': {
                        'needs_revision': False,
                        'revision_status': 'Current',
                        'updation_status': 'Up to date',
                        'risk_score': 0,
                        'last_updated': datetime.now().isoformat()
                    },
                    'revision_history': [],
                    'analysis_data': analysis_data,
                    'revision_data': "",
                    'prev_contract_data': "",
                    'revision_details': []
                }

                st.success(f"✅ Contract '{contract_name}' saved to AstraDB collection: {collection_name}")
                
                # Display LLM-extracted metadata
                if 'parties' in enhanced_metadata and enhanced_metadata['parties']:
                    st.info(f"🤖 LLM-extracted Parties: {', '.join(enhanced_metadata['parties'])}")
                if 'sign_date' in enhanced_metadata:
                    st.info(f"📅 Sign Date: {enhanced_metadata['sign_date']}")
                if 'termination_date' in enhanced_metadata:
                    st.info(f"⏰ Termination Date: {enhanced_metadata['termination_date']}")
                if 'days_left' in enhanced_metadata:
                    st.info(f"📆 Days Left: {enhanced_metadata['days_left']} days")
                
                return True
            else:
                st.error(f"❌ Failed to save contract '{contract_name}' to AstraDB")
                return False

        except Exception as e:
            st.error(f"Error saving contract to AstraDB: {str(e)}")
            return False

    def assess_revision_impact_simple(self, revision_text, revision_type, llm):
        """Smart revision impact assessment with duplicate revision check"""
        affected_contracts = []

        revision_standard = self.name_manager.detect_compliance_standard(revision_text)
        st.info(f"🔍 Detected Revision Standard: **{revision_standard}**")

        for contract_name, contract_info in self.global_contracts.contracts.items():
            contract_metadata = contract_info.get('metadata', {})
            contract_data = contract_info.get('data', '')
            existing_revision_data = contract_info.get('revision_data', '')

            # Skip if contract doesn't have data
            if not contract_data:
                continue

            # ENHANCED LOGIC: Check if contract already has the same revision data
            if existing_revision_data and self.is_same_revision(existing_revision_data, revision_text):
                st.info(f"ℹ️ Contract '{contract_name}' already has this revision data. Skipping assessment.")
                continue

            # Use LLM to determine if revision is needed and get risk score
            revision_result = self.assess_revision_need_with_score(
                contract_data, revision_text, revision_type, llm
            )

            needs_revision = revision_result['needs_revision']
            risk_score = revision_result['risk_score']

            # Apply logic: if needs_revision and score > 60%, mark for revision
            if needs_revision and risk_score > 60:
                # Update contract status in global object
                self.global_contracts.update_contract_status(
                    contract_name,
                    needs_revision=True,
                    revision_status=f"Needs {revision_type} Update",
                    updation_status="Pending Revision",
                    risk_score=risk_score
                )

                # --- CRITICAL IMPROVEMENT: IMMEDIATELY SAVE REVISION DATA TO DB ---
                self.update_contract_revision_data(contract_name, revision_text, risk_score, preserve_metadata=True)

                # --- NEW FEATURE: EMAIL ALERT FOR HIGH RISK ---
                owner_email = contract_metadata.get('owner_email')
                if owner_email:
                    subject = f"⚠️ ACTION REQUIRED: Contract Revision Alert - {contract_name}"
                    body = f"""Dear Contract Owner,

Your contract '{contract_name}' has been flagged for immediate revision due to a new {revision_type} regulatory update.

Analysis Details:
- Detected Standard: {revision_standard}
- Risk Score: {risk_score}%
- Status: Needs Revision

Please log in to the Enhanced Compliance System to review and update the contract using the automated revision tools.

Regards,
Automated Compliance Monitoring System"""
                    self.email_system.send_alert(owner_email, subject, body)

                affected_contracts.append({
                    'name': contract_name,
                    'revision_type': revision_type,
                    'risk_score': risk_score,
                    'risk_level': self.get_risk_level(risk_score),
                    'matching_standard': revision_standard
                })

        # Update sidebar with number of contracts needing revision
        if hasattr(st, 'session_state'):
            st.session_state.need_revision_count = len(affected_contracts)

        return affected_contracts

    def is_same_revision(self, existing_revision, new_revision):
        """Enhanced check if the new revision is the same as existing revision data"""
        # Simple exact match
        if existing_revision.strip() == new_revision.strip():
            return True

        # Remove whitespace and compare
        existing_clean = re.sub(r'\s+', ' ', existing_revision.strip())
        new_clean = re.sub(r'\s+', ' ', new_revision.strip())
        if existing_clean == new_clean:
            return True

        # Check for substantial similarity (80% word match)
        existing_words = set(existing_clean.lower().split())
        new_words = set(new_clean.lower().split())
        
        if not existing_words or not new_words:
            return False
            
        common_words = existing_words.intersection(new_words)
        similarity = len(common_words) / max(len(existing_words), len(new_words))

        return similarity > 0.8

    def update_contract_revision_data(self, contract_name, revision_text, risk_score, preserve_metadata=True):
        """Update contract with revision data immediately in the database while preserving original metadata"""
        try:
            collection_name = self.global_contracts.contracts[contract_name]['collection_name']

            contract_data, metadata, analysis_data, revision_data, prev_contract_data, revision_details = self.database.astra_db.get_contract_data(collection_name)

            if metadata:
                # Preserve original metadata (parties and sign date) during revision updates
                original_parties = metadata.get('parties', [])
                original_sign_date = metadata.get('sign_date')
                original_contract_date = metadata.get('contract_date')
                original_termination_date = metadata.get('termination_date')
                original_days_left = metadata.get('days_left')
                
                metadata['needs_revision'] = True
                metadata['risk_score'] = risk_score
                metadata['last_updated'] = datetime.now().isoformat()
                metadata['revision_data'] = revision_text  # Save uploaded revision
                
                # Restore original metadata if preserve_metadata is True
                if preserve_metadata:
                    if original_parties:
                        metadata['parties'] = original_parties
                    if original_sign_date:
                        metadata['sign_date'] = original_sign_date
                    if original_contract_date:
                        metadata['contract_date'] = original_contract_date
                    if original_termination_date:
                        metadata['termination_date'] = original_termination_date
                    if original_days_left is not None:
                        metadata['days_left'] = original_days_left

                # Persist to AstraDB
                success = self.database.astra_db.update_contract_revision(
                    collection_name,
                    contract_data,
                    metadata,
                    analysis_data
                )

                if success:
                    # Update global contract object
                    if contract_name in self.global_contracts.contracts:
                        self.global_contracts.contracts[contract_name]['metadata'] = metadata
                        self.global_contracts.contracts[contract_name]['revision_data'] = revision_text
                        self.global_contracts.contracts[contract_name]['status'].update({
                            'needs_revision': True,
                            'risk_score': risk_score
                        })
                else:
                    st.error(f"❌ Failed to mark contract '{contract_name}' for revision in DB")

        except Exception as e:
            st.error(f"Error updating contract revision data: {str(e)}")

    def assess_revision_need_with_score(self, contract_data, revision_text, revision_type, llm):
        """Use LLM to determine if contract needs revision and get risk score"""
        if not contract_data or not llm:
            return {'needs_revision': False, 'risk_score': 0}

        try:
            # New prompt format that requests specific response format without JSON
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal compliance expert.
                 Analyze if the contract needs revision based on the regulatory update.

    Return your response in EXACTLY this format:
    Need Revision: "YES" or "NO"
    Risk score: "score in percent (e.g. 85%)"

    Do not include any other text, explanations, or formatting in your response."""),
    ("human", """CONTRACT CONTENT:
    {contract_data}

    REGULATORY REVISION:
    {revision_text}

    Provide your assessment:""")
            ])

            assessment_chain = prompt | llm | StrOutputParser()
            assessment = assessment_chain.invoke({
                "contract_data": contract_data[:8000],
                "revision_text": revision_text
            })

            # Parse the response
            return self.parse_llm_response(assessment)

        except Exception as e:
            st.error(f"Error in LLM revision assessment: {str(e)}")
            return {'needs_revision': False, 'risk_score': 0}

    def parse_llm_response(self, response):
        """Parse LLM assessment response"""
        try:
            lines = response.strip().split('\n')
            needs_revision = False
            risk_score = 0
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('need revision:'):
                    if 'yes' in line.lower():
                        needs_revision = True
                elif line.lower().startswith('risk score:'):
                    # Extract number from string
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        risk_score = int(numbers[0])
            
            return {'needs_revision': needs_revision, 'risk_score': risk_score}
        except:
            return {'needs_revision': False, 'risk_score': 0}

    def get_risk_level(self, risk_score):
        """Determine risk level based on score"""
        if risk_score < 30:
            return "Low"
        elif risk_score <= 60:
            return "Medium"
        else:
            return "High"

    def update_contract_with_revision(self, contract_name, updated_data, revision_type, revision_details=None):
        """Update contract with revised data - WITH PRODUCTION EMAIL AND METADATA PRESERVATION"""
        try:
            collection_name = self.global_contracts.contracts[contract_name]['collection_name']

            contract_data, metadata, analysis_data, revision_data, prev_contract_data, current_revision_details = self.database.astra_db.get_contract_data(collection_name)

            if metadata:
                # PRESERVE ORIGINAL METADATA
                original_parties = metadata.get('parties', [])
                original_sign_date = metadata.get('sign_date')
                original_contract_date = metadata.get('contract_date')
                original_owner_email = metadata.get('owner_email')
                original_termination_date = metadata.get('termination_date')
                original_days_left = metadata.get('days_left')
                
                current_revision = metadata.get('revision', 0)
                metadata['revision'] = current_revision + 1
                metadata['needs_revision'] = False
                metadata['risk_score'] = 0
                metadata['last_updated'] = datetime.now().isoformat()
                metadata['last_revision_type'] = revision_type
                metadata['prev_contract_data'] = contract_data  # Save current as previous
                # # Clear revision_data as it is now incorporated
                # metadata['revision_data'] = ""
                
                # RESTORE ORIGINAL METADATA
                if original_parties:
                    metadata['parties'] = original_parties
                if original_sign_date:
                    metadata['sign_date'] = original_sign_date
                if original_contract_date:
                    metadata['contract_date'] = original_contract_date
                if original_owner_email:
                    metadata['owner_email'] = original_owner_email
                if original_termination_date:
                    metadata['termination_date'] = original_termination_date
                if original_days_left is not None:
                    metadata['days_left'] = original_days_left

                # Add revision history
                if 'revision_history' not in metadata:
                    metadata['revision_history'] = []

                revision_record = {
                    'revision_number': current_revision + 1,
                    'revision_type': revision_type,
                    'revision_date': datetime.now().isoformat(),
                    'revision_details': revision_details,
                    'changes_made': revision_details.get('changes_summary', []) if revision_details else []
                }
                metadata['revision_history'].append(revision_record)

                # Update revision details
                if not current_revision_details:
                    current_revision_details = []
                current_revision_details.append(revision_record)

                success = self.database.astra_db.update_contract_revision(
                    collection_name,
                    updated_data,
                    metadata,
                    analysis_data,
                    current_revision_details
                )

                if success:
                    # Update global contract object with preserved metadata
                    self.global_contracts.update_contract_data(contract_name, updated_data, metadata)
                    self.global_contracts.add_revision_history(contract_name, revision_record)

                    # Update additional data in global object
                    self.global_contracts.contracts[contract_name]['prev_contract_data'] = contract_data
                    self.global_contracts.contracts[contract_name]['revision_details'] = current_revision_details
                    self.global_contracts.contracts[contract_name]['revision_data'] = ""

                    # Send confirmation email
                    owner_email = metadata.get('owner_email')
                    if owner_email:
                        subject = f"✅ Contract Successfully Revised - {contract_name}"
                        body = f"""Dear Contract Owner,

Your contract '{contract_name}' has been successfully revised and updated in the system.

Revision Details:
- Revision Number: {current_revision + 1}
- Revision Type: {revision_type}
- Revision Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Status: Current and Compliant

The contract is now up to date with the latest regulatory requirements.

Regards,
Automated Compliance Monitoring System"""

                        # Create PDF attachments
                        pdf_attachments = {}
                        
                        # 1. Revised Contract PDF
                        revised_contract_pdf = create_pdf_from_text(updated_data, f"Revised_Contract_{contract_name}")
                        pdf_attachments[f"Revised_Contract_{contract_name}.pdf"] = revised_contract_pdf
                        
                        # 2. Revision Details PDF (if available)
                        if revision_details and isinstance(revision_details, dict):
                            revisions_text = revision_details.get('revisions_list', '')
                            if revisions_text:
                                revision_details_pdf = create_revision_pdf(revisions_text, contract_name)
                                pdf_attachments[f"Revision_Details_{contract_name}.pdf"] = revision_details_pdf
                        elif revision_details:  # If it's already text
                            revision_details_pdf = create_revision_pdf(str(revision_details), contract_name)
                            pdf_attachments[f"Revision_Details_{contract_name}.pdf"] = revision_details_pdf
                        

                        self.email_system.send_alert(owner_email, subject, body, pdf_attachments=pdf_attachments)

                    st.success(f"✅ Contract '{contract_name}' updated to revision {metadata['revision']}")
                    st.info(f"📋 Original metadata preserved: {len(original_parties)} parties, Sign date: {original_sign_date}")
                    return True
                else:
                    st.error(f"❌ Failed to update contract '{contract_name}'")
                    return False

            return False

        except Exception as e:
            st.error(f"Error updating contract: {str(e)}")
            return False

    def get_contracts_for_updation(self):
        """Get contracts that need updation from global object"""
        # Ensure we filter out any that might have been updated in session but not refreshed
        all_contracts = self.global_contracts.contracts
        needing_update = {}
        for name, data in all_contracts.items():
            if data['status']['needs_revision']:
                needing_update[name] = data
        return needing_update

    def generate_llm_based_revision(self, contract_data, revision_context, llm):
        """Generate LLM-based contract revision with detailed change tracking"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal expert specializing in contract compliance.
                 Your task is to revise the given contract to ensure it complies with the new regulatory requirements.

INSTRUCTIONS:
1. Carefully analyze the original contract and the regulatory revision context
2. Update ONLY the necessary clauses to ensure compliance
3. Maintain the original contract structure, format, and business intent
4. Preserve all parties, dates, and commercial terms unless they conflict with regulations
5. Clearly indicate any changes made for compliance

IMPORTANT: Return your response in TWO CLEAR SECTIONS:

SECTION 1 - COMPLETE REVISED CONTRACT:
[Provide the complete revised contract text here]

SECTION 2 - LIST OF REVISIONS:
[Provide a numbered list of all changes made. For each change, follow this format:
 - Nature of change (Added/Modified/Removed): [Description]
]

Return both sections clearly separated."""),
                ("human", """ORIGINAL CONTRACT:
{contract_data}

REGULATORY REVISION CONTEXT:
{revision_context}

Please provide the revised contract and list of revisions:""")
            ])

            revision_chain = prompt | llm | StrOutputParser()
            full_response = revision_chain.invoke({
                "contract_data": contract_data[:12000],
                "revision_context": revision_context[:6000]
            })

            # Parse the response to separate revised contract and revisions list
            return self.parse_revision_response(full_response)

        except Exception as e:
            st.error(f"Error generating LLM-based revision: {str(e)}")
            return None, None

    def parse_revision_response(self, response_text):
        """Parse the LLM response to separate revised contract and revisions list"""
        try:
            # Split the response into sections
            sections = response_text.split('SECTION 2 - LIST OF REVISIONS:')

            if len(sections) == 2:
                revised_contract_section = sections[0].replace('SECTION 1 - COMPLETE REVISED CONTRACT:', '').strip()
                revisions_section = sections[1].strip()

                # Further clean up if there are other markers
                revised_contract = revised_contract_section.split('SECTION 2')[0].strip() if 'SECTION 2' in revised_contract_section else revised_contract_section

                return revised_contract, revisions_section
            else:
                # Fallback: try to split by common patterns
                if 'LIST OF REVISIONS:' in response_text:
                    parts = response_text.split('LIST OF REVISIONS:')
                    return parts[0].replace('COMPLETE REVISED CONTRACT:', '').strip(), parts[1].strip()
                else:
                    # If no clear separation, return entire text as contract and empty revisions
                    return response_text.strip(), "No detailed revisions list provided."

        except Exception as e:
            st.error(f"Error parsing revision response: {str(e)}")
            return response_text.strip(), "Error parsing revisions list."

    def add_contract(self, contract_text, source_info, analysis_data=None, llm=None):
        """Add a new contract with LLM-based metadata extraction"""
        contract_name = source_info.get("name", f"Contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        owner_email = source_info.get("owner_email", "")

        # Use LLM for metadata extraction if available
        if llm:
            metadata = self.extract_contract_metadata_with_llm(contract_text, contract_name, owner_email, llm)
        else:
            metadata = self.extract_contract_metadata_fallback(contract_text, contract_name, owner_email)

        contract_record = {
            "id": f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": contract_name,
            "text": contract_text,
            "source_info": source_info,
            "upload_time": datetime.now().isoformat(),
            **metadata
        }

        self.save_contract_to_astra(contract_name, contract_text, metadata, analysis_data)

        return contract_name

    def refresh_contract_statuses(self):
        """Refresh contract statuses and recalculate days left"""
        contracts = self.global_contracts.contracts
        needs_save = False

        for contract_name, contract in contracts.items():
            metadata = contract.get('metadata', {})
            termination_date_str = metadata.get('termination_date')
            sign_date_str = metadata.get('sign_date')
            
            # Only recalculate if we have LLM-extracted dates
            if sign_date_str and sign_date_str != 'Unknown':
                try:
                    sign_date = datetime.fromisoformat(sign_date_str)
                    
                    # If termination date is unknown, use 180 days as fallback
                    if termination_date_str == 'Unknown' or not termination_date_str:
                        termination_date = sign_date + timedelta(days=180)
                        days_left = (termination_date - datetime.now()).days
                        
                        metadata['termination_date'] = termination_date.isoformat()
                        metadata['days_left'] = max(0, days_left)
                        metadata['status'] = "Active" if days_left > 0 else "Expired"
                        metadata['needs_update'] = days_left <= 30
                        
                        needs_save = True
                    
                except Exception as e:
                    st.error(f"Error recalculating dates for {contract_name}: {str(e)}")

        if needs_save:
            for contract_name, contract in contracts.items():
                metadata = contract.get('metadata', {})
                self.save_contract_to_astra(contract_name, contract.get('data', ''), metadata)

    def cleanup_session_data(self):
        """Clean up any temporary session data"""
        keys_to_clear = [
            'contract_saved_locally', 'temp_contract_name',
            'original_contract_content', 'temp_source_info', 'analysis_report',
            'modified_contract', 'extracted_metadata', 'regulatory_focus_used'
        ]
        for key in keys_to_clear:
            if hasattr(st.session_state, key):
                delattr(st.session_state, key)

    def delete_contract_from_database(self, contract_name):
        """Delete contract from database and global manager"""
        try:
            if contract_name in self.global_contracts.contracts:
                collection_name = self.global_contracts.contracts[contract_name]['collection_name']

                # Delete from AstraDB
                success = self.database.astra_db.delete_collection(collection_name)

                if success:
                    # Delete from global manager
                    self.global_contracts.delete_contract(contract_name)
                    st.success(f"✅ Contract '{contract_name}' deleted successfully!")
                    # Clear cache to ensure UI updates
                    st.cache_data.clear()
                    return True
                else:
                    st.error(f"❌ Failed to delete contract '{contract_name}' from database")
                    return False
            else:
                st.error(f"❌ Contract '{contract_name}' not found")
                return False
        except Exception as e:
            st.error(f"Error deleting contract: {str(e)}")
            return False

# -----------------------------------------------------------------------------
# Compliance Name Manager
# -----------------------------------------------------------------------------
class ComplianceNameManager:
    @staticmethod
    def detect_compliance_standard(text):
        """Detect which compliance standard the text belongs to"""
        text_lower = text.lower()

        standards_keywords = {
            "GDPR": ["gdpr", "general data protection regulation", "eu data protection", "data subject", "european union"],
            "DPDPA": ["dpdpa", "digital personal data protection act", "india data protection", "data principal", "data fiduciary"],
            "HIPAA": ["hipaa", "health insurance portability", "protected health information", "phi", "medical records"],
            "CCPA": ["ccpa", "california consumer privacy act", "california privacy", "consumer rights"],
            "SOX": ["sox", "sarbanes-oxley", "financial reporting", "internal controls", "corporate governance"],
            "PCI_DSS": ["pci dss", "payment card industry", "cardholder data", "payment security"]
        }

        standard_scores = {}
        for standard, keywords in standards_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                standard_scores[standard] = score

        if standard_scores:
            return max(standard_scores.items(), key=lambda x: x[1])[0]

        return "GENERAL_COMPLIANCE"

    @staticmethod
    def create_valid_collection_name(text, source_name=""):
        """Create a valid AstraDB collection name from text"""
        main_standard = ComplianceNameManager.detect_compliance_standard(text)
        base_name = main_standard.lower()
        timestamp = datetime.now().strftime("%Y%m%d")
        collection_name = f"comp_{base_name}_{timestamp}"
        collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
        return collection_name, main_standard

# -----------------------------------------------------------------------------
# Contract Database (AstraDB Only - No Local Storage)
# -----------------------------------------------------------------------------
class ContractDatabase:
    def __init__(self, config, astra_db):
        self.config = config
        self.astra_db = astra_db
        self.document_processor = DocumentProcessor()

    def save_contract(self, contract_name, contract_data):
        """Save contract to AstraDB only"""
        try:
            metadata = contract_data.get('metadata', {})
            contract_text = contract_data.get('text', '')
            analysis_data = contract_data.get('analysis_data')

            # Use ContractManager's save logic if possible, but here we construct raw call
            # Handle name cleaning
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(contract_name)[0]).lower()

            success = self.astra_db.save_contract_data(
                f"contract_{clean_name}",
                contract_text,
                metadata,
                analysis_data
            )
            return success
        except Exception as e:
            st.error(f"Error saving contract {contract_name}: {str(e)}")
            return False

    def list_contracts(self):
        """List all contracts from AstraDB"""
        try:
            collections = self.astra_db.get_all_contract_collections()
            return [col.replace('contract_', '').replace('_', ' ').title() for col in collections]
        except Exception as e:
            st.error(f"Error listing contracts: {str(e)}")
            return []

    def get_all_contracts(self):
        """Get all contracts with their data from AstraDB"""
        contracts = {}
        try:
            collections = self.astra_db.get_all_contract_collections()
            for collection_name in collections:
                contract_name = collection_name.replace('contract_', '', '').replace('_', ' ').title()
                contract_data, metadata, analysis_data, revision_data, prev_contract_data, revision_details = self.astra_db.get_contract_data(collection_name)

                if contract_data:
                    contracts[contract_name] = {
                        'data': contract_data,
                        'metadata': metadata,
                        'analysis_data': analysis_data,
                        'revision_data': revision_data,
                        'prev_contract_data': prev_contract_data,
                        'revision_details': revision_details
                    }
            return contracts
        except Exception as e:
            st.error(f"Error getting all contracts: {str(e)}")
            return {}

# -----------------------------------------------------------------------------
# Compliance Analysis System
# -----------------------------------------------------------------------------
class ComplianceAnalysisSystem:
    def __init__(self, astra_vector_store, llm):
        self.astra_vector_store = astra_vector_store
        self.llm = llm

    def parse_analysis_result(self, analysis_result):
        """Parse the analysis result into analysis report and modified contract"""
        try:
            lines = analysis_result.split('\n')
            analysis_lines = []
            contract_lines = []
            in_contract_section = False

            for line in lines:
                if 'REVISED ORIGINAL CONTRACT' in line.upper() or 'MODIFIED CONTRACT' in line.upper():
                    in_contract_section = True
                    continue

                if in_contract_section:
                    contract_lines.append(line)
                else:
                    analysis_lines.append(line)

            analysis_report = '\n'.join(analysis_lines).strip()
            modified_contract = '\n'.join(contract_lines).strip()

            if not modified_contract:
                modified_contract = None

            return analysis_report, modified_contract

        except Exception as e:
            st.error(f"Error parsing analysis result: {str(e)}")
            return analysis_result, None

    def comprehensive_compliance_analysis(self, contract_text, compliance_docs, regulatory_focus, contract_metadata, llm):
        """Perform comprehensive compliance analysis with LLM metadata extraction"""
        try:
            compliance_context = self.format_compliance_docs(compliance_docs)
            regulatory_context = ", ".join(regulatory_focus)

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a senior compliance analyst and legal expert.
                 Perform a compliance analysis for given contract with given compliance context.

COMPLIANCE CONTEXT FROM DATABASE:
{compliance_context}

REGULATORY FOCUS:
{regulatory_context}

ANALYSIS INSTRUCTIONS:

PART 1 - COMPLIANCE ANALYSIS REPORT:
1. KEY CLAUSES IDENTIFIED: List all important clauses with compliance relevance
2. SPECIFIC ISSUES & VIOLATIONS IF exist: Exact compliance violations with references
3. RECTIFIED CLAUSE SUGGESTIONS: Corrected versions of problematic clauses
4. Brief summary: to compare for revision

PART 2 - MODIFIED CONTRACT:
6. If changes are needed, provide the complete updated contract text
   - If no changes are needed, return "NO CHANGES NEEDED"
   - Maintain the contract structure and format
   - Only give contract content no extra header and footer

Return the response text in two clear sections:
"COMPLIANCE ANALYSIS REPORT:" and "Complete MODIFIED CONTRACT:"
"""),
                ("human", """Contract:\n{contract_text}""")
            ])

            analysis_chain = prompt | self.llm | StrOutputParser()
            full_result = analysis_chain.invoke({"contract_text": contract_text})

            # Extract metadata using LLM as part of analysis
            extracted_metadata = self.extract_metadata_with_llm(contract_text, llm)
            
            analysis_report, modified_contract = self.parse_analysis_result(full_result)
            
            return analysis_report, modified_contract, extracted_metadata

        except Exception as e:
            return f"Error in comprehensive compliance analysis: {str(e)}", None, {}

    def extract_metadata_with_llm(self, contract_text, llm):
        """Extract metadata from contract using LLM with textual output"""
        try:
            extractor = LLMMetadataExtractor()
            metadata = extractor.extract_with_llm(contract_text, llm)
            
            if metadata:
                # Add additional fields
                metadata['contract_size'] = len(contract_text)
                metadata['word_count'] = len(contract_text.split())
                
                # Ensure all required fields exist
                if 'parties' not in metadata:
                    metadata['parties'] = []
                if 'sign_date' not in metadata:
                    metadata['sign_date'] = 'Unknown'
                if 'contract_type' not in metadata:
                    metadata['contract_type'] = 'Unknown'
                if 'termination_date' not in metadata:
                    metadata['termination_date'] = 'Unknown'
                if 'days_left' not in metadata:
                    metadata['days_left'] = 'Unknown'
                if 'key_clauses' not in metadata:
                    metadata['key_clauses'] = []
                
                return metadata
            else:
                return {}
            
        except Exception as e:
            st.error(f"Error extracting metadata with LLM: {str(e)}")
            return {}

    def format_compliance_docs(self, docs):
        """Format compliance documents for context"""
        if not docs:
            return "No specific compliance documents retrieved from database."

        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")

        return "\n\n".join(formatted_docs)

# -----------------------------------------------------------------------------
# Smart Contract Chatbot with RAG
# -----------------------------------------------------------------------------
class ContractChatbot:
    def __init__(self, contract_manager, llm, astra_vector_store):
        self.contract_manager = contract_manager
        self.llm = llm
        self.astra_vector_store = astra_vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def get_contract_context(self, contract_name):
        """Get complete contract context for RAG"""
        try:
            if contract_name in self.contract_manager.global_contracts.contracts:
                contract_info = self.contract_manager.global_contracts.contracts[contract_name]
                contract_data = contract_info.get('data', '')
                metadata = contract_info.get('metadata', {})
                analysis_data = contract_info.get('analysis_data', '')

                # Create comprehensive context
                context_parts = []

                # Add contract content
                if contract_data:
                    context_parts.append(f"CONTRACT CONTENT:\n{contract_data}")

                # Add metadata
                if metadata:
                    context_parts.append(f"CONTRACT METADATA:\n{json.dumps(metadata, indent=2)}")

                # Add analysis data if available
                if analysis_data and analysis_data != "No analysis performed - quick save":
                    context_parts.append(f"COMPLIANCE ANALYSIS:\n{analysis_data}")

                return "\n\n".join(context_parts)
            return None
        except Exception as e:
            st.error(f"Error getting contract context: {str(e)}")
            return None

    def generate_response(self, contract_name, question, chat_history):
        """Generate response using RAG with contract context"""
        if not self.llm:
            return "LLM not available for chatbot."

        try:
            # Get contract context
            contract_context = self.get_contract_context(contract_name)
            if not contract_context:
                return "Contract data not available for chatbot."

            # Prepare conversation history
            history_text = ""
            for msg in chat_history[-6:]:  # Last 6 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

            # Create RAG prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a specialized legal assistant for contract analysis.
                Use the provided contract context to answer questions accurately and helpfully.

CONTRACT CONTEXT:
{contract_context}

CONVERSATION HISTORY:
{history}

INSTRUCTIONS:
1. Answer based ONLY on the contract context provided
2. If information is not in the context, say so clearly
3. Be precise and factual about contract details
4. Help users understand clauses, risks, and compliance issues
5. Format responses clearly with bullet points when helpful

Current Question: {question}"""),
                ("human", "Please answer the question based on the contract context.")
            ])

            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "contract_context": contract_context[:12000],  # Limit context size
                "history": history_text,
                "question": question
            })

            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

# -----------------------------------------------------------------------------
# Caching & Initialization Helpers
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Configuration...", ttl=3600)
def get_config_manager():
    """Instantiate and cache ConfigManager."""
    return ConfigManager()

@st.cache_resource(show_spinner="Initializing AstraDB Manager...", ttl=3600)
def get_astra_db_manager(_config_manager):
    """Instantiate and cache AstraDBManager."""
    return AstraDBManager(_config_manager.astra_token, _config_manager.astra_db_id)

@st.cache_resource(show_spinner="Initializing Contract Database...", ttl=3600)
def get_contract_database(_config_manager, _astra_db_manager):
    """Instantiate and cache ContractDatabase."""
    return ContractDatabase(_config_manager, _astra_db_manager)

@st.cache_resource(show_spinner="Initializing Contract Manager...", ttl=3600)
def get_contract_manager(_contract_database):
    """Instantiate and cache ContractManager."""
    # Note: global_contracts is passed from session_state in main
    return ContractManager(_contract_database, global_contracts)

@st.cache_resource(show_spinner="Initializing Embedding Model...", ttl=3600)
def get_embeddings_model():
    """Instantiate and cache HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Connecting to Vector Store...", ttl=3600)
def get_vector_store(_astra_db_manager, _embeddings):
    """Initialize and cache the Astra Vector Store."""
    if not _astra_db_manager.token or not _astra_db_manager.db_id:
        st.error("Vector Store: AstraDB credentials not set. Cannot initialize vector store.")
        return None
    compliance_collection_name = "compliance_vector_store"
    try:
        astra_vector_store = Cassandra(
            embedding=_embeddings,
            session=None,
            keyspace=None,
            table_name=compliance_collection_name
        )
        return astra_vector_store
    except Exception as e:
        st.error(f"Error initializing Astra Vector Store: {str(e)}")
        return None

# LLM Initialization
def get_groq_api():
    try:
        config = get_config_manager()
        api_key = config.groq_api_key
        if not api_key:
            api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
        return api_key
    except Exception as e:
        st.error(f"Error getting Groq API key: {str(e)}")
        return None

@st.cache_resource(show_spinner="Initializing LLM (Groq/OpenAI)...", ttl=3600)
def init_llm(_api_key):
    """Initialize and cache the ChatOpenAI client pointing to Groq."""
    try:
        if not _api_key:
            return None
        llm = ChatOpenAI(
            openai_api_key=_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=8000
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# PDF & Chart Generation Functions
# -----------------------------------------------------------------------------
def create_pdf_from_text(content, title="Document"):
    """Create a PDF from text content"""
    buffer = reportlab_io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, textColor=colors.darkblue
    )
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    content_style = ParagraphStyle(
        'CustomContent', parent=styles['Normal'], fontSize=10, spaceAfter=12, leading=14
    )
    paragraphs = content.split('\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), content_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_analysis_pdf(analysis_data, contract_name):
    """Create a PDF for analysis report"""
    buffer = create_pdf_from_text(analysis_data, f"Compliance Analysis Report for {contract_name}")
    return buffer

def create_revision_pdf(revision_data, contract_name):
    """Create a PDF for revision report"""
    buffer = create_pdf_from_text(revision_data, f"Revision Details for {contract_name}")
    return buffer

# Chart Generation Functions
def create_risk_assessment_chart(risk_score, risk_level):
    """Create a gauge chart for risk assessment"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': f"Risk Level: {risk_level}", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'yellow'},
                {'range': [60, 100], 'color': 'red'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_score}
        }))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_compliance_breakdown_chart(analysis_data):
    """Create a compliance breakdown chart from analysis data"""
    if not analysis_data:
        return None

    categories = {
        'Data Protection': 0,
        'Confidentiality': 0,
        'Liability': 0,
        'Termination': 0,
        'Payment': 0,
        'Intellectual Property': 0
    }
    analysis_lower = analysis_data.lower()

    for category in categories:
        if category.lower() in analysis_lower:
            categories[category] = 75
        else:
            categories[category] = 25

    df_breakdown = pd.DataFrame(list(categories.items()), columns=['Category', 'Score'])

    fig = px.bar(
        df_breakdown,
        x='Category',
        y='Score',
        color='Category',
        title="Compliance Category Breakdown",
        height=300
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def create_revision_impact_chart(affected_contracts):
    """Create a chart showing revision impact across contracts"""
    if not affected_contracts:
        return None

    contract_names = [contract['name'] for contract in affected_contracts]
    risk_scores = [contract['risk_score'] for contract in affected_contracts]
    risk_levels = [contract['risk_level'] for contract in affected_contracts]

    # Color mapping for risk levels
    colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    color_sequence = [colors[level] for level in risk_levels]

    fig = px.bar(
        x=contract_names,
        y=risk_scores,
        title="Revision Impact on Contracts",
        labels={'x': 'Contracts', 'y': 'Risk Score (%)'},
        color=risk_levels,
        color_discrete_map=colors
    )

    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        showlegend=True
    )

    return fig

def create_changes_breakdown_chart(revisions_text):
    """Create a breakdown of changes (Added, Modified, Removed) from the revisions list."""
    if not revisions_text or "NO REVISIONS LISTED" in revisions_text.upper():
        return None

    # IMPROVED PARSING for charts
    revisions_upper = revisions_text.upper()
    added = revisions_upper.count('ADDED')
    modified = revisions_upper.count('MODIFIED')
    removed = revisions_upper.count('REMOVED')

    # If no strict keywords found, try lenient counting based on context
    if added == 0 and modified == 0 and removed == 0:
        added = revisions_upper.count('INSERT') + revisions_upper.count('NEW')
        modified = revisions_upper.count('UPDATE') + revisions_upper.count('CHANGE')
        removed = revisions_upper.count('DELETE')

    # Ensure we have at least some data to show
    if added == 0 and modified == 0 and removed == 0:
        return None

    data = {
        'Change Type': ['Added', 'Modified', 'Removed'],
        'Count': [added, modified, removed]
    }
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        values='Count',
        names='Change Type',
        title='Breakdown of Changes Made',
        hole=.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=False
    )
    return fig

# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_text_from_source(upload_option, uploaded_file=None, url=None, direct_text=None):
    """Universal text extraction function"""
    extracted_content = None
    source_info = {}

    if upload_option == "PDF File" and uploaded_file:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            extracted_content = text
            source_info = {"type": "PDF", "name": uploaded_file.name}
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")

    elif upload_option == "URL" and url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            extracted_content = ' '.join(chunk for chunk in chunks if chunk)
            source_info = {"type": "URL", "name": url}
        except Exception as e:
            st.error(f"Error extracting from URL: {str(e)}")

    elif upload_option == "Text File" and uploaded_file:
        try:
            extracted_content = str(uploaded_file.read(), 'utf-8')
            source_info = {"type": "TXT", "name": uploaded_file.name}
        except Exception as e:
            st.error(f"Error extracting text file: {str(e)}")

    elif upload_option == "Direct Text" and direct_text:
        extracted_content = direct_text
        source_info = {"type": "Direct Input", "name": f"Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}

    return extracted_content, source_info

# -----------------------------------------------------------------------------
# Streamlit Rendering Functions
# -----------------------------------------------------------------------------
def styled_header(title, icon):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; display: flex; align-items: center;">
            <span style="font-size: 3rem; margin-right: 15px;">{icon}</span>
            {title}
        </h1>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Advanced Contract Compliance & Monitoring System
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard(contract_manager):
    styled_header("Contract Dashboard", "🏠")

    # Force refresh from DB to get live status
    contracts = contract_manager.initialize_global_contracts(force_refresh=True)
    config = ConfigManager()

    st.subheader("🔧 System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        astra_status = contract_manager.database.astra_db.init_client()
        st.success("✅ **AstraDB**\nConnected" if astra_status else "❌ **AstraDB**\nNot Connected")

    with col2:
        groq_status = contract_manager.config.groq_api_key is not None
        st.success("✅ **Groq API**\nConnected" if groq_status else "❌ **Groq API**\nNot Connected")

    with col3:
        total_contracts = len(contracts)
        st.info(f"📊 **Contracts**\n{total_contracts} Total")

    with col4:
        needs_update = len(contract_manager.global_contracts.get_contracts_needing_revision())
        st.warning(f"⚠️ **Revisions**\n{needs_update} Needed" if needs_update > 0 else "✅ **Revisions**\nAll Current")

    st.subheader("📋 Regulatory Standards")
    st.write(", ".join(config.regulatory_standards))

    st.subheader("📈 Contract Statistics")
    if contracts:
        # Calculate statistics using LLM-extracted metadata
        active_contracts = 0
        expired_contracts = 0
        total_days_left = 0
        contracts_with_days = 0
        
        for contract in contracts.values():
            metadata = contract.get('metadata', {})
            status = metadata.get('status', 'Unknown')
            days_left = metadata.get('days_left', 'Unknown')
            
            if status == 'Active':
                active_contracts += 1
            elif status == 'Expired':
                expired_contracts += 1
            
            if days_left != 'Unknown' and isinstance(days_left, (int, float)):
                total_days_left += days_left
                contracts_with_days += 1

        avg_days_left = total_days_left / contracts_with_days if contracts_with_days > 0 else 0

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Active Contracts", active_contracts)
        stat_col2.metric("Expired Contracts", expired_contracts)
        stat_col3.metric("Avg Days Left", f"{avg_days_left:.1f}")

        db_contract_count = contract_manager.database.astra_db.get_contract_count()
        st.metric("Database Contracts", db_contract_count)

        attention_contracts = contract_manager.global_contracts.get_contracts_needing_revision()
        if attention_contracts:
            st.subheader("🚨 Contracts Needing Revision")
            for contract_name, contract_info in attention_contracts.items():
                risk_score = contract_info['status']['risk_score']
                risk_level = contract_manager.get_risk_level(risk_score)

                if risk_level == "High":
                    risk_color = "🔴"
                elif risk_level == "Medium":
                    risk_color = "🟡"
                else:
                    risk_color = "🟢"

                st.warning(f"{risk_color} **{contract_name}** - Risk: {risk_score}% ({risk_level} Risk)")

    else:
        st.info("ℹ️ No contracts available. Upload contracts to get started.")

def render_contract_chatbot(contract_name, contract_info, llm, contract_manager, astra_vector_store):
    """Enhanced RAG-based chatbot for a specific contract."""
    st.markdown(f"### 💬 Smart Chat with {contract_name}")

    # Initialize chatbot
    chatbot = ContractChatbot(contract_manager, llm, astra_vector_store)

    if not llm:
        st.warning("⚠️ Please configure the LLM to use the chatbot feature.")
        return

    # Initialize chat history for this specific contract
    chat_key = f"chat_history_{contract_name}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Display chat history
    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input(f"Ask about {contract_name}..."):
        # Add user message
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response with RAG
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing contract and generating response..."):
                try:
                    response = chatbot.generate_response(
                        contract_name,
                        prompt,
                        st.session_state[chat_key]
                    )
                    st.markdown(response)
                    st.session_state[chat_key].append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state[chat_key].append({"role": "assistant", "content": error_msg})

    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat History", key=f"clear_{contract_name}"):
            st.session_state[chat_key] = []
            st.rerun()

    with col2:
        if st.button("📋 Sample Questions", key=f"samples_{contract_name}"):
            sample_questions = [
                "What are the key clauses in this contract?",
                "What compliance risks does this contract have?",
                "Summarize the main obligations of the parties",
                "What data protection provisions are included?",
                "Are there any termination clauses?",
                "What is the risk score and why?"
            ]
            st.info("💡 Try asking: " + " | ".join(sample_questions))

def render_global_contracts_dashboard(contract_manager, llm, astra_vector_store):
    """Render dashboard showing global contract status with view contracts and revisions options"""
    st.subheader("🌐 Global Contracts Management")

    if not contract_manager.global_contracts.contracts:
        st.info("No contracts initialized. Please upload contracts first.")
        return

    total_contracts = len(contract_manager.global_contracts.contracts)
    needs_revision = len(contract_manager.global_contracts.get_contracts_needing_revision())
    up_to_date = total_contracts - needs_revision

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contracts", total_contracts)
    col2.metric("Need Revision", needs_revision, delta=f"-{up_to_date}" if needs_revision else None)
    col3.metric("Up to Date", up_to_date)

    # Display contracts with action buttons
    st.subheader("📋 Available Contracts")

    for contract_name, contract_info in contract_manager.global_contracts.contracts.items():
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])

            with col1:
                status = contract_info['status']
                metadata = contract_info.get('metadata', {})

                # Display contract info with LLM-extracted metadata
                st.write(f"### {contract_name}")

                # Display LLM-extracted metadata
                parties = metadata.get('parties', [])
                parties_display = ", ".join(parties) if parties else "No parties extracted"
                
                sign_date = metadata.get('sign_date', 'Unknown')
                termination_date = metadata.get('termination_date', 'Unknown')
                days_left = metadata.get('days_left', 'Unknown')
                contract_type = metadata.get('contract_type', 'Unknown')
                
                if days_left != 'Unknown':
                    days_left_display = f"{days_left} days left"
                else:
                    days_left_display = "Unknown"

                if sign_date != 'Unknown':
                  try:
                      # Try to parse and format the date
                      sign_date_display = datetime.fromisoformat(sign_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                  except:
                      sign_date_display = sign_date.split('T')[0] if 'T' in sign_date else sign_date
                else:
                    sign_date_display = 'Not extracted'

                # Format the termination date
                if termination_date != 'Unknown':
                    try:
                        termination_date_display = datetime.fromisoformat(termination_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                    except:
                        termination_date_display = termination_date.split('T')[0] if 'T' in termination_date else termination_date
                else:
                    termination_date_display = 'Not extracted'
                    
        
                
                st.write(f"**Type:** {contract_type} | "
                        f"**Revision:** {metadata.get('revision', 0)} | "
                        f"**Risk:** {status['risk_score']}% | "
                        f"**Days Left:** {days_left_display}")
                
                st.write(f"**Parties (LLM-extracted):** {parties_display}")
                st.write(f"**Sign Date:** {sign_date_display} | ")
                # st.write(f"**Sign Date:** {sign_date if sign_date != 'Unknown' else 'Not extracted'} | "
                        # f"**Termination Date:** {termination_date if termination_date != 'Unknown' else 'Not extracted'} | "
                st.write(f"**Termination Date:** {termination_date_display} | ")        
                st.write(f"**Owner:** {metadata.get('owner_email', 'N/A')}")

                # Status indicator
                if status['needs_revision']:
                    st.error("⚠️ Needs Revision")
                else:
                    st.success("✅ Up to Date")

            with col2:
                # View Contract button
                if st.button(f"View", key=f"view_{contract_name}"):
                    st.session_state[f'viewing_contract_{contract_name}'] = True
                    st.session_state[f'viewing_revisions_{contract_name}'] = False
                    st.session_state[f'viewing_prev_{contract_name}'] = False
                    st.session_state[f'chatting_{contract_name}'] = False

            with col3:
                # View Previous Version button
                prev_contract_data = contract_info.get('prev_contract_data', '')
                if prev_contract_data:
                    if st.button(f"Prev", key=f"prev_{contract_name}"):
                        st.session_state[f'viewing_prev_{contract_name}'] = True
                        st.session_state[f'viewing_contract_{contract_name}'] = False
                        st.session_state[f'viewing_revisions_{contract_name}'] = False
                        st.session_state[f'chatting_{contract_name}'] = False
                else:
                    st.button(f"No Prev", key=f"no_prev_{contract_name}", disabled=True)

            with col4:
                # View Revisions button
                revision_history = contract_info.get('revision_history', [])
                revision_details = contract_info.get('revision_details', [])
                if revision_history or revision_details:
                    if st.button(f"Revisions", key=f"revisions_{contract_name}"):
                        st.session_state[f'viewing_revisions_{contract_name}'] = True
                        st.session_state[f'viewing_contract_{contract_name}'] = False
                        st.session_state[f'viewing_prev_{contract_name}'] = False
                        st.session_state[f'chatting_{contract_name}'] = False
                else:
                    st.button(f"No Rev", key=f"no_rev_{contract_name}", disabled=True)

            # --- ENHANCED CHAT BUTTON ---
            with col5:
                if st.button(f"💬 Chat", key=f"chat_{contract_name}"):
                    st.session_state[f'chatting_{contract_name}'] = True
                    st.session_state[f'viewing_contract_{contract_name}'] = False
                    st.session_state[f'viewing_revisions_{contract_name}'] = False
                    st.session_state[f'viewing_prev_{contract_name}'] = False

            # --- DELETE CONTRACT BUTTON ---
            with col6:
                if st.button(f"🗑️ Delete", key=f"delete_{contract_name}"):
                    if st.session_state.get(f'confirm_delete_{contract_name}', False):
                        # Second click - confirm deletion
                        success = contract_manager.delete_contract_from_database(contract_name)
                        if success:
                            # Clear the confirmation state
                            st.session_state[f'confirm_delete_{contract_name}'] = False
                            st.rerun()
                    else:
                        # First click - show confirmation
                        st.session_state[f'confirm_delete_{contract_name}'] = True
                        st.warning(f"Click Delete again to confirm deletion of '{contract_name}'")
                        st.rerun()

            # Reset confirmation if user navigates away
            if st.session_state.get(f'confirm_delete_{contract_name}', False):
                if not any([
                    st.session_state.get(f'viewing_contract_{contract_name}', False),
                    st.session_state.get(f'viewing_revisions_{contract_name}', False),
                    st.session_state.get(f'viewing_prev_{contract_name}', False),
                    st.session_state.get(f'chatting_{contract_name}', False)
                ]):
                    st.session_state[f'confirm_delete_{contract_name}'] = False

            # Display current contract content when View is clicked
            if st.session_state.get(f'viewing_contract_{contract_name}', False):
                st.markdown("---")
                st.subheader(f"📄 Current Contract: {contract_name}")

                contract_data = contract_info.get('data', '')
                if contract_data:
                    st.text_area("Contract Content", contract_data, height=300,
                               key=f"contract_content_{contract_name}")

                    # Download as PDF
                    pdf_buffer = create_pdf_from_text(contract_data, f"Contract_{contract_name}")
                    st.download_button(
                        label="📥 Download Contract as PDF",
                        data=pdf_buffer,
                        file_name=f"contract_{contract_name}.pdf",
                        mime="application/pdf",
                        key=f"download_contract_{contract_name}"
                    )

                    # Display LLM-extracted metadata in a nice format
                    with st.expander("📊 Contract Metadata (LLM-extracted)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            parties = metadata.get('parties', [])
                            st.write("**Parties:**", ", ".join(parties) if parties else "No parties extracted")
                            st.write("**Sign Date:**", metadata.get('sign_date', 'Unknown'))
                            st.write("**Contract Type:**", metadata.get('contract_type', 'Unknown'))
                            st.write("**Key Clauses:**", ", ".join(metadata.get('key_clauses', [])) if metadata.get('key_clauses') else 'None identified')
                        with col2:
                            st.write("**Termination Date:**", metadata.get('termination_date', 'Unknown'))
                            st.write("**Days Left:**", f"{metadata.get('days_left', 'Unknown')} days")
                            st.write("**Status:**", metadata.get('status', 'Unknown'))
                            st.write("**Regulatory Standards:**", ", ".join(metadata.get('regulatory_standards', [])))
                            st.write("**Owner Email:**", metadata.get('owner_email', 'N/A'))
                            st.write("**Last Updated:**", status['last_updated'])

                    # Risk assessment chart with UNIQUE KEY
                    risk_score = status['risk_score']
                    risk_level = contract_manager.get_risk_level(risk_score)
                    risk_chart = create_risk_assessment_chart(risk_score, risk_level)
                    st.plotly_chart(risk_chart, use_container_width=True, key=f"global_risk_chart_{contract_name}")

                if st.button("Close View", key=f"close_view_{contract_name}"):
                    st.session_state[f'viewing_contract_{contract_name}'] = False
                    st.rerun()

            # Display previous contract version
            if st.session_state.get(f'viewing_prev_{contract_name}', False):
                st.markdown("---")
                st.subheader(f"📄 Previous Contract Version: {contract_name}")

                prev_contract_data = contract_info.get('prev_contract_data', '')
                if prev_contract_data:
                    st.text_area("Previous Contract Content", prev_contract_data, height=300,
                               key=f"prev_contract_content_{contract_name}")

                    # Download previous version as PDF
                    pdf_buffer = create_pdf_from_text(prev_contract_data, f"Previous_Contract_{contract_name}")
                    st.download_button(
                        label="📥 Download Previous Version as PDF",
                        data=pdf_buffer,
                        file_name=f"previous_contract_{contract_name}.pdf",
                        mime="application/pdf",
                        key=f"download_prev_{contract_name}"
                    )
                else:
                    st.info("No previous version available")

                if st.button("Close Previous View", key=f"close_prev_{contract_name}"):
                    st.session_state[f'viewing_prev_{contract_name}'] = False
                    st.rerun()

            # Display revision history when View Revisions is clicked
            if st.session_state.get(f'viewing_revisions_{contract_name}', False):
                st.markdown("---")
                st.subheader(f"🔄 Revision History: {contract_name}")

                revision_history = contract_info.get('revision_history', [])
                revision_details = contract_info.get('revision_details', [])
                revision_data = contract_info.get('revision_data', '')

                # if revision_data:
                #     st.subheader("📝 Revision Data")
                #     st.text_area("Uploaded Revision Content", revision_data, height=200,
                #                key=f"revision_data_{contract_name}")

                if revision_history or revision_details:
                    # Display each revision in detail
                    all_revisions = revision_details + revision_history
                    for i, revision in enumerate(reversed(all_revisions)):
                        with st.expander(f"Revision {revision.get('revision_number', i+1)} - {revision.get('revision_type', 'Unknown')} - {revision.get('revision_date', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Revision Type:** {revision.get('revision_type', 'Unknown')}")
                                st.write(f"**Revision Date:** {revision.get('revision_date', 'Unknown')}")
                                st.write(f"**Changes Made:** {len(revision.get('changes_made', []))}")

                            with col2:
                                if revision.get('revision_details'):
                                    st.write("**Revision Details:**")
                                    st.json(revision['revision_details'])

                            # Display changes made
                            changes = revision.get('changes_made', [])
                            if changes:
                                st.write("**Changes Summary:**")
                                for change in changes:
                                    st.write(f"- {change}")
                            else:
                                st.info("No detailed changes recorded for this revision.")
                else:
                    st.info("No revision history available for this contract.")

                if st.button("Close Revisions", key=f"close_revisions_{contract_name}"):
                    st.session_state[f'viewing_revisions_{contract_name}'] = False
                    st.rerun()

            # --- ENHANCED CHAT INTERFACE RENDERING ---
            if st.session_state.get(f'chatting_{contract_name}', False):
                st.markdown("---")
                render_contract_chatbot(contract_name, contract_info, llm, contract_manager, astra_vector_store)

                if st.button("Close Chat", key=f"close_chat_{contract_name}"):
                    st.session_state[f'chatting_{contract_name}'] = False
                    st.rerun()

            st.markdown("---")

def render_upload_compliance(contract_manager, astra_vector_store):
    styled_header("Upload Compliance Documents", "📥")

    st.subheader("Upload Compliance Data")

    upload_option = st.radio("Upload method:",
                           ["PDF File", "URL", "Text File", "Direct Text"],
                           horizontal=True, key="compliance_upload")

    extracted_content = None
    source_info = {}

    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose compliance PDF", type=['pdf'], key="compliance_pdf")
        if uploaded_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=uploaded_file)

    elif upload_option == "URL":
        url = st.text_input("Enter compliance URL:", placeholder="https://example.com/compliance.pdf")
        if st.button("Extract Compliance") and url:
            extracted_content, source_info = extract_text_from_source(upload_option, url=url)

    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose compliance text file", type=['txt'], key="compliance_txt")
        if text_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=text_file)

    elif upload_option == "Direct Text":
        direct_text = st.text_area("Paste compliance text:", height=200)
        if direct_text:
            extracted_content, source_info = extract_text_from_source(upload_option, direct_text=direct_text)

    if extracted_content:
        st.success("✅ Compliance content extracted!")

        collection_name, detected_standard = ComplianceNameManager.create_valid_collection_name(
            extracted_content,
            source_info.get('name', '')
        )

        st.info(f"🔍 Detected Compliance Standard: **{detected_standard}**")
        st.info(f"📁 Collection Name: **{collection_name}**")

        with st.expander("Preview Content"):
            st.text(extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content)

        if st.button("💾 Save Compliance Data"):
            with st.spinner("Processing and saving compliance data..."):
                try:
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=800,
                        chunk_overlap=100,
                        length_function=len,
                    )

                    chunks = text_splitter.split_text(extracted_content)

                    metadatas = []
                    for i, chunk in enumerate(chunks):
                        metadata = {
                            "source": source_info["type"],
                            "source_name": source_info.get("name", "Unknown"),
                            "doc_type": "compliance",
                            "compliance_standard": detected_standard,
                            "chunk_id": i,
                            "timestamp": datetime.now().isoformat()
                        }
                        metadatas.append(metadata)

                    if astra_vector_store:
                        astra_vector_store.add_texts(texts=chunks, metadatas=metadatas)
                        st.success(f"✅ Saved {len(chunks)} compliance chunks to vector store!")

                    st.success(f"✅ Compliance data processed for standard: {detected_standard}")

                except Exception as e:
                    st.error(f"Error processing compliance data: {str(e)}")

def render_smart_analysis_tab(contract_manager, astra_vector_store, llm):
    """Render the new Smart Analysis tab"""
    st.subheader("🤖 Smart Analysis - Available Contracts")

    if not contract_manager.global_contracts.contracts:
        st.info("📭 No contracts available for analysis. Please upload contracts first.")
        return

    # Display available contracts with their dates
    st.markdown("### 📋 Available Contracts")

    contract_list = []
    for contract_name, contract_info in contract_manager.global_contracts.contracts.items():
        metadata = contract_info.get('metadata', {})
        sign_date = metadata.get('sign_date', 'Unknown')
        termination_date = metadata.get('termination_date', 'Unknown')
        last_updated = contract_info['status']['last_updated']
        days_left = metadata.get('days_left', 'Unknown')
        parties = metadata.get('parties', [])

        # Format dates for display
        try:
            if sign_date != 'Unknown':
                sign_date = datetime.fromisoformat(sign_date).strftime("%Y-%m-%d")
            if termination_date != 'Unknown':
                termination_date = datetime.fromisoformat(termination_date).strftime("%Y-%m-%d")
            last_updated = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M")
        except:
            pass

        contract_list.append({
            'Contract Name': contract_name,
            'Sign Date': sign_date,
            'Termination Date': termination_date,
            'Parties': ', '.join(parties[:2]) if parties else 'No parties',
            'Days Left': f"{days_left} days" if days_left != 'Unknown' else 'Unknown',
            'Risk Score': f"{contract_info['status']['risk_score']}%"
        })

    df_contracts = pd.DataFrame(contract_list)

    if not df_contracts.empty:
        # Display contracts in a nice table
        st.dataframe(df_contracts, use_container_width=True)

        # Contract selection for analysis
        st.markdown("### 🔍 Select Contract for Analysis")
        selected_contract = st.selectbox(
            "Choose a contract to analyze:",
            options=list(contract_manager.global_contracts.contracts.keys()),
            key="smart_analysis_select"
        )

        if selected_contract:
            contract_info = contract_manager.global_contracts.contracts[selected_contract]
            contract_data = contract_info.get('data', '')
            metadata = contract_info.get('metadata', {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Contract Overview")
                st.write(f"**Type:** {metadata.get('contract_type', 'Unknown')}")
                
                # Display LLM-extracted parties
                parties = metadata.get('parties', [])
                st.write(f"**Parties (LLM-extracted):** {', '.join(parties) if parties else 'No parties extracted'}")
                
                st.write(f"**Sign Date:** {metadata.get('sign_date', 'Unknown')}")
                st.write(f"**Termination Date:** {metadata.get('termination_date', 'Unknown')}")
                st.write(f"**Days Left:** {metadata.get('days_left', 'Unknown')} days")
                st.write(f"**Key Clauses:** {', '.join(metadata.get('key_clauses', []))}")
                st.write(f"**Regulatory Standards:** {', '.join(metadata.get('regulatory_standards', []))}")

                # Current risk assessment
                current_risk = contract_info['status']['risk_score']
                risk_level = contract_manager.get_risk_level(current_risk)

                st.markdown(f"#### 🎯 Current Risk Assessment")
                risk_chart = create_risk_assessment_chart(current_risk, risk_level)
                st.plotly_chart(risk_chart, use_container_width=True, key="smart_current_risk")

            with col2:
                st.markdown("#### ⚙️ Analysis Configuration")
                regulatory_focus = st.multiselect(
                    "Select regulatory frameworks for analysis:",
                    contract_manager.config.regulatory_standards,
                    default=["GDPR", "DPDPA"],
                    key="smart_analysis_frameworks"
                )

                analysis_type = st.radio(
                    "Analysis Type:",
                    ["Comprehensive Compliance", "Risk Assessment", "Clause Analysis"],
                    key="smart_analysis_type"
                )

            # Perform analysis button
            if st.button("🚀 Perform Smart Analysis", key="smart_analyze_btn"):
                if not llm:
                    st.error("❌ LLM not configured for analysis")
                    return

                with st.spinner("Performing smart analysis..."):
                    # Retrieve compliance documents
                    compliance_docs = []
                    for framework in regulatory_focus:
                        try:
                            docs = astra_vector_store.similarity_search(framework, k=2)
                            compliance_docs.extend(docs)
                        except:
                            continue

                    # Perform comprehensive analysis with LLM metadata extraction
                    analysis_system = ComplianceAnalysisSystem(astra_vector_store, llm)
                    analysis_report, modified_contract, extracted_metadata = analysis_system.comprehensive_compliance_analysis(
                        contract_data, compliance_docs, regulatory_focus, metadata, llm
                    )

                    # Merge extracted metadata with existing metadata (preserve original)
                    final_metadata = {**metadata, **extracted_metadata}
                    
                    # Store results in session state
                    st.session_state.smart_analysis_report = analysis_report
                    st.session_state.smart_modified_contract = modified_contract
                    st.session_state.smart_extracted_metadata = final_metadata
                    st.session_state.smart_selected_contract = selected_contract

                    # Save analysis data to database (without revision)
                    if analysis_report:
                        # FIXED: Use cleaned collection name logic
                        base_name_without_ext = os.path.splitext(selected_contract)[0]
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name_without_ext).lower()
                        collection_name = f"contract_{clean_name}"

                        contract_data_to_save = modified_contract if modified_contract and "NO CHANGES NEEDED" not in modified_contract.upper() else contract_data

                        # Update contract with analysis data
                        current_data, current_metadata, current_analysis, current_revision_data, current_prev_data, current_revision_details = contract_manager.database.astra_db.get_contract_data(collection_name)

                        if current_metadata:
                            # Preserve original metadata while updating analysis data
                            current_metadata['analysis_data'] = analysis_report
                            # Merge LLM-extracted metadata but preserve original key fields
                            if extracted_metadata:
                                for key, value in extracted_metadata.items():
                                    if key not in ['parties', 'sign_date', 'contract_date', 'termination_date', 'days_left'] or not current_metadata.get(key):
                                        current_metadata[key] = value
                            
                            success = contract_manager.database.astra_db.update_contract_revision(
                                collection_name,
                                contract_data_to_save,
                                current_metadata,
                                analysis_report
                            )

                            if success:
                                st.success("✅ Analysis data saved to database!")

            # Display analysis results if available
            if hasattr(st.session_state, 'smart_analysis_report') and st.session_state.smart_selected_contract == selected_contract:
                st.markdown("---")
                st.markdown("## 📋 Smart Analysis Results")

                # Display analysis metrics and charts
                col1, col2 = st.columns(2)

                with col1:
                    # Risk assessment from analysis
                    analysis_text = st.session_state.smart_analysis_report.lower()
                    risk_keywords = {
                        'high risk': 80,
                        'medium risk': 50,
                        'low risk': 20,
                        'critical': 90,
                        'severe': 85,
                        'moderate': 45,
                        'minor': 25
                    }

                    analysis_risk_score = 0
                    for keyword, score in risk_keywords.items():
                        if keyword in analysis_text:
                            analysis_risk_score = max(analysis_risk_score, score)

                    if analysis_risk_score == 0:
                        # Default risk calculation based on analysis content
                        if 'non-compliant' in analysis_text or 'violation' in analysis_text:
                            analysis_risk_score = 75
                        elif 'compliant' in analysis_text and 'fully' in analysis_text:
                            analysis_risk_score = 15
                        else:
                            analysis_risk_score = 40

                    risk_level = contract_manager.get_risk_level(analysis_risk_score)
                    st.markdown("#### 🎯 Analysis Risk Assessment")
                    analysis_risk_chart = create_risk_assessment_chart(analysis_risk_score, risk_level)
                    st.plotly_chart(analysis_risk_chart, use_container_width=True, key="smart_analysis_risk_chart")

                with col2:
                    # Compliance breakdown
                    st.markdown("#### 📊 Compliance Breakdown")
                    compliance_chart = create_compliance_breakdown_chart(st.session_state.smart_analysis_report)
                    if compliance_chart:
                        st.plotly_chart(compliance_chart, use_container_width=True, key="smart_compliance_breakdown")

                # Display analysis report
                st.markdown("#### 📝 Detailed Analysis Report")
                st.text_area("Analysis Details", st.session_state.smart_analysis_report, height=400, key="smart_analysis_details")

                # Download analysis as PDF
                analysis_pdf = create_analysis_pdf(st.session_state.smart_analysis_report, selected_contract)
                st.download_button(
                    label="📥 Download Analysis Report as PDF",
                    data=analysis_pdf,
                    file_name=f"analysis_report_{selected_contract}.pdf",
                    mime="application/pdf",
                    key="download_smart_analysis_pdf"
                )

                # Display modified contract if available
                if (st.session_state.smart_modified_contract and
                    "NO CHANGES NEEDED" not in st.session_state.smart_modified_contract.upper()):
                    st.markdown("#### 📄 Updated Contract")
                    st.text_area("Revised Contract", st.session_state.smart_modified_contract, height=400, key="smart_modified_contract")

                    # Download updated contract as PDF
                    contract_pdf = create_pdf_from_text(st.session_state.smart_modified_contract, f"Updated_Contract_{selected_contract}")
                    st.download_button(
                        label="📥 Download Updated Contract as PDF",
                        data=contract_pdf,
                        file_name=f"updated_contract_{selected_contract}.pdf",
                        mime="application/pdf",
                        key="download_smart_updated_pdf"
                    )

                    # Update contract button (this is NOT a revision, just analysis-based update)
                    if st.button("💾 Update Contract with Analysis", key="smart_update_contract"):
                        # This is analysis-based update, not revision
                        base_name_without_ext = os.path.splitext(selected_contract)[0]
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name_without_ext).lower()
                        collection_name = f"contract_{clean_name}"

                        current_data, current_metadata, current_analysis, current_revision_data, current_prev_data, current_revision_details = contract_manager.database.astra_db.get_contract_data(collection_name)

                        if current_metadata:
                            # Update contract data with analysis results (no revision increment)
                            current_metadata['analysis_data'] = st.session_state.smart_analysis_report
                            # Merge LLM-extracted metadata
                            if st.session_state.smart_extracted_metadata:
                                for key, value in st.session_state.smart_extracted_metadata.items():
                                    if key not in ['parties', 'sign_date', 'contract_date', 'termination_date', 'days_left'] or not current_metadata.get(key):
                                        current_metadata[key] = value
                                        
                            success = contract_manager.database.astra_db.update_contract_revision(
                                collection_name,
                                st.session_state.smart_modified_contract,
                                current_metadata,
                                st.session_state.smart_analysis_report
                            )

                            if success:
                                st.success("✅ Contract successfully updated with analysis results!")
                                # Clear session state
                                del st.session_state.smart_analysis_report
                                del st.session_state.smart_modified_contract
                                del st.session_state.smart_extracted_metadata
                                st.rerun()

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Analysis Report (TXT)",
                        data=st.session_state.smart_analysis_report,
                        file_name=f"smart_analysis_{selected_contract}.txt",
                        mime="text/plain",
                        key="download_smart_analysis"
                    )

                with col2:
                    if (st.session_state.smart_modified_contract and
                        "NO CHANGES NEEDED" not in st.session_state.smart_modified_contract.upper()):
                        st.download_button(
                            label="📥 Download Updated Contract (TXT)",
                            data=st.session_state.smart_modified_contract,
                            file_name=f"updated_{selected_contract}.txt",
                            mime="text/plain",
                            key="download_smart_updated"
                        )
    else:
        st.info("No contract data available for display")

def render_upload_contract(contract_manager, astra_vector_store, llm):
    styled_header("Upload Contract", "📄")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📤 Upload New Contract", "🤖 Smart Analysis"])

    with tab1:
        render_upload_new_contract(contract_manager, astra_vector_store, llm)

    with tab2:
        render_smart_analysis_tab(contract_manager, astra_vector_store, llm)

def render_upload_new_contract(contract_manager, astra_vector_store, llm):
    """Render the upload new contract functionality with STRICT VALIDATION"""
    st.subheader("Upload New Contract")

    # --- EMAIL INPUT (REQUIRED) ---
    owner_email = st.text_input("📧 Owner Email (Required for alerts)", placeholder="name@company.com", key="owner_email_input")

    upload_option = st.radio("Upload method:",
                           ["PDF File", "URL", "Text File", "Direct Text"],
                           horizontal=True, key="contract_upload")

    extracted_content = None
    source_info = {}

    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose contract PDF", type=['pdf'], key="contract_pdf")
        if uploaded_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=uploaded_file)

    elif upload_option == "URL":
        url = st.text_input("Enter contract URL:", placeholder="https://example.com/contract.pdf")
        if st.button("Extract Contract") and url:
            extracted_content, source_info = extract_text_from_source(upload_option, url=url)

    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose contract text file", type=['txt'], key="contract_txt")
        if text_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=text_file)

    elif upload_option == "Direct Text":
        direct_text = st.text_area("Paste contract text:", height=200)
        if direct_text:
            extracted_content, source_info = extract_text_from_source(upload_option, direct_text=direct_text)

    if extracted_content:
        # ---------------------------------------------------------------------
        # STRICT VALIDATION STEP
        # ---------------------------------------------------------------------
        valid_standards = ["GDPR", "DPDPA", "General Data Protection Regulation", "Digital Personal Data Protection Act"]
        is_valid = any(s.lower() in extracted_content.lower() for s in valid_standards)

        if not is_valid:
            st.error("🚫 **Invalid Contract Type**: This system only processes **GDPR** and **DPDPA** related contracts. Please upload a valid regulatory contract.")
            return

        # Validate owner email
        if not owner_email or "@" not in owner_email:
            st.error("🚫 **Owner Email Required**: Please provide a valid owner email address for contract management and alerts.")
            return

        document_size = len(extracted_content.encode('utf-8'))
        if document_size > 10000:
            st.warning(f"⚠️ Large document detected: {document_size} bytes. The system will optimize it for storage.")

        st.success("✅ Contract content extracted and verified as GDPR/DPDPA related!")

        contract_name = source_info.get("name", f"Contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        st.session_state.temp_contract_name = contract_name
        st.session_state.original_contract_content = extracted_content
        st.session_state.temp_source_info = source_info

        st.info(f"📝 Contract ready for analysis: {contract_name}")

        with st.expander("Contract Preview"):
            preview_text = extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content
            st.text_area("Preview", preview_text, height=300, key="preview_contract")
            st.info(f"Document size: {document_size} bytes")

        # Direct save to database option
        st.subheader("💾 Save to Database")
        col1, col2 = st.columns(2)

        with col1:
            # Contract type selection
            contract_type = st.selectbox(
                "Select Contract Type:",
                ["GDPR", "DPDPA", "Both", "Other"],
                key="contract_type_select"
            )

        with col2:
            # Quick save without analysis
            if st.button("💾 Save Contract to Database (Quick Save)", key="quick_save"):
                # Use LLM for metadata extraction if available, otherwise fallback
                if llm:
                    metadata = contract_manager.extract_contract_metadata_with_llm(extracted_content, contract_name, owner_email, llm)
                else:
                    metadata = contract_manager.extract_contract_metadata_fallback(extracted_content, contract_name, owner_email)
                
                metadata['contract_type'] = contract_type
                metadata['regulatory_standards'] = [contract_type] if contract_type != "Both" else ["GDPR", "DPDPA"]

                success = contract_manager.save_contract_to_astra(
                    contract_name,
                    extracted_content,
                    metadata,
                    "No analysis performed - quick save"
                )

                if success:
                    st.success(f"✅ Contract '{contract_name}' saved to database!")
                    st.info(f"📊 Contract Type: {contract_type}")
                    st.info(f"📧 Owner Email Registered: {owner_email}")
                    
                    # Show LLM-extracted metadata
                    if llm:
                        parties = metadata.get('parties', [])
                        if parties:
                            st.info(f"🤖 LLM-extracted Parties: {', '.join(parties)}")
                        st.info(f"📅 Sign Date: {metadata.get('sign_date', 'Unknown')}")
                        st.info(f"⏰ Termination Date: {metadata.get('termination_date', 'Unknown')}")
                        st.info(f"📆 Days Left: {metadata.get('days_left', 'Unknown')} days")

                    # Show the cleaned collection name used
                    base_name_without_ext = os.path.splitext(contract_name)[0]
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name_without_ext).lower()
                    st.info(f"📁 Collection: contract_{clean_name}")
                    st.rerun()

        st.subheader("🔍 Compliance Analysis")
        regulatory_focus = st.multiselect(
            "Select regulatory frameworks for analysis:",
            contract_manager.config.regulatory_standards,
            default=["GDPR", "DPDPA"],
            key="analysis_frameworks"
        )

        # FIXED: Perform Comprehensive Compliance Analysis button
        if st.button("🚀 Perform Comprehensive Compliance Analysis", key="perform_analysis"):
            if not llm:
                st.error("❌ LLM not configured for analysis")
                return

            with st.spinner("Performing comprehensive compliance analysis..."):
                compliance_docs = []
                for framework in regulatory_focus:
                    try:
                        docs = astra_vector_store.similarity_search(framework, k=2)
                        compliance_docs.extend(docs)
                    except:
                        continue

                contract_content = st.session_state.original_contract_content

                analysis_system = ComplianceAnalysisSystem(astra_vector_store, llm)
                analysis_report, modified_contract, extracted_metadata = analysis_system.comprehensive_compliance_analysis(
                    contract_content, compliance_docs, regulatory_focus, {}, llm
                )

                # Merge with owner email
                extracted_metadata['owner_email'] = owner_email

                st.session_state.analysis_report = analysis_report
                st.session_state.modified_contract = modified_contract
                st.session_state.extracted_metadata = extracted_metadata
                st.session_state.regulatory_focus_used = regulatory_focus

        if hasattr(st.session_state, 'analysis_report'):
            st.subheader("📋 Compliance Analysis Results")

            # Display risk assessment chart
            col1, col2 = st.columns(2)

            with col1:
                # Calculate risk score from analysis
                analysis_text = st.session_state.analysis_report.lower()
                risk_score = 50  # Default medium risk

                if 'non-compliant' in analysis_text or 'violation' in analysis_text:
                    risk_score = 75
                elif 'high risk' in analysis_text:
                    risk_score = 80
                elif 'medium risk' in analysis_text:
                    risk_score = 50
                elif 'low risk' in analysis_text or 'compliant' in analysis_text:
                    risk_score = 25

                risk_level = contract_manager.get_risk_level(risk_score)
                st.markdown("#### 🎯 Risk Assessment")
                risk_chart = create_risk_assessment_chart(risk_score, risk_level)
                st.plotly_chart(risk_chart, use_container_width=True, key="new_contract_risk")

            with col2:
                # Compliance breakdown chart
                st.markdown("#### 📊 Compliance Breakdown")
                compliance_chart = create_compliance_breakdown_chart(st.session_state.analysis_report)
                if compliance_chart:
                    st.plotly_chart(compliance_chart, use_container_width=True, key="new_contract_breakdown")

            if hasattr(st.session_state, 'extracted_metadata'):
                st.markdown("### 📊 Extracted Metadata from LLM Analysis")
                metadata = st.session_state.extracted_metadata

                col1, col2 = st.columns(2)


                
                with col1:
                    parties = metadata.get('parties', [])
                    st.write("**Parties:**", ", ".join(parties) if parties else "No parties extracted")
                    
                    # Format sign date for display
                    sign_date = metadata.get('sign_date', 'Unknown')
                    if sign_date != 'Unknown':
                        try:
                            sign_date_display = datetime.fromisoformat(sign_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                        except:
                            sign_date_display = sign_date.split('T')[0] if 'T' in sign_date else sign_date
                    else:
                        sign_date_display = 'Unknown'
                    st.write("**Sign Date:**", sign_date_display)
                    
                    st.write("**Contract Type:**", metadata.get('contract_type', 'Unknown'))
                    st.write("**Key Clauses:**", ", ".join(metadata.get('key_clauses', [])) if metadata.get('key_clauses') else 'None identified')
                with col2:
                    # Format termination date for display
                    termination_date = metadata.get('termination_date', 'Unknown')
                    if termination_date != 'Unknown':
                        try:
                            termination_date_display = datetime.fromisoformat(termination_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                        except:
                            termination_date_display = termination_date.split('T')[0] if 'T' in termination_date else termination_date
                    else:
                        termination_date_display = 'Unknown'
                    st.write("**Termination Date:**", termination_date_display)
                    
                    st.write("**Days Left:**", f"{metadata.get('days_left', 'Unknown')} days")
                    st.write("**Status:**", metadata.get('status', 'Unknown'))
                    st.write("**Regulatory Standards:**", ", ".join(metadata.get('regulatory_standards', [])))
                    st.write("**Owner Email:**", metadata.get('owner_email', 'N/A'))
            

            st.markdown("### Part 1: Analysis Report")
            st.text_area("Compliance Analysis Details", st.session_state.analysis_report, height=400, key="analysis_details")

            # Download analysis as PDF
            analysis_pdf = create_analysis_pdf(st.session_state.analysis_report, st.session_state.temp_contract_name)
            st.download_button(
                label="📥 Download Analysis Report as PDF",
                data=analysis_pdf,
                file_name=f"analysis_{st.session_state.temp_contract_name}.pdf",
                mime="application/pdf",
                key="download_analysis_pdf"
            )

            st.markdown("### Part 2: Contract Content")
            if st.session_state.modified_contract and "NO CHANGES NEEDED" not in st.session_state.modified_contract.upper():
                st.text_area("Updated Contract Content", st.session_state.modified_contract, height=400, key="modified_contract")

                # Download updated contract as PDF
                contract_pdf = create_pdf_from_text(st.session_state.modified_contract, f"Updated_Contract_{st.session_state.temp_contract_name}")
                st.download_button(
                    label="📥 Download Updated Contract as PDF",
                    data=contract_pdf,
                    file_name=f"updated_contract_{st.session_state.temp_contract_name}.pdf",
                    mime="application/pdf",
                    key="download_modified_pdf"
                )

                final_contract_content = st.session_state.modified_contract
                needs_update = False
            else:
                st.info("✅ No changes needed - contract is compliant")
                st.text_area("Original Contract Content", st.session_state.original_contract_content, height=400, key="original_contract")

                # Download original contract as PDF
                contract_pdf = create_pdf_from_text(st.session_state.original_contract_content, f"Contract_{st.session_state.temp_contract_name}")
                st.download_button(
                    label="📥 Download Original Contract as PDF",
                    data=contract_pdf,
                    file_name=f"contract_{st.session_state.temp_contract_name}.pdf",
                    mime="application/pdf",
                    key="download_original_pdf"
                )

                final_contract_content = st.session_state.original_contract_content
                needs_update = False

            st.subheader("💾 Save to Database with Analysis")
            if st.button("💾 Save Contract with Analysis to Database", key="save_with_analysis"):
                if hasattr(st.session_state, 'extracted_metadata'):
                    final_metadata = {
                        **st.session_state.extracted_metadata,
                        'analysis_performed': True,
                        'analysis_timestamp': datetime.now().isoformat(),
                        'regulatory_frameworks_analyzed': st.session_state.regulatory_focus_used,
                        'needs_update': needs_update,
                        'last_updated': datetime.now().isoformat(),
                        'revision': 0,  # Start from 0
                        'needs_revision': False,
                        'risk_score': 0,
                        'owner_email': owner_email
                    }

                    success = contract_manager.save_contract_to_astra(
                        st.session_state.temp_contract_name,
                        final_contract_content,
                        final_metadata,
                        st.session_state.analysis_report
                    )

                    if success:
                        contract_record = {
                            "id": f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "name": st.session_state.temp_contract_name,
                            "text": final_contract_content,
                            "source_info": st.session_state.temp_source_info,
                            "upload_time": datetime.now().isoformat(),
                            **final_metadata
                        }

                        st.success("✅ Contract successfully saved to database!")
                        st.success(f"📄 Contract Name: {st.session_state.temp_contract_name}")
                        st.success(f"📊 Metadata Extracted: {len(final_metadata)} fields")
                        st.success(f"📧 Owner Email: {owner_email}")
                        
                        # Show LLM-extracted metadata
                        parties = final_metadata.get('parties', [])
                        if parties:
                            st.info(f"🤖 LLM-extracted Parties: {', '.join(parties)}")
                        st.info(f"📅 Sign Date: {final_metadata.get('sign_date', 'Unknown')}")
                        st.info(f"⏰ Termination Date: {final_metadata.get('termination_date', 'Unknown')}")
                        st.info(f"📆 Days Left: {final_metadata.get('days_left', 'Unknown')} days")

                        base_name_without_ext = os.path.splitext(st.session_state.temp_contract_name)[0]
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name_without_ext).lower()
                        st.info(f"📁 Database Collection: contract_{clean_name}")

                        contract_manager.cleanup_session_data()
                        st.rerun()
                    else:
                        st.error("❌ Failed to save contract to database")
                else:
                    st.error("❌ No metadata available. Please perform analysis first.")

def render_enhanced_revision_section(contract_manager, llm):
    """Enhanced revision section with smart assessment and change tracking"""
    tab1, tab2 = st.tabs(["📝 Upload Revision", "⚡ Update Contracts"])

    with tab1:
        render_revision_upload(contract_manager, llm)

    with tab2:
        render_contract_updation(contract_manager, llm)

def render_revision_upload(contract_manager, llm):
    """Upload and process regulatory revisions with smart assessment"""
    st.subheader("Upload Regulatory Revision")

    upload_option = st.radio("Upload method:",
                           ["PDF File", "URL", "Text File", "Direct Text"],
                           horizontal=True, key="revision_upload")

    extracted_content = None
    source_info = {}

    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose revision PDF", type=['pdf'], key="revision_pdf")
        if uploaded_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=uploaded_file)

    elif upload_option == "URL":
        url = st.text_input("Enter revision URL:", placeholder="https://example.com/revision.pdf")
        if st.button("Extract Revision") and url:
            extracted_content, source_info = extract_text_from_source(upload_option, url=url)

    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose revision text file", type=['txt'], key="revision_txt")
        if text_file:
            extracted_content, source_info = extract_text_from_source(upload_option, uploaded_file=text_file)

    elif upload_option == "Direct Text":
        direct_text = st.text_area("Paste revision text:", height=200, key="revision_direct")
        if direct_text:
            extracted_content, source_info = extract_text_from_source(upload_option, direct_text=direct_text)

    if extracted_content:
        st.success("✅ Revision content extracted!")

        revision_type = ComplianceNameManager.detect_compliance_standard(extracted_content)
        st.info(f"🔍 Detected Revision Type: **{revision_type}**")

        with st.expander("Revision Preview"):
            st.text_area("Preview", extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content,
                        height=300, key="revision_preview")

        # Store revision context in session state for later use
        st.session_state.current_revision_context = extracted_content
        st.session_state.current_revision_type = revision_type

        # Smart Revision Impact Assessment
        if st.button("🔍 Smart Revision Impact Assessment", key="assess_impact"):
            if not llm:
                st.error("❌ LLM not available for smart assessment")
                return

            with st.spinner("Performing smart impact assessment using LLM..."):
                affected_contracts = contract_manager.assess_revision_impact_simple(
                    extracted_content, revision_type, llm
                )

                if affected_contracts:
                    st.warning(f"🚨 {len(affected_contracts)} contracts affected by this revision:")

                    # Update sidebar with need revision count - FIXED: Now dynamically updates
                    st.session_state.need_revision_count = len(affected_contracts)

                    # Display impact chart
                    impact_chart = create_revision_impact_chart(affected_contracts)
                    if impact_chart:
                        st.plotly_chart(impact_chart, use_container_width=True, key="impact_chart_main")

                    for contract in affected_contracts:
                        risk_color = "🔴" if contract['risk_level'] == "High" else "🟡" if contract['risk_level'] == "Medium" else "🟢"

                        with st.expander(f"{risk_color} {contract['name']} - {contract['risk_score']}% ({contract['risk_level']} Risk)"):
                            st.write(f"**Revision Type:** {contract['revision_type']}")
                            st.write(f"**Risk Score:** {contract['risk_score']}%")
                            st.write(f"**Risk Level:** {contract['risk_level']}")
                            st.write(f"**Matching Standard:** {contract['matching_standard']}")
                            st.write("**Assessment:** Contract needs revision based on LLM analysis")

                            # Show the logic applied
                            if contract['risk_score'] > 60:
                                st.info("✅ **Action:** Contract marked for revision (risk score > 60%)")
                                st.info("📧 **Alert:** Email notification sent to owner.")
                            else:
                                st.info("ℹ️ **Note:** Contract not marked for revision (risk score ≤ 60%)")
                else:
                    st.success("✅ No contracts significantly affected by this revision")
                    # Update sidebar count to 0
                    st.session_state.need_revision_count = 0
                    st.info("""
                    **Why no contracts are affected:**
                    - LLM analysis determined no revision needed
                    - Contracts are already compliant with the new requirements
                    - Risk scores were below 60% threshold
                    - Contracts already have this revision data
                    """)

def render_contract_updation(contract_manager, llm):
    """Interface for updating contracts that need revision - WITH DETAILED CHANGE TRACKING"""
    st.subheader("Contracts Needing Updation")

    contracts_needing_update = contract_manager.get_contracts_for_updation()

    if not contracts_needing_update:
        st.success("🎉 All contracts are up to date!")
        # Update sidebar count to 0
        st.session_state.need_revision_count = 0
        return

    st.warning(f"⚠️ {len(contracts_needing_update)} contracts need updates:")

    # Check if we have revision context available
    if not hasattr(st.session_state, 'current_revision_context') or not st.session_state.current_revision_context:
        # Fallback: Try to fetch from one of the risky contracts if session is lost
        first_risky = next(iter(contracts_needing_update.values()), None)
        if first_risky and first_risky.get('revision_data'):
             st.session_state.current_revision_context = first_risky.get('revision_data')
             st.session_state.current_revision_type = "Detected from DB"
             st.info("Recovered revision context from database.")
        else:
            st.error("❌ No revision context available. Please upload a regulatory revision first in the 'Upload Revision' tab.")
            return

    if not llm:
        st.error("❌ LLM not available for automatic revision generation")
        return

    revision_context = st.session_state.current_revision_context
    revision_type = st.session_state.current_revision_type

    st.info(f"🔧 Using revision context: **{revision_type}**")

    # Display overall risk assessment
    overall_risk_scores = [info['status']['risk_score'] for info in contracts_needing_update.values()]
    avg_risk_score = sum(overall_risk_scores) / len(overall_risk_scores) if overall_risk_scores else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contracts Needing Update", len(contracts_needing_update))
    with col2:
        st.metric("Average Risk Score", f"{avg_risk_score:.1f}%")
    with col3:
        high_risk_count = sum(1 for score in overall_risk_scores if score > 60)
        st.metric("High Risk Contracts", high_risk_count)

    # We iterate a copy of items to allow modification during iteration if needed
    for contract_name, contract_info in list(contracts_needing_update.items()):

        # SKIP if already updated in this session but page hasn't fully refreshed logic
        if not contract_info['status']['needs_revision']:
            continue

        with st.container():
            status = contract_info['status']
            risk_score = status['risk_score']
            risk_level = contract_manager.get_risk_level(risk_score)

            # Dynamic risk display
            if risk_level == "High":
                risk_display = f"🔴 **High Risk** ({risk_score}%)"
                st.markdown(f"### 📄 {contract_name} - {risk_display}")
            elif risk_level == "Medium":
                risk_display = f"🟡 **Medium Risk** ({risk_score}%)"
                st.markdown(f"### 📄 {contract_name} - {risk_display}")
            else:
                risk_display = f"🟢 **Low Risk** ({risk_score}%)"
                st.markdown(f"### 📄 {contract_name} - {risk_display}")

            metadata = contract_info.get('metadata', {})
            contract_data = contract_info.get('data', '')
            revision_data = contract_info.get('revision_data', '')
            prev_contract_data = contract_info.get('prev_contract_data', '')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Current Revision:** {metadata.get('revision', 0)}")
                st.write(f"**Risk Score:** {risk_score}%")
                # Display LLM-extracted parties
                parties = metadata.get('parties', [])
                st.write(f"**Parties:** {', '.join(parties) if parties else 'No parties'}")
            with col2:
                st.write(f"**Status:** {status['revision_status']}")
                st.write(f"**Type:** {metadata.get('contract_type', 'Unknown')}")
                st.write(f"**Sign Date:** {metadata.get('sign_date', 'Unknown')}")
            with col3:
                st.write(f"**Last Updated:** {status['last_updated']}")
                st.write(f"**Days Left:** {metadata.get('days_left', 'Unknown')} days")
                st.write(f"**Owner:** {metadata.get('owner_email', 'N/A')}")

            # Display risk gauge with UNIQUE KEY per contract
            risk_chart = create_risk_assessment_chart(risk_score, risk_level)
            st.plotly_chart(risk_chart, use_container_width=True, key=f"risk_chart_update_{contract_name}")

            # Show revision data if available
            if revision_data:
                with st.expander("📝 View Revision Data"):
                    st.text_area("Revision Content", revision_data, height=200,
                               key=f"revision_view_{contract_name}")

            # Automatic LLM-based revision generation WITH CHANGE TRACKING
            st.subheader("🤖 Automatic Revision Generation")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Generate Revised Contract", key=f"generate_{contract_name}"):
                    with st.spinner("🤖 Generating revised contract using LLM..."):
                        revised_contract, revisions_list = contract_manager.generate_llm_based_revision(
                            contract_data, revision_context, llm
                        )

                        if revised_contract:
                            st.session_state[f'revised_contract_{contract_name}'] = revised_contract
                            st.session_state[f'revisions_list_{contract_name}'] = revisions_list
                            st.success("✅ Revised contract generated successfully!")

                            # Automatically display changes breakdown with UNIQUE KEY
                            changes_chart = create_changes_breakdown_chart(revisions_list)
                            if changes_chart:
                                st.plotly_chart(changes_chart, use_container_width=True, key=f"changes_chart_{contract_name}")
                        else:
                            st.error("❌ Failed to generate revised contract")

            # Display current contract
            with st.expander("View Current Contract"):
                if isinstance(contract_data, str):
                    st.text_area("Current Contract Content", contract_data, height=300,
                               key=f"current_{contract_name}")
                else:
                    st.json(contract_data)

            # Display revised contract and revisions list if generated
            if (hasattr(st.session_state, f'revised_contract_{contract_name}') and
                hasattr(st.session_state, f'revisions_list_{contract_name}')):

                revised_content = st.session_state[f'revised_contract_{contract_name}']
                revisions_list = st.session_state[f'revisions_list_{contract_name}']

                # Display revisions list
                st.subheader("📝 List of Revisions Made")
                st.text_area("Revisions Details", revisions_list, height=200,
                           key=f"revisions_{contract_name}")

                st.subheader("📄 Revised Contract")
                st.text_area("Revised Contract Content", revised_content, height=400,
                           key=f"revised_content_{contract_name}")

                # Compare changes
                with st.expander("🔍 Compare Changes"):
                    st.subheader("Original vs Revised")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Original Contract**")
                        st.text_area("Original", contract_data[:2000] + "..." if len(contract_data) > 2000 else contract_data,
                                   height=200, key=f"compare_orig_{contract_name}")

                    with col2:
                        st.write("**Revised Contract**")
                        st.text_area("Revised", revised_content[:2000] + "..." if len(revised_content) > 2000 else revised_content,
                                   height=200, key=f"compare_rev_{contract_name}")

                # Download revised contract as PDF
                revised_pdf = create_pdf_from_text(revised_content, f"Revised_Contract_{contract_name}")
                st.download_button(
                    label="📥 Download Revised Contract as PDF",
                    data=revised_pdf,
                    file_name=f"revised_contract_{contract_name}.pdf",
                    mime="application/pdf",
                    key=f"download_revised_{contract_name}"
                )

                # Save to database with revision details
                st.subheader("💾 Save Revision")
                if st.button(f"💾 Save Revised Contract to Database", key=f"save_{contract_name}"):
                    # Prepare revision details
                    revision_details = {
                        'revisions_list': revisions_list,
                        'changes_summary': extract_changes_from_revisions(revisions_list),
                        'revision_context': revision_context,
                        'revision_timestamp': datetime.now().isoformat(),
                        'previous_contract_data': contract_data  # Save current as previous
                    }

                    success = contract_manager.update_contract_with_revision(
                        contract_name, revised_content, revision_type, revision_details
                    )

                    if success:
                        st.success(f"✅ Contract '{contract_name}' successfully updated to revision {metadata.get('revision', 0) + 1}!")
                        st.success(f"📧 Confirmation email sent to owner.")
                        
                        # Show preserved metadata
                        parties = metadata.get('parties', [])
                        sign_date = metadata.get('sign_date', 'Unknown')
                        st.info(f"📋 Original metadata preserved: {len(parties)} parties, Sign date: {sign_date}")

                        # Remove from session state to prevent "same selection" issue and refresh UI
                        if hasattr(st.session_state, f'revised_contract_{contract_name}'):
                            delattr(st.session_state, f'revised_contract_{contract_name}')
                        if hasattr(st.session_state, f'revisions_list_{contract_name}'):
                            delattr(st.session_state, f'revisions_list_{contract_name}')

                        # Update sidebar count
                        current_count = st.session_state.get('need_revision_count', 0)
                        st.session_state.need_revision_count = max(0, current_count - 1)

                        st.rerun()
                    else:
                        st.error("❌ Failed to save revised contract to database")

            st.markdown("---")

def extract_changes_from_revisions(revisions_text):
    """Extract individual changes from revisions text for storage"""
    changes = []
    lines = revisions_text.split('\n')

    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('•') or any(word in line.lower() for word in ['added', 'modified', 'removed', 'updated'])):
            changes.append(line)

    return changes[:20]  # Limit to 20 changes for storage

def extract_metadata_from_analysis(analysis_report, contract_content):
    """Extract metadata from the analysis report and contract content"""
    metadata = {
        'contract_type': 'Unknown',
        'contract_date': datetime.now().isoformat(),
        'parties': [],
        'key_clauses': [],
        'regulatory_standards': [],
        'status': 'Active',
        'revision': 0  # Start from 0
    }

    try:
        report_lower = analysis_report.lower()
        content_lower = contract_content.lower()

        contract_types = {
            'Service Agreement': ['service agreement', 'service contract'],
            'Employment Contract': ['employment', 'employee agreement'],
            'NDA': ['non-disclosure', 'confidentiality', 'nda'],
            'Partnership': ['partnership', 'joint venture'],
            'License Agreement': ['license', 'licensing'],
            'Purchase Agreement': ['purchase', 'sales agreement'],
            'Lease Agreement': ['lease', 'rental']
        }

        for contract_type, keywords in contract_types.items():
            if any(keyword in report_lower or keyword in content_lower for keyword in keywords):
                metadata['contract_type'] = contract_type
                break

        party_patterns = [
            r'between\s+([^,]+?)\s+and\s+([^,\.]+)',
            r'parties:\s*(.+?)\s*and\s*(.+)',
            r'this\s+agreement.*?between\s+([^,]+?)\s+and\s+([^,\.]+)'
        ]

        for pattern in party_patterns:
            matches = re.findall(pattern, contract_content, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        metadata['parties'].extend([party.strip() for party in match])
                    else:
                        metadata['parties'].append(match.strip())

        metadata['parties'] = list(set([p for p in metadata['parties'] if p and len(p) > 2]))

        clause_keywords = {
            'Confidentiality': ['confidential', 'non-disclosure'],
            'Termination': ['termination', 'term'],
            'Payment': ['payment', 'compensation', 'fees'],
            'Liability': ['liability', 'indemnification'],
            'Intellectual Property': ['intellectual property', 'ip', 'copyright'],
            'Governing Law': ['governing law', 'jurisdiction']
        }

        for clause, keywords in clause_keywords.items():
            if any(keyword in report_lower for keyword in keywords):
                metadata['key_clauses'].append(clause)

        regulatory_standards = {
            'GDPR': ['gdpr', 'general data protection'],
            'DPDPA': ['dpdpa', 'digital personal data'],
        }

        for standard, keywords in regulatory_standards.items():
            if any(keyword in report_lower for keyword in keywords):
                metadata['regulatory_standards'].append(standard)

        date_patterns = [
            r'\b(\w+\s+\d{1,2},\s*\d{4})',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\b\d{4}-\d{1,2}-\d{1,2}'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, contract_content)
            if matches:
                try:
                    date_str = matches[0]
                    for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                        try:
                            metadata['contract_date'] = datetime.strptime(date_str, fmt).isoformat()
                            break
                        except:
                            continue
                except:
                    continue

        if 'non-compliant' in report_lower or 'violation' in report_lower:
            metadata['status'] = 'Needs Review'

        # Calculate termination date (180 days from contract date)
        try:
            contract_date = datetime.fromisoformat(metadata['contract_date'])
            termination_date = contract_date + timedelta(days=180)
            days_left = (termination_date - datetime.now()).days
            
            metadata['termination_date'] = termination_date.isoformat()
            metadata['days_left'] = max(0, days_left)
            metadata['status'] = "Active" if days_left > 0 else "Expired"
        except:
            pass

    except Exception as e:
        st.error(f"Error extracting metadata from analysis: {str(e)}")

    return metadata


# Main Streamlit App Logic
def main():
    st.set_page_config(
        page_title="Enhanced Contract Compliance System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern AI App Styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .contract-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { border-left-color: #dc3545; }
    .risk-medium { border-left-color: #ffc107; }
    .risk-low { border-left-color: #28a745; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize components
    config_manager = get_config_manager()
    astra_db_manager = get_astra_db_manager(config_manager)
    contract_database = get_contract_database(config_manager, astra_db_manager)
    contract_manager = get_contract_manager(contract_database)

    groq_api_key = get_groq_api()
    llm = init_llm(groq_api_key)
    embeddings = get_embeddings_model()
    astra_vector_store = get_vector_store(astra_db_manager, embeddings)

    # Initialize global contracts
    contract_manager.initialize_global_contracts()

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: white; margin-bottom: 0;">🔍</h1>
            <h2 style="color: white; margin-top: 0;">Enhanced Compliance System</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("📊 Global Contracts")
        if contract_manager.global_contracts.contracts:
            total = len(contract_manager.global_contracts.contracts)
            needs_rev = len(contract_manager.global_contracts.get_contracts_needing_revision())
            st.write(f"**Total:** {total}")
            st.write(f"**Need Revision:** {needs_rev}")
            st.write(f"**Up to Date:** {total - needs_rev}")

            # Display need revision count from session state if available
            need_revision_count = st.session_state.get('need_revision_count', needs_rev)
            st.info(f"📊 Contracts in Risk List: {need_revision_count}")
        else:
            st.write("No contracts loaded")

        st.markdown("---")

        page = st.radio("Navigation", [
            "🏠 Dashboard",
            "🌐 Global Contracts",
            "📥 Upload Compliance",
            "📄 Upload Contract",
            "🔄 Revision Management"
        ])

    if page == "🏠 Dashboard":
        render_dashboard(contract_manager)
    elif page == "🌐 Global Contracts":
        # Pass LLM and astra_vector_store to enable enhanced Chatbot in dashboard
        render_global_contracts_dashboard(contract_manager, llm, astra_vector_store)
    elif page == "📥 Upload Compliance":
        render_upload_compliance(contract_manager, astra_vector_store)
    elif page == "📄 Upload Contract":
        render_upload_contract(contract_manager, astra_vector_store, llm)
    elif page == "🔄 Revision Management":
        render_enhanced_revision_section(contract_manager, llm)

if __name__ == "__main__":
    main()
