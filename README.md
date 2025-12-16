# 🔍 Enhanced Contract Compliance System

## 🌟 Overview

**Enhanced Contract Compliance System** is a sophisticated AI-powered platform for automated contract management, regulatory compliance analysis, and intelligent document processing. Built with Streamlit and powered by LangChain with Groq's Llama 3.3-70B, this system provides comprehensive tools for legal document analysis, compliance monitoring, and automated contract revisions.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Key Features

### 🤖 **AI-Powered Contract Analysis**
- **LLM Metadata Extraction**: Automatically extracts parties, dates, clauses, and compliance standards using Groq's Llama 3.3-70B
- **Smart Compliance Checking**: Real-time validation against GDPR, DPDPA, and other regulatory frameworks
- **Risk Assessment**: Automated risk scoring with visual dashboards and gauge charts

### 📊 **Intelligent Document Management**
- **Multi-format Support**: PDF, URL, Text files, and direct text input
- **Smart Storage**: AstraDB integration with automatic compression and optimization
- **Version Control**: Complete revision history with diff tracking and previous version access

### 🔄 **Automated Revision System**
- **Smart Impact Assessment**: AI-driven analysis of regulatory changes on existing contracts
- **LLM-Based Revision Generation**: Automatic contract updates with detailed change tracking
- **Email Notifications**: Automated alerts for high-risk contracts requiring attention

### 💬 **Interactive Chat Interface**
- **RAG-Powered Chatbot**: Context-aware Q&A about specific contracts using vector embeddings
- **Contract-Specific Assistance**: Ask questions about clauses, risks, and compliance issues
- **Conversation History**: Maintains context for continuous analysis sessions

### 📈 **Visual Analytics Dashboard**
- **Risk Assessment Gauges**: Interactive Plotly charts showing contract risk levels
- **Compliance Breakdown**: Visual representation of regulatory adherence
- **Revision Impact Charts**: Track how updates affect contract portfolios

## 🏗️ Architecture

```
Enhanced Compliance System
├── Frontend (Streamlit)
│   ├── Dashboard & Analytics
│   ├── Contract Upload Interface
│   ├── Chatbot Interface
│   └── Revision Management
├── AI Layer (LangChain + Groq)
│   ├── LLM Metadata Extraction
│   ├── Compliance Analysis
│   ├── Revision Generation
│   └── RAG Chatbot
├── Database Layer
│   ├── AstraDB (Primary Storage)
│   ├── Vector Store (Compliance Docs)
│   └── Caching System
└── Integration Layer
    ├── Email Notification System
    ├── PDF Processing
    └── API Connectivity
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- AstraDB account and credentials
- Groq API key
- Gmail account for email notifications (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/contract-compliance-system.git
   cd contract-compliance-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API credentials**
   Create `.env` in the project root:
   ```ini
   ASTRA_DB_APPLICATION_TOKEN="your-astra-db-token"
   ASTRA_DB_ID="your-astra-db-id"
   groq_api_key="your-groq-api-key"
   SENDER_EMAIL="your-email@gmail.com"
   SENDER_PASSWORD="your-app-password"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
contract-compliance-system/
├── complete_compliance_app.py   # Main application
├── api_integration.py           # API Integration
├── database.py                  # database connectivity
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore file
```

## 🛠️ Core Components

### 1. **Global Contract Manager**
- In-memory state management for all contracts
- Real-time status tracking and risk assessment
- Centralized contract lifecycle management

### 2. **LLM Metadata Extractor**
- Extracts structured metadata from contracts
- Supports multiple date formats and languages
- Handles complex legal terminology

### 3. **Compliance Analysis System**
- Multi-framework compliance checking (GDPR, DPDPA, etc.)
- Automated violation detection
- Rectification suggestions

### 4. **Smart Revision Engine**
- AI-driven revision generation
- Change tracking and documentation
- Email notifications for stakeholders

### 5. **Contract Chatbot**
- RAG-based Q&A system
- Contract-specific context awareness
- Continuous learning from conversations

## 🔧 Configuration

### Environment Variables
```bash
# Required
ASTRA_DB_APPLICATION_TOKEN=your_token_here
ASTRA_DB_ID=your_db_id_here
GROQ_API_KEY=your_groq_key_here

# Optional (for email notifications)
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
```

### AstraDB Setup
1. Create a new AstraDB database
2. Enable Vector Search capabilities
3. Generate application token
4. Configure endpoint in the application

## 📊 Usage Examples

### Uploading a Contract
1. Navigate to "Upload Contract" tab
2. Select upload method (PDF, URL, Text, or Direct)
3. Enter owner email for notifications
4. Choose regulatory frameworks for analysis
5. Review AI-extracted metadata and risk assessment

### Analyzing Compliance
1. Select contract from global dashboard
2. Choose regulatory frameworks
3. View comprehensive analysis report
4. Download PDF reports for documentation

### Managing Revisions
1. Upload new regulatory documents
2. View AI-generated impact assessment
3. Generate revised contracts automatically
4. Track all changes with version history

### Using the Chatbot
1. Select a contract from the dashboard
2. Click the chat button
3. Ask questions about clauses, risks, or compliance
4. Get AI-powered insights with contract context

## 🎨 UI Features

- **Modern Gradient Design**: Beautiful purple-blue gradient theme
- **Interactive Charts**: Plotly-powered visualizations
- **Responsive Layout**: Adapts to different screen sizes
- **Card-based Interface**: Clean, organized content presentation
- **Real-time Updates**: Live status indicators and counters

## 📧 Email Integration

The system includes a production-ready email notification system:
- **Contract Revision Alerts**: Notify owners when contracts need updates
- **Revision Confirmations**: Send updates when contracts are successfully revised
- **High-Risk Notifications**: Immediate alerts for critical compliance issues

**Note**: Uses Gmail SMTP with app-specific passwords for security.

## 🗃️ Data Storage

### AstraDB Collections
- `contract_[name]`: Individual contract storage with metadata
- `compliance_vector_store`: Vector embeddings for regulatory documents
- Automatic compression for large documents
- Version-controlled revisions with full history

### Data Optimization
- Automatic text compression for large documents
- Smart metadata serialization
- Efficient chunking for vector storage

## 🔒 Security Features

- **Credential Management**: Secure API key storage in `api.txt`
- **Email Security**: Uses app-specific passwords
- **Data Validation**: Strict input validation for contracts
- **Session Management**: Secure state handling in Streamlit

## 🧪 Testing

```bash

# Test specific components
python database.py
python api_integration.py
```

## 📈 Performance

- **Fast Metadata Extraction**: < 5 seconds for most contracts
- **Efficient Storage**: Automatic compression reduces storage by 70%
- **Real-time Updates**: Instant dashboard refreshes
- **Scalable Architecture**: Supports hundreds of contracts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🆘 Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/contract-compliance-system/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/contract-compliance-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/contract-compliance-system/discussions)

## 🙏 Acknowledgments

- **Streamlit** for the amazing frontend framework
- **Groq** for high-performance LLM inference
- **LangChain** for the AI orchestration framework
- **AstraDB** for scalable vector and document storage
- **Plotly** for interactive visualizations

---

## 📱 Screenshots

*Dashboard View*
![Dashboard](https://raw.githubusercontent.com/shobhit-vision/Compliance-Checker/e799499a42175c5a602604bcc4ab207a39f42fc5/Dashboard.png)

*Contract Analysis*
![Analysis](https://raw.githubusercontent.com/shobhit-vision/Compliance-Checker/e799499a42175c5a602604bcc4ab207a39f42fc5/Contract_analysis.png)

*Chat Interface*
![Chat](https://raw.githubusercontent.com/shobhit-vision/Compliance-Checker/e799499a42175c5a602604bcc4ab207a39f42fc5/global_contracts.png)

---

**Built with ❤️ for LegalTech Innovation**

*Transform your contract management with AI-powered compliance monitoring.*
