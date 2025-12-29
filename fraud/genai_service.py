import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class ComplianceAgent:

    BANK_POLICIES = "Policy 404: Transactions exceeding $5,000 between 23:00 and 04:00 are flagged as High Risk. Policy 502: Patterns indicative of structuring are strictly prohibited."

    @staticmethod
    def redact_pii(text: str) -> str:
        """
        Detects and redacts PII (PERSON, EMAIL_ADDRESS, CREDIT_CARD) from the input text.
        """
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        
        results = analyzer.analyze(text=text,
                                   entities=["PERSON", "EMAIL_ADDRESS", "CREDIT_CARD"],
                                   language='en')
        
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_text.text

    @staticmethod
    def generate_sar(transaction_details: str, fraud_score: float) -> str:
        """
        Generates a Suspicious Activity Report (SAR) using an LLM.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            return (
                "AUTOMATED SAR (GenAI Offline):\n"
                f"High-risk transaction detected with a fraud probability of {fraud_score:.2%}. "
                "The transaction details match patterns associated with high-risk activities. "
                "Manual review is required."
            )
            
        try:
            # Initialize the Google Gemini Pro model
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

            prompt_template = """
            You are a professional fraud analyst. Generate a formal "Suspicious Activity Report (SAR)" strictly explaining why the transaction violates bank policies.

            [Sanitized Transaction]:
            {transaction}

            [Bank Policies]:
            {policies}
            
            [Fraud Score]:
            {score}
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            chain = prompt | llm
            
            response = chain.invoke({
                "transaction": transaction_details,
                "policies": ComplianceAgent.BANK_POLICIES,
                "score": fraud_score
            })
            return response.content
        except Exception as e:
            # Fallback to a predefined message if the Gemini API fails
            fallback_text = (
                "AUTOMATED SAR (GenAI Error):\n"
                f"High-risk transaction detected with a fraud probability of {fraud_score:.2%}. "
                f"The GenAI system failed to generate a detailed report. Error: {e}. "
                "Manual review is required."
            )
            return fallback_text