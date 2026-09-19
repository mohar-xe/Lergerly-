"""
Ledgerly - Amazon Bedrock Transaction Extraction Service
Converts shopkeeper natural-language notes into structured transaction records.

Financial Rule:
Amazon Bedrock is ONLY used for entity extraction.
Customer balance arithmetic (balance = total CREDIT - total PAYMENT)
is strictly computed deterministically by backend code, NEVER by the AI model.
"""

import os
import json
import re
from typing import Any, Dict, Optional, Union

try:
    import boto3
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        EndpointConnectionError,
    )
except ImportError:
    boto3 = None
    BotoCoreError = ClientError = NoCredentialsError = PartialCredentialsError = EndpointConnectionError = Exception


class BedrockError(Exception):
    """Base exception for Bedrock operations."""
    pass


class BedrockUnavailableError(BedrockError):
    """Raised when AWS Bedrock is unreachable, credentials are missing, or service is offline."""
    pass


class BedrockExtractionError(BedrockError):
    """Raised when model response cannot be parsed or validation of extracted fields fails."""
    pass


DEMO_LANGUAGES = ["en-IN", "hi-IN", "bn-IN", "mr-IN", "ta-IN", "te-IN"]
DEMO_LANGUAGE_LABELS = {
    "en-IN": "English (en)", "hi-IN": "Hindi (hi, हिन्दी)", "bn-IN": "Bengali (bn, বাংলা)",
    "mr-IN": "Marathi (mr, मराठी)", "ta-IN": "Tamil (ta, தமிழ்)", "te-IN": "Telugu (te, తెలుగు)",
}
# Full 23-language list retained for future expansion (not active in demo)
INDIAN_LANGUAGES = [
    "Assamese (as, অসমীয়া)", "Bengali (bn, বাংলা)", "Bodo (brx, बर')", "Dogri (doi, डोगरी)",
    "Gujarati (gu, ગુજરાતી)", "Hindi (hi, हिन्दी)", "Kannada (kn, ಕನ್ನಡ)", "Kashmiri (ks, کٲشُر)",
    "Konkani (kok, कोंकणी)", "Maithili (mai, मैथिली)", "Malayalam (ml, മലയാളം)", "Manipuri (mni, মণিপুরী)",
    "Marathi (mr, मराठी)", "Nepali (ne, नेपाली)", "Odia (or, ଓଡ଼ିଆ)", "Punjabi (pa, ਪੰਜਾਬੀ)",
    "Sanskrit (sa, संस्कृतम्)", "Santali (sat, ᱥᱟᱱᱛᱟᱲᱤ)", "Sindhi (sd, سنڌي)", "Tamil (ta, தமிழ்)",
    "Telugu (te, తెలుగు)", "Urdu (ur, اردو)", "English (en)"
]

EXTRACTION_SYSTEM_PROMPT = """You are a multilingual financial entity extractor for an Indian kirana store ledger assistant.
DEMO MODE: Input will be in ONE of 6 languages: English, Hindi (hi-IN, Devanagari + roman Hinglish), Bengali (bn-IN, Bengali script + roman), Marathi (mr-IN, Devanagari), Tamil (ta-IN, Tamil script + roman), Telugu (te-IN, Telugu script + roman). Other languages are NOT supported yet – if input appears to be another language, still extract best-effort but note language hint.

You MUST extract exactly these 4 fields:
1. "customerName": Name as it appears (preserve original script/roman). Must not be empty.
2. "type": Must be either "CREDIT" or "PAYMENT".
   Lexicon (demo languages, any script/roman): CREDIT = udhar/udhaar/उधार/বাকি/বাকী/उधारी/கடன்/అప్పు/baki/credit/borrowed/lena/देना; PAYMENT = jama/jamaa/जमा/জমা/జమ/செலுத்தினார்/జమ/paid/cleared/jama/bharla/chukta/diya/wapas/return; Hinglish also jama/bhugtan.
   Infer from context if ambiguous; default CREDIT for ambiguous "took/bought/got" + amount.
3. "amount": Positive number in Rupees. Normalize: Devanagari ०-९, Bengali ০-৯, Tamil ௦-௯, Telugu ౦-౯ → 0-9; handle lakh (1L=100000), crore, k (5k=5000), comma 5,000.
4. "description": Goods note VERBATIM in source language/script as spoken (e.g. Tamil "அரிசி", Hindi "चावल", Bengali "চাল", Telugu "బియ్యం", Marathi "तांदूळ"). Do NOT translate to English. If no goods (e.g. "Rahul paid 300"), use "".

CRITICAL RULES:
- NEVER calculate or guess customer balances.
- NEVER output a balance or remaining amount.
- Preserve source script for customerName/description (if input roman, keep roman).
- Output ONLY valid JSON matching this schema:
{"customerName": "...", "type": "CREDIT|PAYMENT", "amount": 100, "description": "..."}
- Do NOT output markdown code blocks (no ```json). Output raw JSON only. Language hint may be provided; trust it but verify from text."""


REPLY_SYSTEM_PROMPT = """You are Ledgerly, a friendly kirana store assistant replying via WhatsApp/Telegram.
Given a newly recorded transaction and the customer's updated balance, generate a concise reply.

Rules:
- Keep it 1-2 lines, under 300 characters.
- Include: customer name, amount with ₹, CREDIT/udhar vs PAYMENT/jama phrasing IN USER'S LANGUAGE, and exact balance provided.
- DEMO MODE: Supported languages are English (en-IN), Hindi (hi-IN), Bengali (bn-IN), Marathi (mr-IN), Tamil (ta-IN), Telugu (te-IN). Reply in SAME language and SAME script as Original note. If Language hint provided (e.g. hi-IN, ta-IN), prioritize it. If Hindi note → Hindi; Tamil→Tamil; Telugu→Telugu; Bengali→Bengali; Marathi→Marathi. Hinglish roman → roman Hinglish.
- Never guess or recalculate balance - use the exact balance provided.
- Use idiomatic demo terms: Hindi उदार/जमा, Bengali বাকি/জমা, Marathi उधारी/जमा, Tamil கடன்/செலுத்தினார், Telugu అప్పు/జమ, English credit/payment. For unsupported, use English "credit/payment".
- For CREDIT: include "udhar/credit/baki" equivalent + "Kul/Balance". For PAYMENT: "jama/paid" + "Bacha/Remaining".
- Add a small emoji (✅ for CREDIT, 🙏 for PAYMENT) at end.
- Do NOT output JSON, output plain text reply only."""

# Fallback templates per demo language when Bedrock is unavailable (6 langs + aliases)
FALLBACK_TEMPLATES = {
    "hi-IN": {"CREDIT": "{name} ke khate me {amt} udhar joda. Kul udhar: {bal}. ✅", "PAYMENT": "{name} ne {amt} jama kiye. Bacha udhar: {bal}. Dhanyavad! 🙏"},
    "en-IN": {"CREDIT": "Recorded {amt} credit for {name}. Total due: {bal}. ✅", "PAYMENT": "Recorded {amt} payment from {name}. Balance: {bal}. Thanks! 🙏"},
    "en": {"CREDIT": "Recorded {amt} credit for {name}. Total due: {bal}. ✅", "PAYMENT": "Recorded {amt} payment from {name}. Balance: {bal}. Thanks! 🙏"},
    "bn-IN": {"CREDIT": "{name}-এর খাতায় {amt} বাকি যোগ হলো। মোট বাকি: {bal}. ✅", "PAYMENT": "{name} {amt} জমা করেছেন। বাকি: {bal}. ধন্যবাদ! 🙏"},
    "bn": {"CREDIT": "{name}-এর খাতায় {amt} বাকি যোগ হলো। মোট বাকি: {bal}. ✅", "PAYMENT": "{name} {amt} জমা করেছেন। বাকি: {bal}. ধন্যবাদ! 🙏"},
    "mr-IN": {"CREDIT": "{name} च्या खात्यात {amt} उधारी जोडली. एकूण बाकी: {bal}. ✅", "PAYMENT": "{name} यांनी {amt} जमा केले. बाकी: {bal}. धन्यवाद! 🙏"},
    "mr": {"CREDIT": "{name} च्या खात्यात {amt} उधारी जोडली. एकूण बाकी: {bal}. ✅", "PAYMENT": "{name} यांनी {amt} जमा केले. बाकी: {bal}. धन्यवाद! 🙏"},
    "ta-IN": {"CREDIT": "{name} கணக்கில் {amt} கடன் சேர்க்கப்பட்டது. மொத்த நிலுவை: {bal}. ✅", "PAYMENT": "{name} {amt} செலுத்தினார். மீதி: {bal}. நன்றி! 🙏"},
    "ta": {"CREDIT": "{name} கணக்கில் {amt} கடன் சேர்க்கப்பட்டது. மொத்த நிலுவை: {bal}. ✅", "PAYMENT": "{name} {amt} செலுத்தினார். மீதி: {bal}. நன்றி! 🙏"},
    "te-IN": {"CREDIT": "{name} ఖాతాలో {amt} అప్పు జోడించారు. మొత్తం బకాయి: {bal}. ✅", "PAYMENT": "{name} {amt} జమ చేశారు. మిగిలిన బకాయి: {bal}. ధన్యవాదాలు! 🙏"},
    "te": {"CREDIT": "{name} ఖాతాలో {amt} అప్పు జోడించారు. మొత్తం బకాయి: {bal}. ✅", "PAYMENT": "{name} {amt} జమ చేశారు. మిగిలిన బకాయి: {bal}. ధన్యవాదాలు! 🙏"},
}
# Future languages (kept for completeness but not active in demo) – map to nearest demo template
FALLBACK_TEMPLATES["gu-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["gu"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["kn-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["kn"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ml-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ml"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["pa-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["pa"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["or-IN"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["or"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["as-IN"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["as"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["ur"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ur-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ne-NP"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ne"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["sd"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["sd-IN"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["sa"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["kok"] = FALLBACK_TEMPLATES["mr-IN"]
FALLBACK_TEMPLATES["kok-IN"] = FALLBACK_TEMPLATES["mr-IN"]
FALLBACK_TEMPLATES["mni"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["mni-IN"] = FALLBACK_TEMPLATES["bn-IN"]
FALLBACK_TEMPLATES["brx"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["doi"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["ks"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["mai"] = FALLBACK_TEMPLATES["hi-IN"]
FALLBACK_TEMPLATES["sat"] = FALLBACK_TEMPLATES["bn-IN"]
# Ensure short ↔ long code both exist for demo langs
for k in ["hi", "bn", "mr", "ta", "te", "en"]:
    short = k
    long_map = {"hi": "hi-IN", "bn": "bn-IN", "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN", "en": "en-IN"}
    long_code = long_map.get(k, k)
    if short in FALLBACK_TEMPLATES and long_code not in FALLBACK_TEMPLATES:
        FALLBACK_TEMPLATES[long_code] = FALLBACK_TEMPLATES[short]
    elif long_code in FALLBACK_TEMPLATES and short not in FALLBACK_TEMPLATES:
        FALLBACK_TEMPLATES[short] = FALLBACK_TEMPLATES[long_code]


DEMO_DETECT_SET = {"en-IN", "hi-IN", "bn-IN", "mr-IN", "ta-IN", "te-IN"}

def detect_language_from_text(text: str) -> str:
    """Detects language from Unicode script + keyword heuristic. For demo, correctly identifies all scripts but handler will reject non-demo."""
    if not text or not text.strip():
        return "en-IN"
    # Keyword heuristics for Devanagari disambiguation (hi vs mr)
    if any(kw in text for kw in ["उधारी", "तांदूळ", "खात्यात", "जमा केले", "एकूण", "ळ"]):
        return "mr-IN"
    # Nepali specific
    if any(kw in text for kw in ["खातामा", "जम्मा", "बाँकी"]):
        return "ne-NP"
    # Check script blocks: full detection for accurate rejection of non-demo langs
    for ch in text:
        cp = ord(ch)
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta-IN"
        if 0x0C00 <= cp <= 0x0C7F:
            return "te-IN"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn-IN"
        if 0x0D00 <= cp <= 0x0D7F:
            return "ml-IN"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gu-IN"
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa-IN"
        if 0x0B00 <= cp <= 0x0B7F:
            return "or-IN"
        if 0x0980 <= cp <= 0x09FF:
            return "bn-IN"
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            return "ur"
        if 0x1C50 <= cp <= 0x1C7F:
            return "sat"
        if 0x0900 <= cp <= 0x097F:
            return "hi-IN"
    lower = text.lower()
    if any(kw in lower for kw in ["jama", "udhar", "udhaar", "baki", "bhaav", "rupaye", "rupya", "bhugtan", "diya", "liya", "kar diya", "ho gaya"]):
        return "hi-IN"
    return "en-IN"

def is_demo_language_supported(code: str) -> bool:
    return code in DEMO_DETECT_SET or code in {"en", "hi", "bn", "mr", "ta", "te", "auto"}


class BedrockService:
    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client

        if boto3 is None:
            raise BedrockUnavailableError("boto3 library is not available in the current environment.")

        try:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
            )
            return self._client
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise BedrockUnavailableError(
                f"AWS credentials not configured for Amazon Bedrock: {str(e)}"
            )
        except Exception as e:
            raise BedrockUnavailableError(
                f"Failed to initialize Amazon Bedrock client: {str(e)}"
            )

    def _build_model_payload(self, text: str, language_code: Optional[str] = None) -> Dict[str, Any]:
        """Formats model payload based on provider API conventions. Supports language hint for 23 Indian langs."""
        model_lower = self.model_id.lower()
        lang_hint = f"\nLanguage hint: {language_code} (trust but verify from text)." if language_code else ""
        user_content = f"Extract transaction information from this shopkeeper message (may be multilingual/code-mix):\n\"{text}\"{lang_hint}"

        if "anthropic" in model_lower:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.0,
                "system": EXTRACTION_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}],
            }
        elif "titan" in model_lower:
            prompt = f"{EXTRACTION_SYSTEM_PROMPT}{lang_hint}\n\nShopkeeper message: \"{text}\"\n\nJSON output:"
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.0,
                    "stopSequences": ["\n\n"],
                },
            }
        else:
            return {
                "prompt": f"{EXTRACTION_SYSTEM_PROMPT}{lang_hint}\n\nInput: \"{text}\"\nJSON:",
                "max_gen_len": 512,
                "temperature": 0.0,
            }

    def _extract_text_from_response(self, response_body: Dict[str, Any]) -> str:
        """Extracts generated text string from model provider response structure."""
        model_lower = self.model_id.lower()

        if "anthropic" in model_lower:
            # Anthropic Claude format: {"content": [{"text": "...", "type": "text"}]}
            contents = response_body.get("content", [])
            for c in contents:
                if isinstance(c, dict) and c.get("type") == "text":
                    return c.get("text", "").strip()
            return ""

        if "titan" in model_lower:
            results = response_body.get("results", [])
            if results and isinstance(results[0], dict):
                return results[0].get("outputText", "").strip()
            return ""

        # Generic fallback checks
        if "generation" in response_body:
            return str(response_body["generation"]).strip()

        if "output" in response_body:
            return str(response_body["output"]).strip()

        return ""

    def parse_and_validate_extraction(self, raw_output: str) -> Dict[str, Any]:
        """
        Parses model text output into JSON and strictly validates:
        - customerName: non-empty string
        - type: CREDIT or PAYMENT
        - amount: positive numeric value
        - description: string (empty string if not specified)
        """
        if not raw_output or not raw_output.strip():
            raise BedrockExtractionError("Model returned an empty response.")

        cleaned = raw_output.strip()

        # Remove markdown code block fences if generated by model
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Locate JSON object boundaries
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            json_str = match.group(0)
        else:
            json_str = cleaned

        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            raise BedrockExtractionError(f"Model output is not valid JSON: {raw_output}. Parse error: {str(e)}")

        if not isinstance(parsed, dict):
            raise BedrockExtractionError("Extracted data must be a JSON object dictionary.")

        # 1. Validate customerName
        if "customerName" not in parsed:
            raise BedrockExtractionError("Extracted transaction missing 'customerName'.")

        customer_name = str(parsed.get("customerName") or "").strip()
        if not customer_name:
            raise BedrockExtractionError("'customerName' cannot be empty.")

        # 2. Validate transaction type
        if "type" not in parsed:
            raise BedrockExtractionError("Extracted transaction missing 'type'.")

        tx_type = str(parsed.get("type") or "").strip().upper()
        if tx_type not in ("CREDIT", "PAYMENT"):
            raise BedrockExtractionError(
                f"Unsupported transaction type '{tx_type}'. Must be CREDIT or PAYMENT."
            )

        # 3. Validate amount
        if "amount" not in parsed:
            raise BedrockExtractionError("Extracted transaction missing 'amount'.")

        raw_amount = parsed.get("amount")
        try:
            amount = float(raw_amount)
        except (ValueError, TypeError):
            raise BedrockExtractionError(f"Amount must be a numeric value, received: {raw_amount}")

        if amount <= 0:
            raise BedrockExtractionError(f"Amount must be greater than zero, received: {amount}")

        # Format integer amounts cleanly (e.g. 500 instead of 500.0)
        formatted_amount: Union[int, float] = int(amount) if amount.is_integer() else amount

        # 4. Description
        description = str(parsed.get("description") or "").strip()

        return {
            "customerName": customer_name,
            "type": tx_type,
            "amount": formatted_amount,
            "description": description,
        }

    def extract_transaction(self, text: str, language_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entrypoint: sends natural language note to Bedrock and returns structured transaction.
        language_code: optional hint like hi-IN, ta-IN, auto, bn-IN etc for 23 Indian langs.
        """
        if not text or not text.strip():
            raise BedrockExtractionError("Input text cannot be empty.")

        client = self._get_client()
        payload = self._build_model_payload(text.strip(), language_code=language_code)

        from services.retry_helper import retry_with_backoff, is_aws_transient_error

        def _do_invoke():
            return client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        try:
            response = retry_with_backoff(
                _do_invoke,
                max_retries=2,
                base_delay=0.2,
                max_delay=2.0,
                is_retryable_fn=is_aws_transient_error,
            )
            raw_body = response.get("body")
            if hasattr(raw_body, "read"):
                response_data = json.loads(raw_body.read().decode("utf-8"))
            elif isinstance(raw_body, (str, bytes)):
                response_data = json.loads(raw_body)
            elif isinstance(raw_body, dict):
                response_data = raw_body
            else:
                response_data = {}
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise BedrockUnavailableError(f"AWS credentials not configured for Amazon Bedrock: {str(e)}")
        except EndpointConnectionError as e:
            raise BedrockUnavailableError(f"Could not connect to Amazon Bedrock endpoint: {str(e)}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            raise BedrockUnavailableError(f"Bedrock ClientError [{code}]: {msg}")
        except BotoCoreError as e:
            raise BedrockUnavailableError(f"Bedrock BotoCoreError: {str(e)}")
        except Exception as e:
            raise BedrockUnavailableError(f"Failed during Bedrock invocation: {str(e)}")

        model_text = self._extract_text_from_response(response_data)
        return self.parse_and_validate_extraction(model_text)

    def _build_reply_payload(self, extracted: Dict[str, Any], balance: Any, original_text: str, language_code: Optional[str] = None) -> Dict[str, Any]:
        model_lower = self.model_id.lower()
        lang_line = f"Language hint: {language_code}. Reply in SAME language+script as Original note (prioritize hint).\n" if language_code else "Language: Reply in SAME language+script as Original note (auto-detect from text).\n"
        user_content = (
            f"{lang_line}"
            f"Original shopkeeper note: \"{original_text}\"\n"
            f"Extracted transaction: customerName={extracted.get('customerName')}, type={extracted.get('type')}, amount={extracted.get('amount')}, description={extracted.get('description','')}\n"
            f"Updated customer balance (deterministic, do NOT recalculate): {balance}\n"
            f"Generate WhatsApp reply:"
        )
        if "anthropic" in model_lower:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "temperature": 0.3,
                "system": REPLY_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}],
            }
        elif "titan" in model_lower:
            prompt = f"{REPLY_SYSTEM_PROMPT}\n\n{user_content}\nReply:"
            return {
                "inputText": prompt,
                "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.3, "stopSequences": ["\n\n"]},
            }
        else:
            return {
                "prompt": f"{REPLY_SYSTEM_PROMPT}\n\n{user_content}\nReply:",
                "max_gen_len": 256,
                "temperature": 0.3,
            }

    def _format_fallback_amount(self, val: Any) -> str:
        try:
            f = float(val)
            return f"₹{int(f) if f.is_integer() else f}"
        except Exception:
            return f"₹{val}"

    def get_fallback_reply(self, extracted: Dict[str, Any], balance: Any, language_code: Optional[str] = None, original_text: Optional[str] = None) -> str:
        name = str(extracted.get("customerName", "Customer")).strip() or "Customer"
        amt = extracted.get("amount", "")
        typ = str(extracted.get("type", "")).strip().upper()
        bal_str = self._format_fallback_amount(balance)
        amt_str = self._format_fallback_amount(amt)
        # Normalize language code; if auto, detect from original_text or name/description
        lang = (language_code or "").strip()
        if not lang or lang.lower() == "auto":
            # Try to detect from original_text, then from name/description
            detect_source = original_text or extracted.get("description") or name or ""
            if detect_source:
                lang = detect_language_from_text(detect_source)
            else:
                lang = "en-IN"
        tmpl_set = FALLBACK_TEMPLATES.get(lang) or FALLBACK_TEMPLATES.get(lang.split("-")[0]) or FALLBACK_TEMPLATES.get(lang.lower()) or FALLBACK_TEMPLATES["en-IN"]
        if typ in tmpl_set:
            return tmpl_set[typ].format(name=name, amt=amt_str, bal=bal_str)
        return f"{name} ke liye {amt_str} ({typ}) record kiya. Kul balance: {bal_str}. ✅"

    def generate_reply(self, extracted: Dict[str, Any], balance: Any, original_text: str, language_code: Optional[str] = None) -> str:
        """
        Generates a human-friendly WhatsApp reply via Bedrock.
        Falls back to deterministic per-language template if Bedrock is unavailable.
        language_code: e.g. hi-IN, ta-IN, bn-IN etc to guide reply language/script.
        """
        if not extracted or not isinstance(extracted, dict):
            raise BedrockExtractionError("extracted transaction is required for reply generation")
        try:
            client = self._get_client()
            payload = self._build_reply_payload(extracted, balance, original_text, language_code=language_code)
            from services.retry_helper import retry_with_backoff, is_aws_transient_error

            def _do_reply_invoke():
                return client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(payload),
                    contentType="application/json",
                    accept="application/json",
                )

            response = retry_with_backoff(
                _do_reply_invoke,
                max_retries=2,
                base_delay=0.2,
                max_delay=2.0,
                is_retryable_fn=is_aws_transient_error,
            )
            raw_body = response.get("body")
            if hasattr(raw_body, "read"):
                response_data = json.loads(raw_body.read().decode("utf-8"))
            elif isinstance(raw_body, (str, bytes)):
                response_data = json.loads(raw_body)
            elif isinstance(raw_body, dict):
                response_data = raw_body
            else:
                response_data = {}
            text = self._extract_text_from_response(response_data).strip()
            if not text:
                raise BedrockExtractionError("Reply generation returned empty response")
            # Sanitize: remove code fences if any, truncate
            if text.startswith("```"):
                text = re.sub(r"^```(?:[\w]+)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return text.strip()[:500]
        except (BedrockUnavailableError, BedrockExtractionError, Exception):
            return self.get_fallback_reply(extracted, balance, language_code=language_code, original_text=original_text)
