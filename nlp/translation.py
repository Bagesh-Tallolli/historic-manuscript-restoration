"""
Sanskrit to English translation
"""

import os
import re

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Warning: googletrans not available")


class SanskritTranslator:
    """Translate Sanskrit (Devanagari) to English"""

    def __init__(self, method='indictrans', device='auto'):
        """
        Args:
            method: 'indictrans', 'google', or 'ensemble'
            device: 'cuda', 'cpu', or 'auto'
        """
        self.method = method

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize translation models
        if method == 'indictrans' or method == 'ensemble':
            self._init_indictrans()

        if method == 'google' or method == 'ensemble':
            self._init_google()

    def _init_indictrans(self):
        """Initialize IndicTrans2 model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, cannot use IndicTrans")
            self.indictrans_model = None
            return

        try:
            print("Loading IndicTrans2 model...")

            # IndicTrans2 for Indic languages
            model_name = "ai4bharat/indictrans2-en-indic-1B"

            self.indictrans_tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.indictrans_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)

            print(f"IndicTrans2 loaded on {self.device}")
        except Exception as e:
            print(f"Could not load IndicTrans2: {e}")
            print("Trying alternative model...")

            try:
                # Fallback to mBART or other multilingual model
                model_name = "facebook/mbart-large-50-many-to-many-mmt"
                self.indictrans_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.indictrans_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name
                ).to(self.device)
                print(f"Loaded fallback model: {model_name}")
            except Exception as e2:
                print(f"Could not load fallback model: {e2}")
                self.indictrans_model = None

    def _init_google(self):
        """Initialize Google Translate"""
        if not GOOGLETRANS_AVAILABLE:
            print("Warning: googletrans not available")
            self.google_translator = None
            return

        try:
            self.google_translator = GoogleTranslator()
            print("Google Translate initialized")
        except Exception as e:
            print(f"Could not initialize Google Translate: {e}")
            self.google_translator = None

    def translate(self, text, source_lang='san', target_lang='en'):
        """
        Translate Sanskrit text to English

        Args:
            text: Sanskrit text (Devanagari script)
            source_lang: Source language code ('san' for Sanskrit)
            target_lang: Target language code ('en' for English)

        Returns:
            Translated English text
        """
        if not text or not text.strip():
            return ""

        if self.method == 'indictrans':
            return self._translate_indictrans(text, source_lang, target_lang)
        elif self.method == 'google':
            return self._translate_google(text, source_lang, target_lang)
        elif self.method == 'ensemble':
            return self._translate_ensemble(text, source_lang, target_lang)
        else:
            raise ValueError(f"Unknown translation method: {self.method}")

    def _translate_indictrans(self, text, source_lang='san', target_lang='en'):
        """Translate using IndicTrans2"""
        if self.indictrans_model is None:
            print("IndicTrans model not available, using fallback")
            return self._translate_google(text, source_lang, target_lang)

        try:
            # Prepare input
            inputs = self.indictrans_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.indictrans_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode
            translation = self.indictrans_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            return translation.strip()

        except Exception as e:
            print(f"IndicTrans translation error: {e}")
            return self._translate_google(text, source_lang, target_lang)

    def _translate_google(self, text, source_lang='sa', target_lang='en'):
        """Translate using Google Translate API"""
        if self.google_translator is None:
            print("Google Translate not available")
            return text  # Return original if no translation available

        try:
            # Google Translate uses 'sa' for Sanskrit
            result = self.google_translator.translate(
                text,
                src=source_lang if source_lang != 'san' else 'sa',
                dest=target_lang
            )
            return result.text

        except Exception as e:
            print(f"Google Translate error: {e}")
            return text

    def _translate_ensemble(self, text, source_lang='san', target_lang='en'):
        """Combine multiple translation methods"""
        translations = []

        # Try IndicTrans
        if self.indictrans_model is not None:
            try:
                trans = self._translate_indictrans(text, source_lang, target_lang)
                if trans and trans != text:
                    translations.append(trans)
            except:
                pass

        # Try Google
        if self.google_translator is not None:
            try:
                trans = self._translate_google(text, source_lang, target_lang)
                if trans and trans != text:
                    translations.append(trans)
            except:
                pass

        if not translations:
            return text

        # For now, return first successful translation
        # Could implement voting or combination logic here
        return translations[0]

    def translate_sentences(self, sentences, source_lang='san', target_lang='en'):
        """
        Translate multiple sentences

        Args:
            sentences: List of sentences

        Returns:
            List of translated sentences
        """
        return [self.translate(sent, source_lang, target_lang) for sent in sentences]

    def translate_with_context(self, text, context=None):
        """
        Translate with additional context

        Args:
            text: Text to translate
            context: Additional context to help translation

        Returns:
            Translation
        """
        if context:
            # Prepend context (model-dependent)
            full_text = f"{context}\n{text}"
            translation = self.translate(full_text)
            # Extract relevant part (simple heuristic)
            lines = translation.split('\n')
            return lines[-1] if len(lines) > 1 else translation
        else:
            return self.translate(text)


class BackTranslation:
    """Back-translation for quality checking"""

    def __init__(self, device='auto'):
        self.en_to_san = SanskritTranslator(method='google', device=device)
        self.san_to_en = SanskritTranslator(method='google', device=device)

    def check_quality(self, original_sanskrit, english_translation):
        """
        Check translation quality via back-translation

        Args:
            original_sanskrit: Original Sanskrit text
            english_translation: English translation

        Returns:
            Dictionary with quality metrics
        """
        # Translate English back to Sanskrit
        back_translated = self.en_to_san.translate(
            english_translation, source_lang='en', target_lang='san'
        )

        # Simple similarity check (character overlap)
        similarity = self._calculate_similarity(original_sanskrit, back_translated)

        return {
            'back_translation': back_translated,
            'similarity_score': similarity,
            'quality': 'good' if similarity > 0.7 else 'fair' if similarity > 0.4 else 'poor'
        }

    def _calculate_similarity(self, text1, text2):
        """Simple character-based similarity"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.replace(' ', ''))
        set2 = set(text2.replace(' ', ''))

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


def translate_document(text, method='indictrans'):
    """
    Convenient function to translate a Sanskrit document

    Args:
        text: Sanskrit text (can be multi-line)
        method: Translation method

    Returns:
        English translation
    """
    translator = SanskritTranslator(method=method)

    # Split into sentences if needed
    if '\n' in text or '।' in text or '॥' in text:
        # Split on Devanagari punctuation or newlines
        sentences = re.split(r'[।॥\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Translate each sentence
        translations = translator.translate_sentences(sentences)

        # Combine
        return ' '.join(translations)
    else:
        return translator.translate(text)


if __name__ == "__main__":
    print("Testing Sanskrit Translation...")

    # Test cases
    test_texts = [
        "रामः वनं गच्छति",
        "सत्यं वद धर्मं चर",
        "अहम् भारतीयः अस्मि",
    ]

    # Test with available methods
    methods = ['google', 'indictrans']

    for method in methods:
        try:
            print(f"\n--- Testing {method} ---")
            translator = SanskritTranslator(method=method)

            for text in test_texts:
                translation = translator.translate(text)
                print(f"Sanskrit: {text}")
                print(f"English:  {translation}\n")
        except Exception as e:
            print(f"Could not test {method}: {e}")

    print("\nTranslation module ready!")

