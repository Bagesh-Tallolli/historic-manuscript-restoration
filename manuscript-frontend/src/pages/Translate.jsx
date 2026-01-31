import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaLanguage, FaDownload, FaCopy, FaCheckCircle, FaSave } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useManuscript } from '../contexts/ManuscriptContext';
import StepProgressBar from '../components/StepProgressBar';

const Translate = () => {
  const navigate = useNavigate();
  const { extractedText, translations, setTranslations, saveToHistory, resetWorkflow } = useManuscript();
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedLanguages, setSelectedLanguages] = useState({
    english: true,
    hindi: true,
    kannada: true,
  });

  if (!extractedText) {
    navigate('/ocr');
    return null;
  }

  const handleTranslate = async () => {
    setIsTranslating(true);
    toast.info('Translating Sanskrit text...');

    try {
      // Simulate API call to Python backend
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
          text: extractedText,
          languages: selectedLanguages,
        }),
      });

      // For demo purposes, use sample translations
      setTimeout(() => {
        setTranslations({
          english: `Salutations to Lord Ganesha.
Thus begins the Bhagavad Gita.
In the holy field of Kurukshetra, assembled together with the desire to fight,
What did my people and the Pandavas do, O Sanjaya?

You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.
Never consider yourself to be the cause of the results of your activities, nor be attached to inaction.`,

          hindi: `श्री गणेश को नमस्कार।
इस प्रकार श्रीमद्भगवद्गीता का आरंभ होता है।
धर्मक्षेत्र कुरुक्षेत्र में युद्ध की इच्छा से एकत्रित हुए,
हे संजय, मेरे और पांडवों ने क्या किया?

तुम्हें केवल कर्म करने का अधिकार है, फल की आसक्ति का नहीं।
कर्मफल का हेतु मत बनो और न ही अकर्म में आसक्त हो।`,

          kannada: `ಶ್ರೀ ಗಣೇಶನಿಗೆ ನಮಸ್ಕಾರ.
ಹೀಗೆ ಭಗವದ್ಗೀತೆ ಪ್ರಾರಂಭವಾಗುತ್ತದೆ.
ಧರ್ಮಕ್ಷೇತ್ರವಾದ ಕುರುಕ್ಷೇತ್ರದಲ್ಲಿ ಯುದ್ಧದ ಇಚ್ಛೆಯಿಂದ ಒಟ್ಟುಗೂಡಿದ,
ಓ ಸಂಜಯ, ನನ್ನವರು ಮತ್ತು ಪಾಂಡವರು ಏನು ಮಾಡಿದರು?

ನಿನಗೆ ಕರ್ತವ್ಯ ಮಾಡುವ ಹಕ್ಕಿದೆ, ಆದರೆ ಫಲದ ಮೇಲೆ ಹಕ್ಕಿಲ್ಲ.
ಕರ್ಮಫಲಕ್ಕೆ ಕಾರಣವೆಂದು ನಿನ್ನನ್ನು ಎಂದಿಗೂ ಭಾವಿಸಬೇಡ, ಅಕರ್ಮದಲ್ಲಿ ಆಸಕ್ತನಾಗಬೇಡ.`,
        });
        toast.success('Translation completed successfully!');
        setIsTranslating(false);
      }, 3000);
    } catch (error) {
      toast.error('Translation failed. Please try again.');
      setIsTranslating(false);
    }
  };

  const handleCopyTranslation = (text, language) => {
    navigator.clipboard.writeText(text);
    toast.success(`${language} translation copied to clipboard!`);
  };

  const handleDownloadAll = () => {
    const content = `
Sanskrit Manuscript Translation
================================

Original Sanskrit Text:
${extractedText}

English Translation:
${translations.english}

हिंदी अनुवाद:
${translations.hindi}

ಕನ್ನಡ ಅನುವಾದ:
${translations.kannada}

Generated on: ${new Date().toLocaleString()}
    `;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'manuscript-translation.txt';
    link.click();
    toast.success('Translation downloaded!');
  };

  const handleSaveToHistory = () => {
    saveToHistory();
    toast.success('Saved to archive!');
    setTimeout(() => {
      resetWorkflow();
      navigate('/');
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        <StepProgressBar />

        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-heritage-600 mb-3">Translation</h1>
          <p className="text-lg text-heritage-400">
            Step 4 of 4: Translate Sanskrit text to multiple languages
          </p>
        </div>

        {/* Original Sanskrit Text */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold text-heritage-600 mb-4 flex items-center">
            <FaLanguage className="mr-3 text-saffron-500" />
            Original Sanskrit Text
          </h2>
          <div className="sanskrit-text font-devanagari text-xl leading-loose">
            {extractedText}
          </div>
        </div>

        {/* Language Selection & Translate Button */}
        {!translations.english && !isTranslating && (
          <div className="card mb-8">
            <h2 className="text-xl font-bold text-heritage-600 mb-4">Select Translation Languages</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedLanguages.english}
                  onChange={(e) => setSelectedLanguages({ ...selectedLanguages, english: e.target.checked })}
                  className="w-5 h-5 text-saffron-500 rounded"
                />
                <span className="text-heritage-600 font-medium">English</span>
              </label>
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedLanguages.hindi}
                  onChange={(e) => setSelectedLanguages({ ...selectedLanguages, hindi: e.target.checked })}
                  className="w-5 h-5 text-saffron-500 rounded"
                />
                <span className="text-heritage-600 font-medium">हिंदी (Hindi)</span>
              </label>
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedLanguages.kannada}
                  onChange={(e) => setSelectedLanguages({ ...selectedLanguages, kannada: e.target.checked })}
                  className="w-5 h-5 text-saffron-500 rounded"
                />
                <span className="text-heritage-600 font-medium">ಕನ್ನಡ (Kannada)</span>
              </label>
            </div>
            <div className="text-center">
              <button
                onClick={handleTranslate}
                className="btn-primary inline-flex items-center space-x-2"
              >
                <FaLanguage />
                <span>Translate Text</span>
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isTranslating && (
          <div className="card text-center py-12">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-saffron-400 border-t-transparent mb-4"></div>
            <p className="text-lg text-heritage-600 font-medium">Translating using AI...</p>
            <p className="text-sm text-heritage-400 mt-2">Processing multiple language translations</p>
          </div>
        )}

        {/* Translation Results */}
        {translations.english && !isTranslating && (
          <>
            <div className="space-y-6 mb-8">
              {/* English Translation */}
              {selectedLanguages.english && translations.english && (
                <div className="card border-l-4 border-saffron-400">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-2xl font-bold text-heritage-600">English Translation</h3>
                    <button
                      onClick={() => handleCopyTranslation(translations.english, 'English')}
                      className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                    >
                      <FaCopy />
                      <span>Copy</span>
                    </button>
                  </div>
                  <p className="text-lg leading-relaxed text-heritage-700 text-justify">
                    {translations.english}
                  </p>
                </div>
              )}

              {/* Hindi Translation */}
              {selectedLanguages.hindi && translations.hindi && (
                <div className="card border-l-4 border-saffron-400">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-2xl font-bold text-heritage-600">हिंदी अनुवाद</h3>
                    <button
                      onClick={() => handleCopyTranslation(translations.hindi, 'Hindi')}
                      className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                    >
                      <FaCopy />
                      <span>Copy</span>
                    </button>
                  </div>
                  <p className="font-devanagari text-lg leading-relaxed text-heritage-700 text-justify">
                    {translations.hindi}
                  </p>
                </div>
              )}

              {/* Kannada Translation */}
              {selectedLanguages.kannada && translations.kannada && (
                <div className="card border-l-4 border-saffron-400">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-2xl font-bold text-heritage-600">ಕನ್ನಡ ಅನುವಾದ</h3>
                    <button
                      onClick={() => handleCopyTranslation(translations.kannada, 'Kannada')}
                      className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                    >
                      <FaCopy />
                      <span>Copy</span>
                    </button>
                  </div>
                  <p className="font-kannada text-lg leading-relaxed text-heritage-700 text-justify">
                    {translations.kannada}
                  </p>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <button
                onClick={handleDownloadAll}
                className="btn-secondary inline-flex items-center justify-center space-x-2"
              >
                <FaDownload />
                <span>Download All Translations</span>
              </button>
              <button
                onClick={handleSaveToHistory}
                className="btn-primary inline-flex items-center justify-center space-x-2"
              >
                <FaSave />
                <span>Save to Archive & Complete</span>
                <FaCheckCircle />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Translate;

