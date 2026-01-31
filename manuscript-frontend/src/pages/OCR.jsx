import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaBookOpen, FaCopy, FaArrowRight } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useManuscript } from '../contexts/ManuscriptContext';
import StepProgressBar from '../components/StepProgressBar';

const OCR = () => {
  const navigate = useNavigate();
  const { restoredImage, extractedText, setExtractedText } = useManuscript();
  const [isExtracting, setIsExtracting] = useState(false);

  if (!restoredImage) {
    navigate('/restore');
    return null;
  }

  const handleExtractText = async () => {
    setIsExtracting(true);
    toast.info('Extracting Sanskrit text from manuscript...');

    try {
      // Simulate API call to Python backend
      const response = await fetch('/api/ocr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: restoredImage,
        }),
      });

      // For demo purposes, use sample Sanskrit text
      setTimeout(() => {
        const sampleText = `श्रीगणेशाय नमः।

अथ श्रीमद्भगवद्गीतासु उपनिषत्सु ब्रह्मविद्यायां योगशास्त्रे 
श्रीकृष्णार्जुनसंवादे अर्जुनविषादयोगो नाम प्रथमोऽध्यायः॥

धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥१॥

कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।
मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥४७॥`;

        setExtractedText(sampleText);
        toast.success('Sanskrit text extracted successfully!');
        setIsExtracting(false);
      }, 3000);
    } catch (error) {
      toast.error('OCR extraction failed. Please try again.');
      setIsExtracting(false);
    }
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(extractedText);
    toast.success('Text copied to clipboard!');
  };

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        <StepProgressBar />

        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-heritage-600 mb-3">OCR Extraction</h1>
          <p className="text-lg text-heritage-400">
            Step 3 of 4: Extract Sanskrit text using AI-powered OCR
          </p>
        </div>

        {/* Restored Image Preview */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold text-heritage-600 mb-4 flex items-center">
            <FaBookOpen className="mr-3 text-saffron-500" />
            Restored Manuscript
          </h2>
          <div className="flex justify-center">
            <img
              src={restoredImage}
              alt="Restored Manuscript"
              className="max-w-full h-auto rounded-lg shadow-xl border-2 border-heritage-200"
            />
          </div>
        </div>

        {/* Extract Button */}
        {!extractedText && !isExtracting && (
          <div className="text-center mb-8">
            <button
              onClick={handleExtractText}
              className="btn-primary inline-flex items-center space-x-2 text-lg"
            >
              <FaBookOpen />
              <span>Extract Sanskrit Text (OCR)</span>
            </button>
          </div>
        )}

        {/* Loading State */}
        {isExtracting && (
          <div className="card text-center py-12">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-saffron-400 border-t-transparent mb-4"></div>
            <p className="text-lg text-heritage-600 font-medium">Extracting text using Tesseract OCR...</p>
            <p className="text-sm text-heritage-400 mt-2">Processing manuscript image</p>
          </div>
        )}

        {/* Extracted Text Display */}
        {extractedText && !isExtracting && (
          <>
            <div className="card mb-8">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-heritage-600">Extracted Sanskrit Text</h2>
                <button
                  onClick={handleCopyText}
                  className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                >
                  <FaCopy />
                  <span>Copy Text</span>
                </button>
              </div>

              <div className="sanskrit-text font-devanagari text-xl leading-loose">
                {extractedText}
              </div>
            </div>

            {/* Script Information */}
            <div className="card mb-8 bg-gradient-to-br from-saffron-50 to-parchment-50">
              <h3 className="text-xl font-bold text-heritage-600 mb-3">OCR Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-heritage-500 mb-1">Script Detected</p>
                  <p className="text-lg font-semibold text-heritage-700">Devanagari</p>
                </div>
                <div>
                  <p className="text-sm text-heritage-500 mb-1">Language</p>
                  <p className="text-lg font-semibold text-heritage-700">Sanskrit</p>
                </div>
                <div>
                  <p className="text-sm text-heritage-500 mb-1">OCR Engine</p>
                  <p className="text-lg font-semibold text-heritage-700">Tesseract OCR</p>
                </div>
              </div>
            </div>

            {/* Next Step Button */}
            <div className="text-center">
              <button
                onClick={() => navigate('/translate')}
                className="btn-primary inline-flex items-center space-x-2 text-lg"
              >
                <span>Proceed to Translation</span>
                <FaArrowRight />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default OCR;

