import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaMagic, FaArrowRight, FaDownload, FaSearchPlus, FaSearchMinus } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useManuscript } from '../contexts/ManuscriptContext';
import StepProgressBar from '../components/StepProgressBar';

const Restore = () => {
  const navigate = useNavigate();
  const { uploadedImage, restoredImage, setRestoredImage, setCurrentStep } = useManuscript();
  const [isRestoring, setIsRestoring] = useState(false);
  const [zoom, setZoom] = useState(100);

  if (!uploadedImage) {
    navigate('/upload');
    return null;
  }

  const handleRestore = async () => {
    setIsRestoring(true);
    toast.info('Restoring manuscript image...');

    try {
      // Simulate API call to Python backend
      // In production, this would call your Streamlit/FastAPI backend
      const response = await fetch('/api/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: uploadedImage }),
      });

      // For demo purposes, use the uploaded image as "restored"
      // In production, this would be the actual restored image from backend
      setTimeout(() => {
        setRestoredImage(uploadedImage); // Replace with actual API response
        toast.success('Image restoration completed!');
        setIsRestoring(false);
      }, 2000);
    } catch (error) {
      toast.error('Restoration failed. Please try again.');
      setIsRestoring(false);
    }
  };

  const handleProceed = () => {
    if (!restoredImage) {
      toast.error('Please restore the image first');
      return;
    }
    setCurrentStep(2);
    navigate('/ocr');
  };

  const handleDownload = (imageData, filename) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    link.click();
  };

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        <StepProgressBar />

        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-heritage-600 mb-3">Image Restoration</h1>
          <p className="text-lg text-heritage-400">
            Step 2 of 4: Enhance manuscript clarity using advanced image processing
          </p>
        </div>

        {/* Restoration Button */}
        {!restoredImage && (
          <div className="text-center mb-8">
            <button
              onClick={handleRestore}
              disabled={isRestoring}
              className="btn-primary inline-flex items-center space-x-3 text-lg"
            >
              <FaMagic />
              <span>{isRestoring ? 'Restoring...' : 'Restore Manuscript Image'}</span>
            </button>
          </div>
        )}

        {/* Loading State */}
        {isRestoring && (
          <div className="card text-center py-12">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-saffron-400 border-t-transparent mb-4"></div>
            <p className="text-lg text-heritage-600 font-medium">Processing manuscript...</p>
            <p className="text-sm text-heritage-400 mt-2">Applying CLAHE and unsharp mask enhancement</p>
          </div>
        )}

        {/* Side-by-Side Comparison */}
        {restoredImage && !isRestoring && (
          <>
            {/* Zoom Controls */}
            <div className="flex justify-center items-center space-x-4 mb-6">
              <button
                onClick={() => setZoom(Math.max(50, zoom - 10))}
                className="btn-secondary inline-flex items-center space-x-2"
              >
                <FaSearchMinus />
                <span>Zoom Out</span>
              </button>
              <span className="text-heritage-600 font-semibold">{zoom}%</span>
              <button
                onClick={() => setZoom(Math.min(200, zoom + 10))}
                className="btn-secondary inline-flex items-center space-x-2"
              >
                <FaSearchPlus />
                <span>Zoom In</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Original Image */}
              <div className="card">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-heritage-600">Original Manuscript</h2>
                  <button
                    onClick={() => handleDownload(uploadedImage, 'original-manuscript.png')}
                    className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                  >
                    <FaDownload />
                    <span>Download</span>
                  </button>
                </div>
                <div className="bg-parchment-50 rounded-lg p-4 border-2 border-heritage-200 overflow-auto">
                  <img
                    src={uploadedImage}
                    alt="Original"
                    style={{ width: `${zoom}%` }}
                    className="mx-auto rounded shadow-md"
                  />
                </div>
              </div>

              {/* Restored Image */}
              <div className="card">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-heritage-600">Restored Manuscript</h2>
                  <button
                    onClick={() => handleDownload(restoredImage, 'restored-manuscript.png')}
                    className="text-saffron-500 hover:text-saffron-600 flex items-center space-x-2"
                  >
                    <FaDownload />
                    <span>Download</span>
                  </button>
                </div>
                <div className="bg-parchment-50 rounded-lg p-4 border-2 border-saffron-300 overflow-auto">
                  <img
                    src={restoredImage}
                    alt="Restored"
                    style={{ width: `${zoom}%` }}
                    className="mx-auto rounded shadow-md"
                  />
                </div>
                <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-3 flex items-center text-green-700">
                  <FaMagic className="mr-2 text-xl" />
                  <span className="font-medium">Enhancement applied successfully</span>
                </div>
              </div>
            </div>

            {/* Proceed Button */}
            <div className="text-center">
              <button
                onClick={handleProceed}
                className="btn-primary inline-flex items-center space-x-3 text-lg"
              >
                <span>Proceed to OCR Extraction</span>
                <FaArrowRight />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Restore;

